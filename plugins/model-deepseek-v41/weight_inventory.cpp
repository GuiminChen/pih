#include "weight_inventory.h"
#include <algorithm>
#include <new>

namespace pih::deepseek_v41 {
Result<BackboneWeightInventory> BackboneWeightInventory::Create(
    const FlashConfig& config, std::uint32_t world, std::uint32_t rank) {
  if (config.config_sha256() == Sha256Digest{} ||
      (world != 1 && world != 2 && world != 4 && world != 8) || rank >= world)
    return Status::InvalidArgument("V4.1 backbone inventory requires admitted config and TP1/2/4/8");
  try {
    BackboneWeightInventory result;
    result.weights_.reserve(40ULL * (384 / world * 6 + 40) + 32);
    auto add = [&](std::string name, WeightStorage storage, std::uint64_t rows,
                   std::uint64_t columns, int axis = -1, bool vector = false) {
      const auto extent = axis == 0 ? rows : columns;
      const auto local = axis < 0 ? 0 : extent / world;
      if (axis == 0) rows = local;
      if (axis == 1) columns = local;
      const std::uint64_t element = storage == WeightStorage::kF32 ? 4 :
          storage == WeightStorage::kBF16 ? 2 : 1;
      const auto bytes = rows * columns * element;
      result.weights_.push_back({{std::move(name), storage, vector ? 1U : 2U,
                                  rows, columns, bytes}, axis, rank * local, local, 0});
      result.bytes_ += bytes;
    };
    auto vec = [&](const std::string& name, std::uint64_t rows,
                   WeightStorage storage = WeightStorage::kBF16, int axis = -1) {
      add(name, storage, rows, 1, axis, true);
    };
    auto fp8 = [&](const std::string& name, std::uint64_t out,
                   std::uint64_t in, int axis = -1) {
      add(name + ".weight", WeightStorage::kE4M3FN, out, in, axis);
      add(name + ".scale", WeightStorage::kE8M0, out / 32, in / 32, axis);
    };
    auto expert = [&](const std::string& name, bool packed) {
      for (const auto* suffix : {"w1", "w2", "w3"}) {
        const bool down = std::string_view(suffix) == "w2";
        const std::uint64_t out = down ? 5120 : 2304, in = down ? 2304 : 5120;
        const auto prefix = name + "." + suffix;
        if (!packed) { fp8(prefix, out, in); continue; }
        add(prefix + ".weight", WeightStorage::kPackedE2M1, out, in / 2);
        add(prefix + ".scale", WeightStorage::kE8M0, out, in / 32);
      }
    };
    add("embed.weight", WeightStorage::kBF16, 129280, 5120, 0);
    vec("norm.weight", 5120);
    add("head.weight", WeightStorage::kF32, 129280, 5120, 0);
    for (std::uint32_t layer = 0; layer < FlashConfig::kMainLayers; ++layer) {
      const auto p = "layers." + std::to_string(layer) + ".";
      const auto& sharing = config.attention_sharing()[layer];
      vec(p + "attn_norm.weight", 5120);
      vec(p + "ffn_norm.weight", 5120);
      for (const auto* branch : {"attn", "ffn"}) {
        const auto hc = p + "hc_" + branch;
        add(hc + "_fn", WeightStorage::kF32, 24, 20480);
        vec(hc + "_base", 24, WeightStorage::kF32);
        vec(hc + "_scale", 3, WeightStorage::kF32);
      }
      const auto a = p + "attn.";
      vec(a + "attn_sink", 64, WeightStorage::kF32, 0);
      fp8(a + "wq_a", 1280, 5120);
      vec(a + "q_norm.weight", 1280);
      fp8(a + "wq_b", 32768, 1280, 0);
      fp8(a + "wkv", 512, 5120);
      vec(a + "kv_norm.weight", 512);
      add(a + "wo_a.weight", WeightStorage::kBF16, 8192, 4096, 0);
      fp8(a + "wo_b", 5120, 8192, 1);
      if (sharing.owns_kv) {
        const bool pool = sharing.compression_ratio > 1;
        add(a + "compressor.wkv.weight", pool ? WeightStorage::kF32 : WeightStorage::kBF16, 512, 5120);
        if (pool) add(a + "compressor.wgate.weight", WeightStorage::kF32, 512, 5120);
        vec(a + "compressor.norm.weight", 512);
      }
      if (sharing.owns_index) {
        fp8(a + "indexer.wq_b", 4096, 1280, 0);
        add(a + "indexer.weights_proj.weight", WeightStorage::kBF16, 32, 5120, 0);
        if (sharing.owns_kv) {
          add(a + "indexer.wk.weight", WeightStorage::kBF16, 128, 512);
          vec(a + "indexer.k_norm.weight", 128);
        }
      }
      add(p + "ffn.gate.weight", WeightStorage::kF32, 384, 5120);
      vec(p + "ffn.gate.bias", 384, WeightStorage::kF32);
      expert(p + "ffn.shared_experts", false);
      const auto count = 384 / world;
      for (std::uint32_t id = rank * count; id < (rank + 1) * count; ++id)
        expert(p + "ffn.experts." + std::to_string(id), true);
      if (sharing.has_engram) {
        auto engram = EngramWeightPlan::Create(config, layer, world, rank, EngramStorage::kBF16);
        if (!engram.ok()) return engram.status();
        for (const auto& m : engram.value().matrices()) {
          const auto storage = m.storage == EngramStorage::kE4M3FN ? WeightStorage::kE4M3FN :
              m.storage == EngramStorage::kE8M0 ? WeightStorage::kE8M0 : WeightStorage::kBF16;
          const bool table = m.name.starts_with("embed.");
          result.weights_.push_back({{p + "engram." + std::string(m.name), storage,
              2, m.rows, m.columns, m.bytes}, table ? 0 : -1,
              table ? engram.value().first_row() : 0,
              table ? engram.value().valid_rows() : 0,
              table ? engram.value().padded_rows() : 0});
          result.bytes_ += m.bytes;
        }
      }
    }
    std::sort(result.weights_.begin(), result.weights_.end(), [](const auto& a, const auto& b) {
      return a.tensor.name < b.tensor.name;
    });
    return result;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 backbone inventory allocation failed");
  }
}

Status BackboneWeightInventory::Validate(std::span<const RuntimeWeight> supplied) const {
  if (supplied.size() != weights_.size())
    return Status::InvalidArgument("V4.1 backbone weight member count differs");
  try {
    std::vector<bool> seen(weights_.size(), false);
    for (const auto& actual : supplied) {
      auto it = std::lower_bound(weights_.begin(), weights_.end(), actual.name,
          [](const auto& a, const auto& name) { return a.tensor.name < name; });
      if (it == weights_.end() || it->tensor.name != actual.name)
        return Status::InvalidArgument("Unknown V4.1 backbone runtime weight");
      const auto slot = static_cast<std::size_t>(it - weights_.begin());
      if (seen[slot]) return Status::InvalidArgument("Duplicate V4.1 backbone runtime weight");
      seen[slot] = true;
      const auto& expected = it->tensor;
      if (actual.storage != expected.storage || actual.dimensions != expected.dimensions ||
          actual.rows != expected.rows || actual.columns != expected.columns || actual.bytes != expected.bytes)
        return Status::InvalidArgument("V4.1 backbone runtime weight geometry/storage differs");
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 backbone weight validation allocation failed");
  }
}
}  // namespace pih::deepseek_v41
