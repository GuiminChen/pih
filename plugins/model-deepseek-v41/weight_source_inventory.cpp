#include "weight_source_inventory.h"
#include "pih/model/safetensors_shard_index.h"
#include <algorithm>
#include <new>

namespace pih::deepseek_v41 {
namespace {
Result<std::string> Normalize(std::string_view name) {
  if (name.empty() || name.size() > 512)
    return Status::InvalidArgument("V4.1 source tensor name exceeds bounds");
  if (name.starts_with("model.")) name.remove_prefix(6);
  const bool vision = name.starts_with("vision.");
  std::string result;
  while (!name.empty()) {
    const auto end = name.find('.');
    auto part = name.substr(0, end);
    if (part.empty()) return Status::InvalidArgument("V4.1 source tensor name has empty segment");
    for (const unsigned char c : part)
      if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_'))
        return Status::InvalidArgument("V4.1 source tensor name contains unsupported characters");
    if (part == "self_attn") part = "attn";
    else if (part == "mlp" && !vision) part = "ffn";
    else if (part == "weight_scale_inv") part = "scale";
    else if (part == "e_score_correction_bias") part = "bias";
    if (!result.empty()) result += '.';
    result += part;
    if (end == std::string_view::npos) break;
    name.remove_prefix(end + 1);
    if (name.empty()) return Status::InvalidArgument("V4.1 source tensor name ends in separator");
  }
  if (result.empty()) return Status::InvalidArgument("V4.1 normalized tensor name is empty");
  return result;
}
}
Result<WeightSourceInventory> WeightSourceInventory::Create(std::string_view json,
    const Sha256Digest& expected, std::span<const WeightSourceFile* const> files) {
  if (json.empty() || json.size() > SafetensorsShardIndex::kMaximumIndexBytes ||
      expected == Sha256Digest{} || files.empty() || files.size() > 256)
    return Status::InvalidArgument("V4.1 source inventory index/digest/shard bound invalid");
  try {
    auto hash = sha256(std::as_bytes(std::span(json.data(), json.size())));
    if (!hash.ok()) return hash.status();
    if (*hash != expected) return Status::FailedPrecondition("V4.1 source index differs from trusted digest");
    auto index = SafetensorsShardIndex::Parse(json);
    if (!index.ok()) return index.status();
    if (index->shard_names().size() != files.size() || index->bindings().size() > 200000)
      return Status::InvalidArgument("V4.1 source inventory shard count differs or tensor budget exceeded");
    std::vector<bool> seen(index->bindings().size(), false);
    WeightSourceInventory result;
    result.bindings_.reserve(index->bindings().size());
    std::uint64_t total = 0;
    for (std::size_t shard = 0; shard < files.size(); ++shard) {
      const auto* file = files[shard];
      if (!file) return Status::InvalidArgument("V4.1 source inventory has null shard");
      for (std::size_t prior = 0; prior < shard; ++prior)
        if (files[prior]->member_name() == file->member_name())
          return Status::InvalidArgument("V4.1 source inventory has duplicate shard");
      if (!std::binary_search(index->shard_names().begin(), index->shard_names().end(), file->member_name()))
        return Status::InvalidArgument("V4.1 source shard absent from index");
      auto status = file->Revalidate(); if (!status.ok()) return status;
      if (total > index->total_size() || file->header().data_bytes() > index->total_size() - total)
        return Status::InvalidArgument("V4.1 source payload exceeds index total_size");
      total += file->header().data_bytes();
      for (const auto& tensor : file->header().tensors()) {
        const auto& bindings = index->bindings();
        const auto found = std::lower_bound(bindings.begin(), bindings.end(), tensor.name,
            [](const auto& binding, auto name) { return binding.tensor_name < name; });
        if (found == bindings.end() || found->tensor_name != tensor.name ||
            found->shard_name != file->member_name())
          return Status::InvalidArgument("V4.1 source tensor absent from index or stored in wrong shard");
        const auto slot = static_cast<std::size_t>(found - bindings.begin());
        if (seen[slot]) return Status::InvalidArgument("V4.1 source tensor appears more than once");
        seen[slot] = true;
        auto name = Normalize(tensor.name); if (!name.ok()) return name.status();
        result.bindings_.push_back({std::move(*name), tensor.name, static_cast<std::uint32_t>(shard)});
      }
    }
    if (total != index->total_size() || std::find(seen.begin(), seen.end(), false) != seen.end())
      return Status::InvalidArgument("V4.1 source index tensor set or total_size incomplete");
    std::sort(result.bindings_.begin(), result.bindings_.end(), [](const auto& a, const auto& b) {
      return a.canonical_name < b.canonical_name;
    });
    for (std::size_t i = 1; i < result.bindings_.size(); ++i)
      if (result.bindings_[i - 1].canonical_name == result.bindings_[i].canonical_name)
        return Status::InvalidArgument("V4.1 source aliases collide after normalization");
    for (const auto* file : files) {
      auto status = file->Revalidate(); if (!status.ok()) return status;
    }
    return result;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 source inventory allocation failed");
  }
}
const SourceNameBinding* WeightSourceInventory::Find(std::string_view name) const noexcept {
  const auto found = std::lower_bound(bindings_.begin(), bindings_.end(), name,
      [](const auto& binding, auto key) { return binding.canonical_name < key; });
  return found == bindings_.end() || found->canonical_name != name ? nullptr : &*found;
}
}  // namespace pih::deepseek_v41
