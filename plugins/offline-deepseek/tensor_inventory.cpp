#include "tensor_inventory.h"
#include <algorithm>
#include <string_view>
#include <charconv>
#include <array>

namespace pih::offline_deepseek {
Status ValidatePp1DenseGeometry(std::string_view name, DType dtype,
                              std::span<const std::uint64_t> shape) {
  struct Rule { std::string_view name; DType dtype; std::uint64_t rows, columns; };
  // Direct checkpoint bindings, not runtime-only fused projection names.
  static constexpr Rule endpoint[]{
      {"embed.weight", DType::kBFloat16, 129280, 4096},
      {"head.weight", DType::kBFloat16, 129280, 4096},
      {"norm.weight", DType::kBFloat16, 4096, 0},
      {"hc_head_fn", DType::kFloat32, 4, 16384},
      {"hc_head_scale", DType::kFloat32, 1, 0},
      {"hc_head_base", DType::kFloat32, 4, 0}};
  static constexpr Rule layer[]{
      {"attn_norm.weight", DType::kBFloat16, 4096, 0},
      {"ffn_norm.weight", DType::kBFloat16, 4096, 0},
      {"hc_attn_fn", DType::kFloat32, 24, 16384},
      {"hc_ffn_fn", DType::kFloat32, 24, 16384},
      {"hc_attn_scale", DType::kFloat32, 3, 0},
      {"hc_ffn_scale", DType::kFloat32, 3, 0},
      {"hc_attn_base", DType::kFloat32, 24, 0},
      {"hc_ffn_base", DType::kFloat32, 24, 0},
      {"ffn.gate.weight", DType::kBFloat16, 256, 4096},
      {"attn.q_norm.weight", DType::kBFloat16, 1024, 0},
      {"attn.kv_norm.weight", DType::kBFloat16, 512, 0},
      {"attn.attn_sink", DType::kFloat32, 64, 0},
      {"attn.wq_a.weight", DType::kFloat8E4M3, 1024, 4096},
      {"attn.wq_a.scale", DType::kFloat8E8M0, 8, 32},
      {"attn.wq_b.weight", DType::kFloat8E4M3, 32768, 1024},
      {"attn.wq_b.scale", DType::kFloat8E8M0, 256, 8},
      {"attn.wkv.weight", DType::kFloat8E4M3, 512, 4096},
      {"attn.wkv.scale", DType::kFloat8E8M0, 4, 32},
      {"attn.wo_a.weight", DType::kFloat8E4M3, 8192, 4096},
      {"attn.wo_a.scale", DType::kFloat8E8M0, 64, 32},
      {"attn.wo_b.weight", DType::kFloat8E4M3, 4096, 8192},
      {"attn.wo_b.scale", DType::kFloat8E8M0, 32, 64},
      {"ffn.gate.bias", DType::kFloat32, 256, 0},
      {"ffn.gate.tid2eid", DType::kInt64, 129280, 6},
      {"ffn.shared_experts.w1.weight", DType::kFloat8E4M3, 2048, 4096},
      {"ffn.shared_experts.w3.weight", DType::kFloat8E4M3, 2048, 4096},
      {"ffn.shared_experts.w2.weight", DType::kFloat8E4M3, 4096, 2048},
      {"ffn.shared_experts.w1.scale", DType::kFloat8E8M0, 16, 32},
      {"ffn.shared_experts.w3.scale", DType::kFloat8E8M0, 16, 32},
      {"ffn.shared_experts.w2.scale", DType::kFloat8E8M0, 32, 16},
      {"attn.indexer.wq_b.weight", DType::kFloat8E4M3, 8192, 1024},
      {"attn.indexer.wq_b.scale", DType::kFloat8E8M0, 64, 8},
      {"attn.indexer.weights_proj.weight", DType::kBFloat16, 64, 4096},
      {"attn.indexer.compressor.ape", DType::kFloat32, 4, 256},
      {"attn.indexer.compressor.norm.weight", DType::kBFloat16, 128, 0},
      {"attn.indexer.compressor.wkv.weight", DType::kBFloat16, 256, 4096},
      {"attn.indexer.compressor.wgate.weight", DType::kBFloat16, 256, 4096}};
  std::span<const Rule> rules = endpoint;
  unsigned layer_number = 43;
  if (name.starts_with("layers.")) {
    const auto dot = name.find('.', 7);
    if (dot == name.npos) return Status::InvalidArgument("dense tensor layer name malformed");
    const auto number = name.substr(7, dot - 7);
    const auto parsed = std::from_chars(number.data(), number.data() + number.size(), layer_number);
    if (number.empty() || (number.size() > 1 && number.front() == '0') ||
        parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() || layer_number >= 43)
      return Status::InvalidArgument("dense tensor layer number invalid");
    name.remove_prefix(dot + 1);
    if (name.starts_with("ffn.experts.")) return Status::Ok(); // checked by routed helper
    rules = layer;
  }
  const auto validate = [&](const Rule& rule) {
    const std::size_t rank = rule.columns == 0 ? 1 : 2;
    if (dtype != rule.dtype || shape.size() != rank || shape[0] != rule.rows ||
        (rank == 2 && shape[1] != rule.columns))
      return Status::InvalidArgument("dense tensor dtype/shape differs from fixed checkpoint geometry");
    return Status::Ok();
  };
  for (const auto& rule : rules) {
    if (name != rule.name) continue;
    return validate(rule);
  }
  if (layer_number >= 2 && layer_number < 43) {
    const bool overlap = layer_number % 2 == 0;
    const std::uint64_t width = overlap ? 1024 : 512;
    const Rule compressor[]{
        {"attn.compressor.ape", DType::kFloat32, overlap ? 4ULL : 128ULL, width},
        {"attn.compressor.norm.weight", DType::kBFloat16, 512, 0},
        {"attn.compressor.wkv.weight", DType::kBFloat16, width, 4096},
        {"attn.compressor.wgate.weight", DType::kBFloat16, width, 4096}};
    for (const auto& rule : compressor) if (name == rule.name) return validate(rule);
  }
  return Status::InvalidArgument("no dense checkpoint geometry rule for admitted tensor");
}
Status ValidateRoutedExpertGeometry(std::string_view name, DType dtype,
                                   std::span<const std::uint64_t> shape) {
  if (!name.starts_with("layers.") || name.find(".ffn.experts.") == name.npos)
    return Status::Ok();
  const auto layer_end = name.find('.', 7);
  const auto parse_index = [](std::string_view value, unsigned maximum) {
    if (value.empty() || (value.size() > 1 && value.front() == '0')) return false;
    unsigned result = maximum;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && result < maximum;
  };
  if (layer_end == name.npos || !parse_index(name.substr(7, layer_end - 7), 43))
    return Status::InvalidArgument("routed expert layer name invalid");
  auto tail = name.substr(layer_end);
  constexpr std::string_view prefix = ".ffn.experts.";
  if (!tail.starts_with(prefix)) return Status::InvalidArgument("routed expert namespace invalid");
  tail.remove_prefix(prefix.size());
  const auto expert_end = tail.find('.');
  if (expert_end == tail.npos || !parse_index(tail.substr(0, expert_end), 256))
    return Status::InvalidArgument("routed expert index invalid");
  tail.remove_prefix(expert_end + 1);
  const auto matrix_end = tail.find('.');
  if (matrix_end == tail.npos) return Status::InvalidArgument("routed expert matrix missing");
  const auto matrix = tail.substr(0, matrix_end);
  const auto component = tail.substr(matrix_end + 1);
  if ((matrix != "w1" && matrix != "w2" && matrix != "w3") ||
      (component != "weight" && component != "scale"))
    return Status::InvalidArgument("routed expert matrix/component invalid");
  // Mirrors DeepSeekExpertBundleManifest's checkpoint storage ABI, not the
  // unpacked numerical matrix geometry. Packed FP4 remains signed byte storage.
  const bool scale = component == "scale";
  const bool down = matrix == "w2";
  const auto expected_dtype = scale ? DType::kFloat8E8M0 : DType::kInt8;
  const std::uint64_t rows = down ? 4096 : 2048;
  const std::uint64_t columns = scale ? (down ? 64 : 128) : (down ? 1024 : 2048);
  if (dtype != expected_dtype || shape.size() != 2 || shape[0] != rows || shape[1] != columns)
    return Status::InvalidArgument("routed expert dtype/shape differs from frozen checkpoint ABI");
  return Status::Ok();
}
namespace {
std::vector<std::string> Common() {
  std::vector<std::string> names{
      "attn_norm.weight", "attn.attn_sink", "attn.kv_norm.weight", "attn.q_norm.weight",
      "ffn_norm.weight", "ffn.gate.weight", "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
      "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale"};
  for (auto projection : {"wkv", "wo_a", "wo_b", "wq_a", "wq_b"})
    for (auto component : {"weight", "scale"})
      names.push_back("attn." + std::string(projection) + "." + component);
  for (auto projection : {"w1", "w2", "w3"})
    for (auto component : {"weight", "scale"})
      names.push_back("ffn.shared_experts." + std::string(projection) + "." + component);
  for (unsigned expert = 0; expert < 256; ++expert)
    for (auto projection : {"w1", "w2", "w3"})
      for (auto component : {"weight", "scale"})
        names.push_back("ffn.experts." + std::to_string(expert) + "." + projection + "." + component);
  return names;
}
void Append(std::vector<std::string>& destination, std::string_view prefix,
            const std::vector<std::string>& suffixes) {
  for (const auto& suffix : suffixes) destination.push_back(std::string(prefix) + suffix);
}
}  // namespace
Status ValidateSourceTensorGeometry(std::string_view name, DType dtype,
                                   std::span<const std::uint64_t> shape) {
  if (!name.starts_with("mtp.")) {
    auto status = ValidateRoutedExpertGeometry(name, dtype, shape);
    if (!status.ok()) return status;
    return ValidatePp1DenseGeometry(name, dtype, shape);
  }
  if (name.size() <= 6 || name[4] < '0' || name[4] > '2' || name[5] != '.')
    return Status::InvalidArgument("source MTP stage name invalid");
  const auto stage = static_cast<unsigned>(name[4] - '0');
  const auto suffix = name.substr(6);
  struct Rule { unsigned stage; std::string_view name; DType dtype; std::uint64_t rows, columns; };
  static constexpr Rule special[]{
      {0, "main_proj.weight", DType::kFloat8E4M3, 4096, 12288},
      {0, "main_proj.scale", DType::kFloat8E8M0, 32, 96},
      {0, "main_norm.weight", DType::kBFloat16, 4096, 0},
      {2, "norm.weight", DType::kBFloat16, 4096, 0},
      {2, "markov_head.markov_w1.weight", DType::kBFloat16, 129280, 256},
      {2, "markov_head.markov_w2.weight", DType::kBFloat16, 129280, 256},
      {2, "confidence_head.proj.weight", DType::kBFloat16, 1, 4352},
      {2, "hc_head_fn", DType::kFloat32, 4, 16384},
      {2, "hc_head_scale", DType::kFloat32, 1, 0},
      {2, "hc_head_base", DType::kFloat32, 4, 0}};
  for (const auto& rule : special) {
    if (suffix != rule.name) continue;
    const auto rank = rule.columns == 0 ? 1U : 2U;
    if (stage != rule.stage || dtype != rule.dtype || shape.size() != rank ||
        shape[0] != rule.rows || (rank == 2 && shape[1] != rule.columns))
      return Status::InvalidArgument("MTP stage-specific tensor geometry invalid");
    return Status::Ok();
  }
  // MTP common blocks have no compressor/indexer or token-id hash routing.
  if (suffix.starts_with("attn.compressor.") || suffix.starts_with("attn.indexer.") ||
      suffix == "ffn.gate.tid2eid")
    return Status::InvalidArgument("main-layer-only tensor is not part of MTP geometry");
  // The common attention/mHC/shared/routed matrices use the same checkpoint
  // storage geometry. Do not reinterpret shared FP8 as routed packed FP4.
  const auto common_name = "layers.0." + std::string(suffix);
  auto status = ValidateRoutedExpertGeometry(common_name, dtype, shape);
  if (!status.ok()) return status;
  return ValidatePp1DenseGeometry(common_name, dtype, shape);
}
Result<std::vector<std::string>> ExpectedTensorNames(bool include_mtp) {
  std::vector<std::string> result{
      "embed.weight", "head.weight", "norm.weight", "hc_head_fn", "hc_head_scale", "hc_head_base"};
  result.reserve(include_mtp ? 72'317 : 67'612);
  const auto common = Common();
  if (common.size() != 28 + 256 * 6)
    return Status::Internal("native 0731 common tensor grammar count drifted");
  for (unsigned layer = 0; layer < 43; ++layer) {
    auto names = common;
    names.push_back(layer <= 2 ? "ffn.gate.tid2eid" : "ffn.gate.bias");
    if (layer >= 2) {
      for (auto suffix : {"attn.compressor.ape", "attn.compressor.norm.weight",
                         "attn.compressor.wgate.weight", "attn.compressor.wkv.weight"})
        names.emplace_back(suffix);
    }
    if (layer >= 2 && layer % 2 == 0) {
      for (auto suffix : {"attn.indexer.wq_b.weight", "attn.indexer.wq_b.scale",
                         "attn.indexer.weights_proj.weight", "attn.indexer.compressor.ape",
                         "attn.indexer.compressor.norm.weight", "attn.indexer.compressor.wgate.weight",
                         "attn.indexer.compressor.wkv.weight"}) names.emplace_back(suffix);
    }
    Append(result, "layers." + std::to_string(layer) + ".", names);
  }
  if (include_mtp) {
    for (unsigned stage = 0; stage < 3; ++stage) {
      auto names = common;
      names.emplace_back("ffn.gate.bias");
      if (stage == 0)
        for (auto suffix : {"main_norm.weight", "main_proj.weight", "main_proj.scale"}) names.emplace_back(suffix);
      if (stage == 2)
        for (auto suffix : {"confidence_head.proj.weight", "hc_head_base", "hc_head_fn", "hc_head_scale",
                           "markov_head.markov_w1.weight", "markov_head.markov_w2.weight", "norm.weight"})
          names.emplace_back(suffix);
      Append(result, "mtp." + std::to_string(stage) + ".", names);
    }
  }
  std::ranges::sort(result);
  if (result.size() != (include_mtp ? 72'317 : 67'612) ||
      std::adjacent_find(result.begin(), result.end()) != result.end())
    return Status::Internal("native 0731 tensor grammar count or uniqueness drifted");
  return result;
}
}  // namespace pih::offline_deepseek
