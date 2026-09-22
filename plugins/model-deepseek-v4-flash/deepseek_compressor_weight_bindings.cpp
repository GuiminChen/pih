#include "pih/model/deepseek_compressor_weight_bindings.h"

#include <initializer_list>
#include <string>

namespace pih { namespace {

Result<std::uintptr_t> exact(
    std::string_view name, DType dtype,
    std::initializer_list<std::uint64_t> shape,
    const DeepSeekCompressorWeightBindings::Resolver& resolver,
    std::uint64_t& generation) {
  auto view = resolver(name);
  if (!view.ok()) return view.status();
  if (view->data() == nullptr || view->dtype() != dtype ||
      view->rank() != shape.size() ||
      view->device().type() != DeviceType::kCuda || view->generation() == 0 ||
      (generation != 0 && generation != view->generation())) {
    return Status::FailedPrecondition(
        "DeepSeek compressor weight violates checkpoint ABI");
  }
  std::uint64_t stride = 1;
  auto reverse = shape.end();
  for (std::size_t axis = shape.size(); axis > 0; --axis) {
    --reverse;
    if (view->dim(axis - 1) != *reverse ||
        view->stride(axis - 1) != stride) {
      return Status::FailedPrecondition(
          "DeepSeek compressor weight shape or stride violates checkpoint ABI");
    }
    stride *= *reverse;
  }
  generation = view->generation();
  return reinterpret_cast<std::uintptr_t>(view->data());
}

}  // namespace

Result<DeepSeekCompressorWeightBindings>
DeepSeekCompressorWeightBindings::Resolve(
    std::uint32_t layer, DeepSeekCompressedAttentionKind kind,
    const Resolver& resolver) {
  if (layer > 42 || !resolver ||
      (kind != DeepSeekCompressedAttentionKind::kRatio4 &&
       kind != DeepSeekCompressedAttentionKind::kRatio128)) {
    return Status::InvalidArgument(
        "DeepSeek compressor weight request is invalid");
  }
  const auto prefix = "layers." + std::to_string(layer) + ".attn.";
  DeepSeekCompressorWeightBindings result;
  const auto bind = [&](std::uintptr_t& target, std::string_view suffix,
                        DType dtype,
                        std::initializer_list<std::uint64_t> shape) -> Status {
    auto value = exact(prefix + std::string(suffix), dtype, shape, resolver,
                       result.generation);
    if (!value.ok()) return value.status();
    target = *value;
    return Status::Ok();
  };
#define PIH_BIND(field, suffix, dtype, ...) \
  do { auto s = bind(result.field, suffix, dtype, {__VA_ARGS__}); \
       if (!s.ok()) return s; } while (false)
  if (kind == DeepSeekCompressedAttentionKind::kRatio4) {
    PIH_BIND(main_wkv_bf16,
                   "compressor.wkv.weight", DType::kBFloat16, 1024, 4096);
    PIH_BIND(main_wgate_bf16,
                   "compressor.wgate.weight", DType::kBFloat16, 1024, 4096);
    PIH_BIND(main_ape_f32, "compressor.ape", DType::kFloat32, 4, 1024);
    PIH_BIND(main_norm_bf16, "compressor.norm.weight",
                   DType::kBFloat16, 512);
    PIH_BIND(indexer_wq_b_e4m3, "indexer.wq_b.weight",
                   DType::kFloat8E4M3, 8192, 1024);
    PIH_BIND(indexer_wq_b_scale_bits, "indexer.wq_b.scale",
                   DType::kFloat8E8M0, 64, 8);
    PIH_BIND(indexer_weights_proj_bf16, "indexer.weights_proj.weight",
                   DType::kBFloat16, 64, 4096);
    PIH_BIND(indexer_wkv_bf16,
                   "indexer.compressor.wkv.weight", DType::kBFloat16, 256, 4096);
    PIH_BIND(indexer_wgate_bf16,
                   "indexer.compressor.wgate.weight", DType::kBFloat16, 256, 4096);
    PIH_BIND(indexer_ape_f32, "indexer.compressor.ape",
                   DType::kFloat32, 4, 256);
    PIH_BIND(indexer_norm_bf16, "indexer.compressor.norm.weight",
                   DType::kBFloat16, 128);
  } else {
    PIH_BIND(main_wkv_bf16,
                   "compressor.wkv.weight", DType::kBFloat16, 512, 4096);
    PIH_BIND(main_wgate_bf16,
                   "compressor.wgate.weight", DType::kBFloat16, 512, 4096);
    PIH_BIND(main_ape_f32, "compressor.ape", DType::kFloat32, 128, 512);
    PIH_BIND(main_norm_bf16, "compressor.norm.weight",
                   DType::kBFloat16, 512);
  }
#undef PIH_BIND
  return result;
}

Result<DeepSeekCompressorWeightBindings>
DeepSeekCompressorWeightBindings::Resolve(
    std::uint32_t layer, DeepSeekCompressedAttentionKind kind,
    const DeepSeekResidentWeightArena& arena) {
  auto result = Resolve(layer, kind, [&arena](std::string_view name) {
    return arena.tensor(name);
  });
  if (!result.ok()) return result.status();
  if (result->generation != arena.generation()) {
    return Status::FailedPrecondition(
        "DeepSeek compressor weight generation is foreign");
  }
  return result;
}

}  // namespace pih
