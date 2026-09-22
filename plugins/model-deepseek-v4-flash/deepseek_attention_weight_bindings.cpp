#include "pih/model/deepseek_attention_weight_bindings.h"

#include <initializer_list>
#include <string>

namespace pih { namespace {

Result<std::uintptr_t> exact(
    std::string_view name, DType dtype,
    std::initializer_list<std::uint64_t> shape,
    const DeepSeekAttentionWeightBindings::Resolver& resolver,
    std::uint64_t& generation) {
  auto view = resolver(name);
  if (!view.ok()) return view.status();
  if (view->data() == nullptr || view->dtype() != dtype ||
      view->rank() != shape.size() ||
      view->device().type() != DeviceType::kCuda || view->generation() == 0 ||
      (generation != 0 && generation != view->generation())) {
    return Status::FailedPrecondition(
        "DeepSeek attention weight violates checkpoint ABI");
  }
  std::uint64_t stride = 1;
  auto reverse = shape.end();
  for (std::size_t axis = shape.size(); axis > 0; --axis) {
    --reverse;
    if (view->dim(axis - 1) != *reverse ||
        view->stride(axis - 1) != stride) {
      return Status::FailedPrecondition(
          "DeepSeek attention weight shape or stride violates checkpoint ABI");
    }
    stride *= *reverse;
  }
  generation = view->generation();
  return reinterpret_cast<std::uintptr_t>(view->data());
}

Result<DeepSeekAttentionWeightBindings>
resolve_with_prefix(std::string prefix,
                    const DeepSeekAttentionWeightBindings::Resolver& resolver) {
  if (!resolver) {
    return Status::InvalidArgument(
        "DeepSeek attention weight resolver is invalid");
  }
  DeepSeekAttentionWeightBindings result;
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
  PIH_BIND(wq_a_fp8, "wq_a.weight", DType::kFloat8E4M3, 1024, 4096);
  PIH_BIND(wq_a_scale_ue8m0, "wq_a.scale", DType::kFloat8E8M0, 8, 32);
  PIH_BIND(q_norm_bf16, "q_norm.weight", DType::kBFloat16, 1024);
  PIH_BIND(wq_b_fp8, "wq_b.weight", DType::kFloat8E4M3, 32768, 1024);
  PIH_BIND(wq_b_scale_ue8m0, "wq_b.scale", DType::kFloat8E8M0, 256, 8);
  PIH_BIND(wkv_fp8, "wkv.weight", DType::kFloat8E4M3, 512, 4096);
  PIH_BIND(wkv_scale_ue8m0, "wkv.scale", DType::kFloat8E8M0, 4, 32);
  PIH_BIND(kv_norm_bf16, "kv_norm.weight", DType::kBFloat16, 512);
  PIH_BIND(attention_sink_f32, "attn_sink", DType::kFloat32, 64);
  PIH_BIND(wo_a_fp8, "wo_a.weight", DType::kFloat8E4M3, 8192, 4096);
  PIH_BIND(wo_a_scale_ue8m0, "wo_a.scale", DType::kFloat8E8M0, 64, 32);
  PIH_BIND(wo_b_fp8, "wo_b.weight", DType::kFloat8E4M3, 4096, 8192);
  PIH_BIND(wo_b_scale_ue8m0, "wo_b.scale", DType::kFloat8E8M0, 32, 64);
#undef PIH_BIND
  return result;
}

}  // namespace

Result<DeepSeekAttentionWeightBindings>
DeepSeekAttentionWeightBindings::Resolve(
    std::uint32_t layer, const Resolver& resolver) {
  if (layer >= 43) {
    return Status::InvalidArgument(
        "DeepSeek attention weight layer is invalid");
  }
  return resolve_with_prefix(
      "layers." + std::to_string(layer) + ".attn.", resolver);
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Result<DeepSeekAttentionWeightBindings>
DeepSeekAttentionWeightBindings::ResolveDspark(
    DeepSeekDsparkStageId stage, const Resolver& resolver) {
  auto namespace_name = deepseek_dspark_namespace(stage);
  if (!namespace_name.ok()) return namespace_name.status();
  return resolve_with_prefix(*namespace_name + ".attn.", resolver);
}
#endif

Result<DeepSeekAttentionWeightBindings>
DeepSeekAttentionWeightBindings::Resolve(
    std::uint32_t layer, const DeepSeekResidentWeightArena& arena) {
  auto result = Resolve(layer, [&arena](std::string_view name) {
    return arena.tensor(name);
  });
  if (!result.ok()) return result.status();
  if (result->generation != arena.generation()) {
    return Status::FailedPrecondition(
        "DeepSeek attention weight generation is foreign");
  }
  return result;
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Result<DeepSeekAttentionWeightBindings>
DeepSeekAttentionWeightBindings::ResolveDspark(
    DeepSeekDsparkStageId stage,
    const DeepSeekResidentWeightArena& arena) {
  auto result = ResolveDspark(stage, [&arena](std::string_view name) {
    return arena.tensor(name);
  });
  if (!result.ok()) return result.status();
  if (result->generation != arena.generation()) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark attention weight generation is foreign");
  }
  return result;
}
#endif

}  // namespace pih
