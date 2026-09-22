#include "pih/model/deepseek_dspark_weight_bindings.h"

#include <initializer_list>
#include <string>

namespace pih { namespace {

Result<std::uintptr_t> resolve_exact(
    std::string_view name, DType dtype,
    std::initializer_list<std::uint64_t> dimensions,
    const DeepSeekDsparkWeightBindings::Resolver& resolver,
    std::uint64_t& generation) {
  auto view = resolver(name);
  if (!view.ok()) return view.status();
  if (view->data() == nullptr || view->dtype() != dtype ||
      view->rank() != dimensions.size() ||
      view->device().type() != DeviceType::kCuda || view->generation() == 0 ||
      (generation != 0 && generation != view->generation())) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark weight violates checkpoint ABI");
  }
  std::uint64_t expected_stride = 1;
  auto reverse = dimensions.end();
  for (std::size_t axis = dimensions.size(); axis > 0; --axis) {
    --reverse;
    if (view->dim(axis - 1) != *reverse ||
        view->stride(axis - 1) != expected_stride) {
      return Status::FailedPrecondition(
          "DeepSeek DSpark weight shape violates checkpoint ABI");
    }
    expected_stride *= *reverse;
  }
  generation = view->generation();
  return reinterpret_cast<std::uintptr_t>(view->data());
}

Status merge_generation(std::uint64_t candidate, std::uint64_t& generation) {
  if (candidate == 0 || (generation != 0 && candidate != generation)) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark common weight generations differ");
  }
  generation = candidate;
  return Status::Ok();
}

Result<DeepSeekExpertDeviceSpan> resolve_shared_segment(
    std::string_view name, DType dtype, std::uint64_t rows,
    std::uint64_t columns,
    const DeepSeekDsparkWeightBindings::Resolver& resolver,
    std::uint64_t& generation) {
  auto address = resolve_exact(name, dtype, {rows, columns}, resolver,
                               generation);
  if (!address.ok()) return address.status();
  const auto bytes = dtype == DType::kInt8
                         ? DeepSeekExpertBundleLayout::kPackedBytesPerMatrix
                         : DeepSeekExpertBundleLayout::kScaleBytesPerMatrix;
  return DeepSeekExpertDeviceSpan{*address, bytes};
}

Result<DeepSeekExpertMatrixDeviceView> resolve_shared_matrix(
    std::string_view prefix, std::string_view matrix,
    const DeepSeekDsparkWeightBindings::Resolver& resolver,
    std::uint64_t& generation) {
  const bool w2 = matrix == "w2";
  const std::string base = std::string(prefix) + ".ffn.shared_experts." +
                           std::string(matrix);
  auto packed = resolve_shared_segment(
      base + ".weight", DType::kInt8, w2 ? 4096U : 2048U,
      w2 ? 1024U : 2048U, resolver, generation);
  if (!packed.ok()) return packed.status();
  auto scales = resolve_shared_segment(
      base + ".scale", DType::kFloat8E8M0, w2 ? 4096U : 2048U,
      w2 ? 64U : 128U, resolver, generation);
  if (!scales.ok()) return scales.status();
  return DeepSeekExpertMatrixDeviceView{*packed, *scales};
}

Result<DeepSeekDsparkCommonWeightBindings> resolve_common(
    DeepSeekDsparkStageId stage,
    const DeepSeekDsparkWeightBindings::Resolver& resolver,
    std::uint64_t& generation) {
  auto namespace_name = deepseek_dspark_namespace(stage);
  if (!namespace_name.ok()) return namespace_name.status();
  DeepSeekDsparkCommonWeightBindings result;
  result.stage = stage;
  auto attention = DeepSeekAttentionWeightBindings::ResolveDspark(
      stage, resolver);
  if (!attention.ok()) return attention.status();
  auto status = merge_generation(attention->generation, generation);
  if (!status.ok()) return status;
  result.attention = std::move(*attention);
  auto mhc = DeepSeekMhcWeightBindings::ResolveDspark(stage, resolver);
  if (!mhc.ok()) return mhc.status();
  status = merge_generation(mhc->generation, generation);
  if (!status.ok()) return status;
  result.mhc = std::move(*mhc);
  auto router_weight = resolve_exact(
      *namespace_name + ".ffn.gate.weight", DType::kBFloat16,
      {256, 4096}, resolver, generation);
  if (!router_weight.ok()) return router_weight.status();
  result.router_weight_bf16 = *router_weight;
  auto router_bias = resolve_exact(
      *namespace_name + ".ffn.gate.bias", DType::kFloat32, {256}, resolver,
      generation);
  if (!router_bias.ok()) return router_bias.status();
  result.router_bias_f32 = *router_bias;
  auto w1 = resolve_shared_matrix(*namespace_name, "w1", resolver,
                                  generation);
  if (!w1.ok()) return w1.status();
  auto w2 = resolve_shared_matrix(*namespace_name, "w2", resolver,
                                  generation);
  if (!w2.ok()) return w2.status();
  auto w3 = resolve_shared_matrix(*namespace_name, "w3", resolver,
                                  generation);
  if (!w3.ok()) return w3.status();
  result.shared_expert = {*w1, *w2, *w3};
  result.generation = generation;
  return result;
}

}  // namespace

Result<DeepSeekDsparkWeightBindings> DeepSeekDsparkWeightBindings::Resolve(
    DeepSeekStagePlan stage, const Resolver& resolver) {
  if (!stage.owns_dspark || !stage.owns_lm_head ||
      stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer != 42 || !resolver) {
    return Status::InvalidArgument(
        "DeepSeek DSpark weight stage identity is invalid");
  }
  DeepSeekDsparkWeightBindings result;
  for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    auto stage_id = deepseek_dspark_stage_id(index);
    if (!stage_id.ok()) return stage_id.status();
    auto common = resolve_common(*stage_id, resolver, result.generation_);
    if (!common.ok()) return common.status();
    result.common_[index] = std::move(*common);
  }
#define PIH_DSPARK_BIND(target, name, dtype, ...)                    \
  do {                                                                     \
    auto value = resolve_exact(name, dtype, {__VA_ARGS__}, resolver,       \
                               result.generation_);                        \
    if (!value.ok()) return value.status();                                \
    target = *value;                                                       \
  } while (false)
  PIH_DSPARK_BIND(result.boundary_.main_proj_fp8,
                        "mtp.0.main_proj.weight", DType::kFloat8E4M3,
                        4096, 12288);
  PIH_DSPARK_BIND(result.boundary_.main_proj_scale_ue8m0,
                        "mtp.0.main_proj.scale", DType::kFloat8E8M0,
                        32, 96);
  PIH_DSPARK_BIND(result.boundary_.main_norm_weight_bf16,
                        "mtp.0.main_norm.weight", DType::kBFloat16, 4096);
  PIH_DSPARK_BIND(result.head_.norm_weight_bf16,
                        "mtp.2.norm.weight", DType::kBFloat16, 4096);
  PIH_DSPARK_BIND(result.head_.markov_embedding_bf16,
                        "mtp.2.markov_head.markov_w1.weight",
                        DType::kBFloat16, 129280, 256);
  PIH_DSPARK_BIND(result.head_.markov_head_bf16,
                        "mtp.2.markov_head.markov_w2.weight",
                        DType::kBFloat16, 129280, 256);
  PIH_DSPARK_BIND(result.head_.confidence_weight_bf16,
                        "mtp.2.confidence_head.proj.weight",
                        DType::kBFloat16, 1, 4352);
  PIH_DSPARK_BIND(result.head_.hc_head_fn_f32,
                        "mtp.2.hc_head_fn", DType::kFloat32, 4, 16384);
  PIH_DSPARK_BIND(result.head_.hc_head_scale_f32,
                        "mtp.2.hc_head_scale", DType::kFloat32, 1);
  PIH_DSPARK_BIND(result.head_.hc_head_base_f32,
                        "mtp.2.hc_head_base", DType::kFloat32, 4);
#undef PIH_DSPARK_BIND
  return result;
}

Result<DeepSeekDsparkWeightBindings> DeepSeekDsparkWeightBindings::Resolve(
    DeepSeekStagePlan stage, const DeepSeekResidentWeightArena& arena) {
  auto result = Resolve(stage, [&arena](std::string_view name) {
    return arena.tensor(name);
  });
  if (!result.ok()) return result.status();
  if (result->generation() != arena.generation()) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark weight generation is foreign");
  }
  return result;
}

}  // namespace pih
