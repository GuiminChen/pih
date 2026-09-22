#include "pih/model/deepseek_endpoint_weight_bindings.h"

#include <initializer_list>

namespace pih {
namespace {

Result<std::uintptr_t> resolve_exact(
    std::string_view name, DType dtype,
    std::initializer_list<std::uint64_t> dimensions,
    const DeepSeekEndpointWeightBindings::Resolver& resolver,
    std::uint64_t& generation) {
  auto view = resolver(name);
  if (!view.ok()) return view.status();
  if (view->data() == nullptr || view->dtype() != dtype ||
      view->rank() != dimensions.size() ||
      view->device().type() != DeviceType::kCuda ||
      view->generation() == 0 ||
      (generation != 0 && generation != view->generation())) {
    return Status::FailedPrecondition(
        "DeepSeek endpoint weight violates checkpoint ABI");
  }
  std::size_t axis = 0;
  for (const auto dimension : dimensions) {
    if (view->dim(axis) != dimension) {
      return Status::FailedPrecondition(
          "DeepSeek endpoint weight shape violates checkpoint ABI");
    }
    ++axis;
  }
  std::uint64_t expected_stride = 1;
  for (std::size_t reverse = dimensions.size(); reverse > 0; --reverse) {
    const auto current = reverse - 1;
    if (view->stride(current) != expected_stride) {
      return Status::FailedPrecondition(
          "DeepSeek endpoint weight must be contiguous");
    }
    expected_stride *= view->dim(current);
  }
  generation = view->generation();
  return reinterpret_cast<std::uintptr_t>(view->data());
}

}  // namespace

Result<DeepSeekEndpointWeightBindings>
DeepSeekEndpointWeightBindings::Resolve(
    DeepSeekStagePlan stage, const Resolver& resolver) {
  if (stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43 || !resolver) {
    return Status::InvalidArgument(
        "DeepSeek endpoint weight stage identity is invalid");
  }
  DeepSeekEndpointWeightBindings result;
  if (stage.owns_embedding) {
    auto weight = resolve_exact("embed.weight", DType::kBFloat16,
                                {129280, 4096}, resolver,
                                result.generation);
    if (!weight.ok()) return weight.status();
    result.embedding_weight_bf16 = *weight;
  }
  if (stage.owns_lm_head) {
    auto fn = resolve_exact("hc_head_fn", DType::kFloat32, {4, 16384},
                            resolver, result.generation);
    if (!fn.ok()) return fn.status();
    result.hc_head_fn_f32 = *fn;
    auto scale = resolve_exact("hc_head_scale", DType::kFloat32, {1},
                               resolver, result.generation);
    if (!scale.ok()) return scale.status();
    result.hc_head_scale_f32 = *scale;
    auto base = resolve_exact("hc_head_base", DType::kFloat32, {4},
                              resolver, result.generation);
    if (!base.ok()) return base.status();
    result.hc_head_base_f32 = *base;
    auto norm = resolve_exact("norm.weight", DType::kBFloat16, {4096},
                              resolver, result.generation);
    if (!norm.ok()) return norm.status();
    result.norm_weight_bf16 = *norm;
    auto head = resolve_exact("head.weight", DType::kBFloat16,
                              {129280, 4096}, resolver,
                              result.generation);
    if (!head.ok()) return head.status();
    result.head_weight_bf16 = *head;
  }
  return result;
}

Result<DeepSeekEndpointWeightBindings>
DeepSeekEndpointWeightBindings::Resolve(
    DeepSeekStagePlan stage, const DeepSeekResidentWeightArena& arena) {
  auto result = Resolve(stage, [&arena](std::string_view name) {
    return arena.tensor(name);
  });
  if (!result.ok()) return result.status();
  if (result->generation != 0 && result->generation != arena.generation()) {
    return Status::FailedPrecondition(
        "DeepSeek endpoint weight generation is foreign");
  }
  return result;
}

}  // namespace pih
