#include "pih/model/deepseek_mhc_weight_bindings.h"

#include <array>
#include <string>

namespace pih { namespace {

Result<std::uintptr_t> exact(
    std::string_view name, DType dtype, std::span<const std::uint64_t> shape,
    const DeepSeekMhcWeightBindings::Resolver& resolver,
    std::uint64_t& generation) {
  auto view = resolver(name);
  if (!view.ok()) return view.status();
  if (view->data() == nullptr || view->dtype() != dtype ||
      view->rank() != shape.size() ||
      view->device().type() != DeviceType::kCuda || view->generation() == 0 ||
      (generation != 0 && generation != view->generation())) {
    return Status::FailedPrecondition(
        "DeepSeek mHC weight violates checkpoint ABI");
  }
  std::uint64_t stride = 1;
  for (std::size_t reverse = shape.size(); reverse > 0; --reverse) {
    const auto axis = reverse - 1;
    if (view->dim(axis) != shape[axis] || view->stride(axis) != stride) {
      return Status::FailedPrecondition(
          "DeepSeek mHC weight shape or stride violates checkpoint ABI");
    }
    stride *= shape[axis];
  }
  generation = view->generation();
  return reinterpret_cast<std::uintptr_t>(view->data());
}

Result<DeepSeekMhcWeightBindings> resolve_with_prefix(
    std::string prefix, const DeepSeekMhcWeightBindings::Resolver& resolver) {
  if (!resolver) {
    return Status::InvalidArgument("DeepSeek mHC weight resolver is invalid");
  }
  DeepSeekMhcWeightBindings result;
  const auto bind = [&](std::uintptr_t& target, std::string_view suffix,
                        DType dtype,
                        std::span<const std::uint64_t> shape) -> Status {
    auto value = exact(prefix + std::string(suffix), dtype, shape, resolver,
                       result.generation);
    if (!value.ok()) return value.status();
    target = *value;
    return Status::Ok();
  };
  const std::array<std::uint64_t, 1> norm{4096};
  const std::array<std::uint64_t, 2> fn{24, 16384};
  const std::array<std::uint64_t, 1> scale{3};
  const std::array<std::uint64_t, 1> base{24};
#define PIH_BIND(field, suffix, dtype, shape) \
  do { auto s = bind(result.field, suffix, dtype, shape); \
       if (!s.ok()) return s; } while (false)
  PIH_BIND(attention_norm_bf16, "attn_norm.weight", DType::kBFloat16, norm);
  PIH_BIND(attention_fn_f32, "hc_attn_fn", DType::kFloat32, fn);
  PIH_BIND(attention_scale_f32, "hc_attn_scale", DType::kFloat32, scale);
  PIH_BIND(attention_base_f32, "hc_attn_base", DType::kFloat32, base);
  PIH_BIND(feed_forward_norm_bf16, "ffn_norm.weight", DType::kBFloat16, norm);
  PIH_BIND(feed_forward_fn_f32, "hc_ffn_fn", DType::kFloat32, fn);
  PIH_BIND(feed_forward_scale_f32, "hc_ffn_scale", DType::kFloat32, scale);
  PIH_BIND(feed_forward_base_f32, "hc_ffn_base", DType::kFloat32, base);
#undef PIH_BIND
  return result;
}

}  // namespace

Result<DeepSeekMhcWeightBindings> DeepSeekMhcWeightBindings::Resolve(
    std::uint32_t layer, const Resolver& resolver) {
  if (layer >= 43) {
    return Status::InvalidArgument("DeepSeek mHC weight layer is invalid");
  }
  return resolve_with_prefix("layers." + std::to_string(layer) + ".",
                             resolver);
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Result<DeepSeekMhcWeightBindings> DeepSeekMhcWeightBindings::ResolveDspark(
    DeepSeekDsparkStageId stage, const Resolver& resolver) {
  auto namespace_name = deepseek_dspark_namespace(stage);
  if (!namespace_name.ok()) return namespace_name.status();
  return resolve_with_prefix(*namespace_name + ".", resolver);
}
#endif

Result<DeepSeekMhcWeightBindings> DeepSeekMhcWeightBindings::Resolve(
    std::uint32_t layer, const DeepSeekResidentWeightArena& arena) {
  auto result = Resolve(layer, [&arena](std::string_view name) {
    return arena.tensor(name);
  });
  if (!result.ok()) return result.status();
  if (result->generation != arena.generation()) {
    return Status::FailedPrecondition("DeepSeek mHC generation is foreign");
  }
  return result;
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Result<DeepSeekMhcWeightBindings> DeepSeekMhcWeightBindings::ResolveDspark(
    DeepSeekDsparkStageId stage,
    const DeepSeekResidentWeightArena& arena) {
  auto result = ResolveDspark(stage, [&arena](std::string_view name) {
    return arena.tensor(name);
  });
  if (!result.ok()) return result.status();
  if (result->generation != arena.generation()) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark mHC weight generation is foreign");
  }
  return result;
}
#endif

}  // namespace pih
