#include "pih/model/deepseek_resident_expert_bindings.h"

#include <string>

namespace pih { namespace {

Result<DeepSeekExpertDeviceSpan> resolve_segment(
    std::string_view name, DType dtype, std::uint64_t rows,
    std::uint64_t columns,
    const DeepSeekResidentExpertBindings::Resolver& resolver,
    std::uint64_t& generation) {
  auto view = resolver(name);
  if (!view.ok()) return view.status();
  if (view->data() == nullptr || view->dtype() != dtype || view->rank() != 2 ||
      view->dim(0) != rows || view->dim(1) != columns ||
      view->stride(1) != 1 || view->stride(0) != columns ||
      view->device().type() != DeviceType::kCuda || view->generation() == 0 ||
      (generation != 0 && generation != view->generation())) {
    return Status::FailedPrecondition(
        "DeepSeek resident expert violates checkpoint ABI");
  }
  generation = view->generation();
  const auto bytes = dtype == DType::kInt8
                         ? DeepSeekExpertBundleLayout::kPackedBytesPerMatrix
                         : DeepSeekExpertBundleLayout::kScaleBytesPerMatrix;
  return DeepSeekExpertDeviceSpan{
      reinterpret_cast<std::uintptr_t>(view->data()), bytes};
}

Result<DeepSeekExpertMatrixDeviceView> resolve_matrix(
    std::uint32_t layer, std::uint32_t expert, std::string_view matrix,
    const DeepSeekResidentExpertBindings::Resolver& resolver,
    std::uint64_t& generation) {
  const bool w2 = matrix == "w2";
  const auto prefix = "layers." + std::to_string(layer) +
                      ".ffn.experts." + std::to_string(expert) + "." +
                      std::string(matrix);
  auto packed = resolve_segment(prefix + ".weight", DType::kInt8,
                                w2 ? 4096U : 2048U,
                                w2 ? 1024U : 2048U, resolver, generation);
  if (!packed.ok()) return packed.status();
  auto scales = resolve_segment(prefix + ".scale", DType::kFloat8E8M0,
                                w2 ? 4096U : 2048U,
                                w2 ? 64U : 128U, resolver, generation);
  if (!scales.ok()) return scales.status();
  return DeepSeekExpertMatrixDeviceView{*packed, *scales};
}

}  // namespace

Result<DeepSeekResidentExpertBindings>
DeepSeekResidentExpertBindings::Resolve(DeepSeekStagePlan stage,
                                        const Resolver& resolver) {
  if (stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43 || !resolver) {
    return Status::InvalidArgument(
        "DeepSeek resident expert stage identity is invalid");
  }
  DeepSeekResidentExpertBindings result;
  const auto first = stage.layers.first_layer;
  for (auto layer = first; layer <= stage.layers.last_layer; ++layer) {
    auto& bindings = result.layers_[layer];
    for (std::uint32_t expert = 0; expert < kExpertCount; ++expert) {
      auto w1 = resolve_matrix(layer, expert, "w1", resolver,
                               result.generation_);
      if (!w1.ok()) return w1.status();
      auto w2 = resolve_matrix(layer, expert, "w2", resolver,
                               result.generation_);
      if (!w2.ok()) return w2.status();
      auto w3 = resolve_matrix(layer, expert, "w3", resolver,
                               result.generation_);
      if (!w3.ok()) return w3.status();
      bindings[expert] = {*w1, *w2, *w3};
    }
  }
  return result;
}

Result<DeepSeekResidentExpertBindings>
DeepSeekResidentExpertBindings::Resolve(
    DeepSeekStagePlan stage, const DeepSeekResidentWeightArena& arena) {
  auto result = Resolve(stage, [&arena](std::string_view name) {
    return arena.tensor(name);
  });
  if (!result.ok()) return result.status();
  if (result->generation() != arena.generation()) {
    return Status::FailedPrecondition(
        "DeepSeek resident expert generation is foreign");
  }
  return result;
}

bool DeepSeekResidentExpertBindings::contains(
    std::uint32_t layer) const noexcept {
  return layers_.contains(layer);
}

const DeepSeekExpertBundleDeviceView& DeepSeekResidentExpertBindings::expert(
    std::uint32_t layer, std::uint32_t id) const {
  return layers_.at(layer).at(id);
}

}  // namespace pih
