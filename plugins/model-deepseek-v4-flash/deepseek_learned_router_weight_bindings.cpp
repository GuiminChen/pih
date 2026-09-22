#include "pih/model/deepseek_learned_router_weight_bindings.h"

#include <algorithm>
#include <string>

namespace pih {

Result<DeepSeekLearnedRouterWeightBindings>
DeepSeekLearnedRouterWeightBindings::Resolve(
    DeepSeekStageRange owned_layers, const Resolver& resolver) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || !resolver) {
    return Status::InvalidArgument(
        "DeepSeek learned router weight binding identity is invalid");
  }
  DeepSeekLearnedRouterWeightBindings result;
  const auto first = std::max(owned_layers.first_layer, 3U);
  if (first > owned_layers.last_layer) return result;
  result.bindings_.reserve(owned_layers.last_layer - first + 1);
  for (auto layer = first; layer <= owned_layers.last_layer; ++layer) {
    const auto name = "layers." + std::to_string(layer) +
                      ".ffn.gate.weight";
    auto view = resolver(name);
    if (!view.ok()) return view.status();
    if (view->data() == nullptr || view->dtype() != DType::kBFloat16 ||
        view->rank() != 2 || view->dim(0) != 256 ||
        view->dim(1) != 4096 || view->stride(1) != 1 ||
        view->stride(0) != 4096 ||
        view->device().type() != DeviceType::kCuda ||
        view->generation() == 0 ||
        (result.generation_ != 0 &&
         result.generation_ != view->generation())) {
      return Status::FailedPrecondition(
          "DeepSeek learned router weight violates checkpoint ABI");
    }
    result.generation_ = view->generation();
    result.bindings_.push_back(
        {layer, reinterpret_cast<std::uintptr_t>(view->data())});
  }
  return result;
}

Result<DeepSeekLearnedRouterWeightBindings>
DeepSeekLearnedRouterWeightBindings::Resolve(
    DeepSeekStageRange owned_layers,
    const DeepSeekResidentWeightArena& arena) {
  auto resolved = Resolve(owned_layers, [&arena](std::string_view name) {
    return arena.tensor(name);
  });
  if (!resolved.ok()) return resolved.status();
  if (resolved->size() != 0 && resolved->generation() != arena.generation()) {
    return Status::FailedPrecondition(
        "DeepSeek learned router weight generation is foreign");
  }
  return resolved;
}

}  // namespace pih
