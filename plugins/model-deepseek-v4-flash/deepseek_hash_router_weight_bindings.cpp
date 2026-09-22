#include "pih/model/deepseek_hash_router_weight_bindings.h"

#include <algorithm>
#include <string>

namespace pih {

Result<DeepSeekHashRouterWeightBindings>
DeepSeekHashRouterWeightBindings::Resolve(
    DeepSeekStageRange owned_layers, const Resolver& resolver) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || !resolver) {
    return Status::InvalidArgument(
        "DeepSeek hash router weight binding identity is invalid");
  }
  DeepSeekHashRouterWeightBindings result;
  const auto last = std::min(owned_layers.last_layer, 2U);
  if (owned_layers.first_layer > last) return result;
  result.bindings_.reserve(last - owned_layers.first_layer + 1U);
  for (auto layer = owned_layers.first_layer; layer <= last; ++layer) {
    auto view = resolver("layers." + std::to_string(layer) +
                         ".ffn.gate.weight");
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
          "DeepSeek hash router weight violates checkpoint ABI");
    }
    result.generation_ = view->generation();
    result.bindings_.push_back(
        {layer, reinterpret_cast<std::uintptr_t>(view->data())});
  }
  return result;
}

Result<DeepSeekHashRouterWeightBindings>
DeepSeekHashRouterWeightBindings::Resolve(
    DeepSeekStageRange owned_layers,
    const DeepSeekResidentWeightArena& arena) {
  auto result = Resolve(owned_layers, [&arena](std::string_view name) {
    return arena.tensor(name);
  });
  if (!result.ok()) return result.status();
  if (result->size() != 0 && result->generation() != arena.generation()) {
    return Status::FailedPrecondition(
        "DeepSeek hash router weight generation is foreign");
  }
  return result;
}

}  // namespace pih
