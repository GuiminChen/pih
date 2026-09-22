#pragma once
#include "expert_dispatch.h"
#include "expert_fp8.h"
#include <optional>

namespace pih::deepseek_v41 {
struct ExpertTokenChainLaunch final {
  ExpertGatherLaunch gather;
  // Absent exactly when gather.rows == 0. Weights must already be admitted
  // for gather.expert; this descriptor cannot authenticate their identity.
  std::optional<RoutedExpertLaunch> expert;
  EngramDeviceRegion accumulator;
};
Status ValidateExpertTokenChain(const ExpertTokenChainLaunch& launch);
// Enqueue only. Owner initializes accumulator, submits every owned expert once
// in fixed order, and admits completion/error before consuming any results.
Status LaunchExpertTokenChain(const ExpertTokenChainLaunch& launch);
}  // namespace pih::deepseek_v41
