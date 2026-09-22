#pragma once
#include "expert_fp8.h"
#include "mhc_launch.h"

namespace pih::deepseek_v41 {
struct ExpertResidualLaunch final {
  ExpertSharedMergeLaunch experts;
  MhcPostLaunch residual;
};
Status ValidateExpertResidual(const ExpertResidualLaunch& launch);
// Routed accumulation must already be globally reduced and stream-ordered.
Status LaunchExpertResidual(const ExpertResidualLaunch& launch);
}  // namespace pih::deepseek_v41
