#pragma once
#include "config.h"
#include "mhc_launch.h"
#include "router.h"
#include "expert_counts.h"

namespace pih::deepseek_v41 {
struct FfnRouteLaunch final {
  MhcSublayerInputLaunch input;
  RouterLaunch router;
  ExpertDispatchLaunch dispatch;
};
Status ValidateFfnRoute(const FlashConfig& config, const FfnRouteLaunch& launch);
// Borrowed buffers and completion storage. Caller initializes error_flag and
// retains every resource until observation or safe retirement, even on failure.
Result<ExpertCounts> LaunchFfnRoute(const FlashConfig& config, const FfnRouteLaunch& launch,
    EngramDeviceRegion host_counts, const EngramCompletionResources& resources,
    ExpertCounts::Clock::time_point deadline);
}  // namespace pih::deepseek_v41
