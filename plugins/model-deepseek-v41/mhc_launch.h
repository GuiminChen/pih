#pragma once
#include "engram_launch.h"
#include "norm_launch.h"

namespace pih::deepseek_v41 {
// Same contiguous device-region contract as Engram; all storage is caller owned.
// Fixed V4.1: four streams, 5120 features, 24 FP32 mixing rows, 20 Sinkhorn steps.
struct MhcMixLaunch final {
  EngramDeviceRegion residual, fn, scale, base, pre, post, comb, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0;
};
struct MhcPreLaunch final {
  // pre is explicitly supplied from the preceding sub-block, never implicitly
  // replaced by the coefficients computed for the current sub-block.
  EngramDeviceRegion residual, pre, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0;
};
struct MhcPostLaunch final {
  EngramDeviceRegion sublayer, residual, post, comb, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0;
};
Status ValidateMhcMix(const MhcMixLaunch& launch);
Status ValidateMhcPre(const MhcPreLaunch& launch);
Status ValidateMhcPost(const MhcPostLaunch& launch);
Status LaunchMhcMix(const MhcMixLaunch& launch);
Status LaunchMhcPre(const MhcPreLaunch& launch);
Status LaunchMhcPost(const MhcPostLaunch& launch);
// Produces [1,0,0,0] for every token at the beginning of a residual stream.
Status LaunchMhcInitialPre(EngramDeviceRegion pre, std::uint32_t tokens, std::uintptr_t stream);
Status ValidateMhcInitialPre(EngramDeviceRegion pre, std::uint32_t tokens, std::uintptr_t stream);
struct MhcInputLaunch final {
  MhcPreLaunch collapse;
  RmsNormLaunch norm;
};
// Preserve BF16 rounding between pre-collapse and sublayer normalization.
Status ValidateMhcInput(const MhcInputLaunch& launch);
Status LaunchMhcInput(const MhcInputLaunch& launch);
struct MhcSublayerInputLaunch final {
  MhcMixLaunch mix;
  // collapse.pre is the carried input coefficient, not mix.pre. mix.pre is
  // retained for the next sublayer; mix.post/comb are for this residual return.
  MhcInputLaunch input;
};
Status ValidateMhcSublayerInput(const MhcSublayerInputLaunch& launch);
Status LaunchMhcSublayerInput(const MhcSublayerInputLaunch& launch);
}  // namespace pih::deepseek_v41
