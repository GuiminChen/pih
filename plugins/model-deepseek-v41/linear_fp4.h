#pragma once
#include "engram_launch.h"

namespace pih::deepseek_v41 {
struct Fp4LinearLaunch final {
  // BF16 [M,K], packed E2M1 [N,K/2] (even K in low nibble), E8M0 [N,K/32].
  // Activation scratch remains E4M3 [M,K] and E8M0 [M,K/32], NOT FP4.
  EngramDeviceRegion input, weight, weight_scales, quantized, activation_scales, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t rows = 0, in_features = 0, out_features = 0;
};
Status ValidateFp4Linear(const Fp4LinearLaunch& launch);
Status LaunchFp4Linear(const Fp4LinearLaunch& launch);
}  // namespace pih::deepseek_v41
