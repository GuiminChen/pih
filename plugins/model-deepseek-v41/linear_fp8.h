#pragma once
#include "engram_launch.h"

namespace pih::deepseek_v41 {
struct Fp8LinearLaunch final {
  // BF16 input [rows,K], E4M3FN weight [N,K], E8M0 weight scale [N/32,K/32].
  // Scratch: E4M3FN [rows,K], E8M0 [rows,K/32]; BF16 output [rows,N].
  EngramDeviceRegion input, weight, weight_scales, quantized, activation_scales, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t rows = 0, in_features = 0, out_features = 0;
};
Status ValidateFp8Linear(const Fp8LinearLaunch& launch);
Status LaunchFp8Linear(const Fp8LinearLaunch& launch);
}  // namespace pih::deepseek_v41
