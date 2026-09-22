#pragma once
#include "engram_launch.h"

namespace pih::deepseek_v41 {
struct RmsNormLaunch final {
  // Contiguous BF16 [rows,width] input/output, one shared [width] weight vector.
  // Frozen text-model widths: index 128, KV 512, Q rank 1280, hidden 5120.
  EngramDeviceRegion input, weight, output, error_flag;
  EngramStorage weight_storage = EngramStorage::kBF16;
  std::uintptr_t stream = 0;
  std::uint32_t rows = 0, width = 0;
};
Status ValidateRmsNorm(const RmsNormLaunch& launch);
// FP32 square mean, rsqrt(mean + 1e-20), normalization and weight product;
// BF16 output. Error bits 1=input/weight non-finite, 2=arithmetic/rounding overflow.
Status LaunchRmsNorm(const RmsNormLaunch& launch);
}  // namespace pih::deepseek_v41
