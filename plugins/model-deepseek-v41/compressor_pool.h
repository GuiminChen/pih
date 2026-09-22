#pragma once
#include "norm_launch.h"
#include <optional>

namespace pih::deepseek_v41 {
struct CompressorPoolLaunch final {
  // One ratio-two source, one sequence. FP32 values/scores [tokens,512],
  // FP32 persistent state [2,512] each; BF16 output [emitted_rows,512].
  // Incomplete steps require output={0,0}; they still update persistent state.
  EngramDeviceRegion values, scores, state_values, state_scores, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t start = 0, tokens = 0;
};
Result<std::uint32_t> CompressorOutputRows(std::uint32_t start, std::uint32_t tokens);
Status ValidateCompressorPool(const CompressorPoolLaunch& launch);
Status LaunchCompressorPool(const CompressorPoolLaunch& launch);
struct CompressorNormalizeLaunch final {
  CompressorPoolLaunch pool;
  // Must be absent on an incomplete step, present with matching rows otherwise.
  std::optional<RmsNormLaunch> norm;
};
Status ValidateCompressorNormalize(const CompressorNormalizeLaunch& launch);
Status LaunchCompressorNormalize(const CompressorNormalizeLaunch& launch);

struct CompressorProjectionLaunch final {
  // BF16 hidden [tokens,5120]. Ratio 1: BF16 weight [512,5120] and
  // BF16 values [tokens,512], with absent gate_weight/scores. Ratio 2:
  // FP32 weights and FP32 values/scores. No bias, quantization or TF32.
  EngramDeviceRegion input, value_weight, gate_weight, values, scores, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0, ratio = 0;
};
Status ValidateCompressorProjection(const CompressorProjectionLaunch& launch);
Status LaunchCompressorProjection(const CompressorProjectionLaunch& launch);
struct CompressorLaunch final {
  CompressorProjectionLaunch projection;
  // Exactly one branch: ratio 1 normalizes every projected token; ratio 2
  // updates persistent state and normalizes only complete groups.
  std::optional<RmsNormLaunch> direct_norm;
  std::optional<CompressorNormalizeLaunch> pooled;
};
Status ValidateCompressor(const CompressorLaunch& launch);
Status LaunchCompressor(const CompressorLaunch& launch);
}  // namespace pih::deepseek_v41
