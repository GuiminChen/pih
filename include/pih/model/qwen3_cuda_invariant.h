#pragma once

#include <cstdint>

namespace pih {

// Stable device-to-host error ABI for qwen_bf16_primitives.cubin.
enum class QwenCudaInvariant : std::uint32_t {
  kNone = 0,
  kKvAppendLaunch = 1,
  kKvAppendSlotRange = 2,
  kKvAppendHandleState = 3,
  kPagedGqaLaunch = 4,
  kPagedGqaSlotRange = 5,
  kPagedGqaHandleState = 6,
  kPagedGqaNonfiniteScore = 7,
  kPagedGqaSoftmaxDenominator = 8,
  kPagedGqaNonfiniteOutput = 9,
  kArgmaxNonfiniteLogit = 10,
  kW4A16Launch = 11,
  kW4A16ReservedNibble = 12,
  kW4A16InvalidScale = 13,
  kW4A16NonfiniteOutput = 14,
  kPackedSampleLaunch = 15,
  kPackedSampleRow = 16,
  kPackedArgmaxLaunch = 17,
  kPackedSamplerLaunch = 18,
  kPackedSamplerDescriptor = 19,
  kTeacherForcedMetricLaunch = 20,
};

constexpr bool qwen_cuda_invariant_known(std::uint32_t code) noexcept {
  return code <= static_cast<std::uint32_t>(
                     QwenCudaInvariant::kTeacherForcedMetricLaunch);
}

}  // namespace pih
