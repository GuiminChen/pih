#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "pih/core/result.h"
#include "pih/model/qwen3_bf16_execution_schedule.h"

namespace pih {

enum class QwenBf16ActivationSlot : std::uint8_t {
  kTokenIds,
  kPositions,
  kHidden,
  kNormalized,
  kQuery,
  kKey,
  kValue,
  kAttention,
  kGate,
  kUp,
  kRopeCosine,
  kRopeSine,
  kKvBacking,
  kKvSlotStates,
  kKvAppendHandles,
  kKvVisibleHandles,
  kKvTokenOffsets,
  kDeviceError,
  kLogits,
  kSampledToken,
  kCount,
};

struct QwenBf16ActivationRoute final {
  static constexpr std::size_t kMaximumReads = 5;
  static constexpr std::size_t kMaximumWrites = 2;

  std::array<QwenBf16ActivationSlot, kMaximumReads> reads{};
  std::array<QwenBf16ActivationSlot, kMaximumWrites> writes{};
  std::uint8_t read_count = 0;
  std::uint8_t write_count = 0;
};

class QwenBf16ActivationRoutePlan final {
 public:
  static Result<QwenBf16ActivationRoutePlan> Create(
      const QwenBf16ExecutionSchedule& schedule);

  [[nodiscard]] const QwenBf16ActivationRoute& operator[](
      std::size_t step) const noexcept {
    return routes_[step];
  }

 private:
  std::array<QwenBf16ActivationRoute, QwenBf16ExecutionSchedule::kStepCount>
      routes_{};
};

}  // namespace pih
