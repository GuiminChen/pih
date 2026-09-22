#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "pih/core/result.h"
#include "pih/model/qwen3_bf16_execution_schedule.h"
#include "pih/model/qwen3_manifest.h"

namespace pih {

struct QwenBf16WeightBinding final {
  static constexpr std::uint16_t kNoTensor = UINT16_MAX;
  std::uint16_t tensor_index = kNoTensor;

  [[nodiscard]] bool has_weight() const noexcept {
    return tensor_index != kNoTensor;
  }
};

class QwenBf16WeightBindingPlan final {
 public:
  static Result<QwenBf16WeightBindingPlan> Create(
      const QwenBf16ExecutionSchedule& schedule);

  [[nodiscard]] const QwenBf16WeightBinding& operator[](
      std::size_t step) const noexcept {
    return bindings_[step];
  }
  [[nodiscard]] const ExpectedTensor* tensor(std::size_t step) const noexcept;
  [[nodiscard]] std::size_t bound_weight_count() const noexcept {
    return bound_weight_count_;
  }

 private:
  std::array<QwenBf16WeightBinding, QwenBf16ExecutionSchedule::kStepCount>
      bindings_{};
  std::size_t bound_weight_count_ = 0;
};

}  // namespace pih
