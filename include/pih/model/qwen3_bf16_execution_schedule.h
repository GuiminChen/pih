#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "pih/core/result.h"
#include "pih/model/qwen3_bf16_linear_shape.h"
#include "pih/model/qwen3_config.h"

namespace pih {

enum class QwenBf16ExecutionOp : std::uint8_t {
  kEmbedding,
  kPrepareRopeAngles,
  kInputRmsNorm,
  kQueryLinear,
  kKeyLinear,
  kValueLinear,
  kQueryRmsNorm,
  kKeyRmsNorm,
  kRope,
  kKvAppend,
  kPagedGqa,
  kAttentionOutputLinear,
  kAttentionResidual,
  kPostAttentionRmsNorm,
  kGateLinear,
  kUpLinear,
  kSiluMul,
  kDownLinear,
  kMlpResidual,
  kFinalRmsNorm,
  kLmHead,
  kGreedyArgmax,
};

struct QwenBf16ExecutionStep final {
  QwenBf16ExecutionOp operation;
  std::uint32_t layer;

  bool operator==(const QwenBf16ExecutionStep&) const = default;
};

class QwenBf16ExecutionSchedule final {
 public:
  static constexpr std::uint32_t kGlobalLayer =
      std::numeric_limits<std::uint32_t>::max();
  static constexpr std::size_t kLayerCount = 28;
  static constexpr std::size_t kOperationsPerLayer = 17;
  static constexpr std::size_t kStepCount =
      2 + kLayerCount * kOperationsPerLayer + 3;

  static Result<QwenBf16ExecutionSchedule> Create(const Qwen3Config& config);

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return steps_.size();
  }
  [[nodiscard]] constexpr const QwenBf16ExecutionStep& operator[](
      std::size_t index) const noexcept {
    return steps_[index];
  }
  [[nodiscard]] constexpr auto begin() const noexcept { return steps_.begin(); }
  [[nodiscard]] constexpr auto end() const noexcept { return steps_.end(); }

 private:
  explicit QwenBf16ExecutionSchedule(
      std::array<QwenBf16ExecutionStep, kStepCount> steps) noexcept
      : steps_(steps) {}

  std::array<QwenBf16ExecutionStep, kStepCount> steps_;
};

Result<QwenBf16LinearKind> qwen_bf16_linear_kind(
    QwenBf16ExecutionOp operation);

}  // namespace pih
