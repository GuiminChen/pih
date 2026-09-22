#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "pih/core/result.h"
#include "pih/model/qwen3_bf16_execution_schedule.h"
#include "pih/model/qwen3_bf16_kernel_manifest.h"
#include "pih/model/qwen3_bf16_linear_shape.h"
#include "pih/model/qwen3_bf16_weight_binding.h"

namespace pih {

enum class QwenBf16CommandBackend : std::uint8_t {
  kKernel,
  kLinear,
};

struct QwenBf16PreparedCommand final {
  static constexpr std::uint16_t kNoTensor = UINT16_MAX;

  QwenBf16CommandBackend backend;
  std::uint16_t logical_step;
  std::uint8_t subcommand;
  std::uint16_t tensor_index;
  QwenBf16Primitive primitive;
  QwenBf16LinearKind linear_kind;
  QwenBf16ExecutionStep execution_step;

  [[nodiscard]] bool has_weight() const noexcept {
    return tensor_index != kNoTensor;
  }
};

class QwenBf16CommandBuffer final {
 public:
  static constexpr std::size_t kCommandCount = 509;
  static constexpr std::size_t kKernelCommandCount = 312;
  static constexpr std::size_t kLinearCommandCount = 197;
  static constexpr std::size_t kCanonicalHeaderBytes = 16;
  static constexpr std::size_t kCanonicalCommandBytes = 16;
  static constexpr std::size_t kCanonicalWireBytes =
      kCanonicalHeaderBytes + kCommandCount * kCanonicalCommandBytes;
  using CanonicalWire = std::array<std::byte, kCanonicalWireBytes>;

  static Result<QwenBf16CommandBuffer> Create(
      const QwenBf16ExecutionSchedule& schedule,
      const QwenBf16WeightBindingPlan& weights);

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return commands_.size();
  }
  [[nodiscard]] constexpr const QwenBf16PreparedCommand& operator[](
      std::size_t index) const noexcept {
    return commands_[index];
  }
  [[nodiscard]] constexpr auto begin() const noexcept {
    return commands_.begin();
  }
  [[nodiscard]] constexpr auto end() const noexcept { return commands_.end(); }
  [[nodiscard]] CanonicalWire canonical_wire() const noexcept;
  [[nodiscard]] constexpr std::uint16_t command_begin(
      std::size_t logical_step) const noexcept {
    return offsets_[logical_step];
  }
  [[nodiscard]] constexpr std::uint16_t command_end(
      std::size_t logical_step) const noexcept {
    return offsets_[logical_step + 1];
  }

 private:
  std::array<QwenBf16PreparedCommand, kCommandCount> commands_{};
  std::array<std::uint16_t, QwenBf16ExecutionSchedule::kStepCount + 1>
      offsets_{};
};

}  // namespace pih
