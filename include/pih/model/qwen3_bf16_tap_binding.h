#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/qwen3_bf16_activation_route.h"
#include "pih/model/qwen3_bf16_command_buffer.h"
#include "pih/model/qwen3_numerical_tap_plan.h"

namespace pih {

enum class QwenBf16TapKvComponent : std::uint8_t {
  kNotApplicable = 0,
  kKey,
  kValue,
};

struct QwenBf16TapBinding final {
  std::size_t capture_index;
  QwenNumericalTapRequest request;
  std::size_t producer_command_index;
  QwenBf16ExecutionStep producer;
  std::uint8_t producer_subcommand;
  QwenBf16ActivationSlot output_slot;
  QwenBf16TapKvComponent kv_component;
};

class QwenBf16TapBindingPlan final {
 public:
  static Result<QwenBf16TapBindingPlan> Create(
      const QwenNumericalTapPlan& taps,
      const QwenBf16CommandBuffer& commands);

  [[nodiscard]] std::size_t size() const noexcept { return bindings_.size(); }
  [[nodiscard]] const QwenBf16TapBinding& operator[](
      std::size_t index) const noexcept {
    return bindings_[index];
  }
  [[nodiscard]] auto begin() const noexcept { return bindings_.begin(); }
  [[nodiscard]] auto end() const noexcept { return bindings_.end(); }
  [[nodiscard]] std::span<const std::uint16_t> bindings_after_command(
      std::size_t command_index) const noexcept;

 private:
  QwenBf16TapBindingPlan(
      std::vector<QwenBf16TapBinding> bindings,
      std::array<std::uint16_t, QwenBf16CommandBuffer::kCommandCount + 1>
          command_offsets,
      std::vector<std::uint16_t> ordered_binding_indices)
      : bindings_(std::move(bindings)),
        command_offsets_(command_offsets),
        ordered_binding_indices_(std::move(ordered_binding_indices)) {}

  std::vector<QwenBf16TapBinding> bindings_;
  std::array<std::uint16_t, QwenBf16CommandBuffer::kCommandCount + 1>
      command_offsets_{};
  std::vector<std::uint16_t> ordered_binding_indices_;
};

}  // namespace pih
