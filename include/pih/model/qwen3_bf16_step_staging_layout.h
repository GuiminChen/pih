#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/model/qwen3_bf16_execution_arena.h"
#include "pih/model/qwen3_bf16_step_input_plan.h"

namespace pih {

class QwenBf16StepStagingLayout final {
 public:
  static constexpr std::uint64_t kAlignment = 256;

  static Result<QwenBf16StepStagingLayout> Create(
      const QwenBf16StepInputPlan& input);
  static Result<QwenBf16StepStagingLayout> CreateBounded(
      std::uint64_t token_count, std::uint64_t visible_handle_count);

  Status materialize(const QwenBf16StepInputPlan& input,
                     std::span<std::byte> backing) const;

  [[nodiscard]] QwenBf16ArenaSpan token_ids() const noexcept {
    return token_ids_;
  }
  [[nodiscard]] QwenBf16ArenaSpan positions() const noexcept {
    return positions_;
  }
  [[nodiscard]] QwenBf16ArenaSpan append_handles() const noexcept {
    return append_handles_;
  }
  [[nodiscard]] QwenBf16ArenaSpan token_offsets() const noexcept {
    return token_offsets_;
  }
  [[nodiscard]] QwenBf16ArenaSpan visible_handles() const noexcept {
    return visible_handles_;
  }
  [[nodiscard]] std::uint64_t total_bytes() const noexcept {
    return total_bytes_;
  }
  [[nodiscard]] std::uint64_t token_count() const noexcept {
    return token_count_;
  }
  [[nodiscard]] std::uint64_t visible_handle_count() const noexcept {
    return visible_handle_count_;
  }

 private:
  QwenBf16ArenaSpan token_ids_{};
  QwenBf16ArenaSpan positions_{};
  QwenBf16ArenaSpan append_handles_{};
  QwenBf16ArenaSpan token_offsets_{};
  QwenBf16ArenaSpan visible_handles_{};
  std::uint64_t total_bytes_ = 0;
  std::uint64_t token_count_ = 0;
  std::uint64_t visible_handle_count_ = 0;
};

}  // namespace pih
