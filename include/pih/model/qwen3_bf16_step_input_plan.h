#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/qwen3_kv_block_table.h"

namespace pih {

class QwenBf16StepInputPlan final {
 public:
  static constexpr std::uint32_t kMaximumStepTokens = 4096;

  static Result<QwenBf16StepInputPlan> Create(
      std::span<const std::int64_t> tokens, std::uint64_t first_position,
      const QwenKvBlockTable& block_table,
      const QwenKvAppendPlan& append_plan);

  [[nodiscard]] std::span<const std::int64_t> tokens() const noexcept {
    return tokens_;
  }
  [[nodiscard]] std::span<const std::int64_t> positions() const noexcept {
    return positions_;
  }
  [[nodiscard]] std::span<const QwenKvBlockHandle> append_handles()
      const noexcept {
    return append_handles_;
  }
  [[nodiscard]] std::span<const std::uint16_t> token_offsets()
      const noexcept {
    return token_offsets_;
  }
  [[nodiscard]] std::span<const QwenKvBlockHandle> visible_handles()
      const noexcept {
    return visible_handles_;
  }
  [[nodiscard]] std::uint32_t key_token_count() const noexcept {
    return key_token_count_;
  }

 private:
  std::vector<std::int64_t> tokens_;
  std::vector<std::int64_t> positions_;
  std::vector<QwenKvBlockHandle> append_handles_;
  std::vector<std::uint16_t> token_offsets_;
  std::vector<QwenKvBlockHandle> visible_handles_;
  std::uint32_t key_token_count_ = 0;
};

}  // namespace pih
