#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"
#include "pih/model/qwen3_kv_slot_pool.h"

namespace pih {

struct QwenKvSequenceDescriptor final {
  std::uint32_t sequence_generation;
  std::uint32_t owner_sequence_index;
  std::uint32_t reserved_tokens;
  std::uint32_t committed_tokens;
  std::uint32_t handle_count;
  std::uint32_t block_table_generation;
  std::uint32_t active;
  std::array<std::uint32_t, 57> reserved_zero;
};

static_assert(sizeof(QwenKvSequenceDescriptor) == 256);

struct QwenKvAppendPlan final {
  std::uint32_t owner_sequence_index;
  std::uint32_t expected_sequence_generation;
  std::uint32_t expected_block_table_generation;
  std::uint32_t previous_committed_tokens;
  std::uint32_t target_committed_tokens;
  std::uint32_t visible_handle_count_after_commit;
};

class QwenKvBlockTable final {
 public:
  static constexpr std::uint32_t kMaximumReservedTokens = 40'960;

  static Result<QwenKvBlockTable> Create(
      std::uint32_t owner_sequence_index, std::uint32_t sequence_generation,
      std::uint32_t reserved_tokens,
      std::span<const QwenKvBlockHandle> reserved_handles);

  Result<QwenKvAppendPlan> prepare_append(
      std::uint32_t target_committed_tokens) const;
  Status commit_append(const QwenKvAppendPlan& plan);
  Status rollback_append(const QwenKvAppendPlan& plan) const;

  [[nodiscard]] const QwenKvSequenceDescriptor& descriptor() const noexcept {
    return descriptor_;
  }
  [[nodiscard]] std::span<const QwenKvBlockHandle> visible_handles() const;
  [[nodiscard]] std::span<const QwenKvBlockHandle> reserved_handles() const {
    return handles_;
  }

 private:
  QwenKvBlockTable(QwenKvSequenceDescriptor descriptor,
                   std::vector<QwenKvBlockHandle> handles)
      : descriptor_(descriptor), handles_(std::move(handles)) {}

  Status validate_plan(const QwenKvAppendPlan& plan) const;
  static Result<std::uint32_t> handles_for_tokens(std::uint32_t tokens);

  QwenKvSequenceDescriptor descriptor_{};
  std::vector<QwenKvBlockHandle> handles_;
};

}  // namespace pih
