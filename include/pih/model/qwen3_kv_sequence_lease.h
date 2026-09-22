#pragma once

#include <cstdint>

#include "pih/core/result.h"
#include "pih/model/qwen3_kv_block_table.h"
#include "pih/model/qwen3_kv_slot_pool.h"

namespace pih {

class QwenKvSequenceLease final {
 public:
  static Result<QwenKvSequenceLease> Admit(
      QwenKvSlotPool& pool, std::uint32_t owner_sequence_index,
      std::uint32_t sequence_generation, std::uint32_t reserved_tokens);

  QwenKvSequenceLease(const QwenKvSequenceLease&) = delete;
  QwenKvSequenceLease& operator=(const QwenKvSequenceLease&) = delete;
  QwenKvSequenceLease(QwenKvSequenceLease&& other) noexcept;
  QwenKvSequenceLease& operator=(QwenKvSequenceLease&&) noexcept = delete;

  Result<QwenKvAppendPlan> prepare_append(
      std::uint32_t target_committed_tokens) const;
  Status commit_append(const QwenKvAppendPlan& plan);
  Status rollback_append(const QwenKvAppendPlan& plan) const;
  Status release(QwenKvCompletionEvent last_use_event);

  [[nodiscard]] bool released() const noexcept { return released_; }
  [[nodiscard]] const QwenKvBlockTable& block_table() const noexcept {
    return table_;
  }

 private:
  QwenKvSequenceLease(QwenKvSlotPool& pool, QwenKvBlockTable table)
      : pool_(&pool), table_(std::move(table)) {}

  QwenKvSlotPool* pool_;
  QwenKvBlockTable table_;
  bool released_ = false;
};

}  // namespace pih
