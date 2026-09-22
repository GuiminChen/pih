#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "pih/model/qwen3_kv_block_table.h"

namespace pih {

enum class QwenKvAdmissionState : std::uint8_t {
  kReserved = 1,
  kPublished = 2,
  kRolledBack = 3,
  kFailed = 4,
};

class QwenKvAdmissionTransaction final {
 public:
  static Result<QwenKvAdmissionTransaction> Reserve(
      QwenKvSlotPool& pool, std::uint32_t owner_sequence_index,
      std::uint32_t reserved_tokens);

  QwenKvAdmissionTransaction(const QwenKvAdmissionTransaction&) = delete;
  QwenKvAdmissionTransaction& operator=(const QwenKvAdmissionTransaction&) =
      delete;
  QwenKvAdmissionTransaction(QwenKvAdmissionTransaction&&) noexcept = default;
  QwenKvAdmissionTransaction& operator=(
      QwenKvAdmissionTransaction&&) noexcept = default;

  Status publish();
  Status rollback();

  [[nodiscard]] QwenKvAdmissionState state() const noexcept { return state_; }
  [[nodiscard]] std::uint32_t sequence_generation() const noexcept {
    return sequence_generation_;
  }
  [[nodiscard]] const QwenKvBlockTable* block_table() const noexcept {
    return table_ ? &*table_ : nullptr;
  }
  [[nodiscard]] QwenKvBlockTable* block_table() noexcept {
    return table_ ? &*table_ : nullptr;
  }

 private:
  QwenKvAdmissionTransaction(QwenKvSlotPool& pool,
                             std::uint32_t owner_sequence_index,
                             std::uint32_t sequence_generation,
                             std::uint32_t reserved_tokens,
                             std::vector<QwenKvBlockHandle> handles)
      : pool_(&pool), owner_sequence_index_(owner_sequence_index),
        sequence_generation_(sequence_generation),
        reserved_tokens_(reserved_tokens), handles_(std::move(handles)) {}

  QwenKvSlotPool* pool_;
  std::uint32_t owner_sequence_index_;
  std::uint32_t sequence_generation_;
  std::uint32_t reserved_tokens_;
  std::vector<QwenKvBlockHandle> handles_;
  std::optional<QwenKvBlockTable> table_;
  QwenKvAdmissionState state_ = QwenKvAdmissionState::kReserved;
};

}  // namespace pih
