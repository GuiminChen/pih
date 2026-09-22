#pragma once

#include <cstdint>
#include <span>

#include "pih/model/qwen3_bf16_greedy_decode.h"
#include "pih/model/qwen3_kv_sequence_lease.h"

namespace pih {

enum class QwenBf16SequenceSessionState : std::uint8_t {
  kReady,
  kRunning,
  kPoisoned,
  kReleased,
};

class QwenBf16SequenceBackend {
 public:
  virtual ~QwenBf16SequenceBackend() = default;
  virtual Result<std::int64_t> execute_and_read_token(
      std::span<const std::int64_t> tokens, std::uint64_t first_position,
      const QwenKvBlockTable& block_table,
      const QwenKvAppendPlan& append_plan) = 0;
};

class QwenBf16SequenceSession final : public QwenBf16GreedyStepDriver {
 public:
  static constexpr std::int64_t kVocabularySize = 151936;

  static Result<QwenBf16SequenceSession> Admit(
      QwenKvSlotPool& pool, std::uint32_t owner_sequence_index,
      std::uint32_t sequence_generation, std::uint32_t reserved_tokens,
      QwenBf16SequenceBackend& backend);

  QwenBf16SequenceSession(const QwenBf16SequenceSession&) = delete;
  QwenBf16SequenceSession& operator=(const QwenBf16SequenceSession&) = delete;
  QwenBf16SequenceSession(QwenBf16SequenceSession&&) noexcept = default;
  QwenBf16SequenceSession& operator=(QwenBf16SequenceSession&&) = delete;

  Result<std::int64_t> execute(std::span<const std::int64_t> tokens,
                               std::uint64_t first_position) override;
  Status release(QwenKvCompletionEvent last_use_event);

  [[nodiscard]] QwenBf16SequenceSessionState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint32_t committed_tokens() const noexcept {
    return lease_.block_table().descriptor().committed_tokens;
  }
  [[nodiscard]] std::span<const QwenKvBlockHandle> reserved_handles()
      const noexcept {
    return lease_.block_table().reserved_handles();
  }

 private:
  QwenBf16SequenceSession(QwenKvSequenceLease lease,
                          QwenBf16SequenceBackend& backend)
      : lease_(std::move(lease)), backend_(&backend) {}

  QwenKvSequenceLease lease_;
  QwenBf16SequenceBackend* backend_;
  QwenBf16SequenceSessionState state_ =
      QwenBf16SequenceSessionState::kReady;
};

}  // namespace pih
