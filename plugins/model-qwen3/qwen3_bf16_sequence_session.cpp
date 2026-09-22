#include "pih/model/qwen3_bf16_sequence_session.h"

#include <limits>

namespace pih {

Result<QwenBf16SequenceSession> QwenBf16SequenceSession::Admit(
    QwenKvSlotPool& pool, std::uint32_t owner_sequence_index,
    std::uint32_t sequence_generation, std::uint32_t reserved_tokens,
    QwenBf16SequenceBackend& backend) {
  auto lease = QwenKvSequenceLease::Admit(
      pool, owner_sequence_index, sequence_generation, reserved_tokens);
  if (!lease.ok()) return lease.status();
  return QwenBf16SequenceSession(std::move(*lease), backend);
}

Result<std::int64_t> QwenBf16SequenceSession::execute(
    std::span<const std::int64_t> tokens, std::uint64_t first_position) {
  if (state_ == QwenBf16SequenceSessionState::kPoisoned ||
      state_ == QwenBf16SequenceSessionState::kReleased) {
    return Status::FailedPrecondition("Qwen sequence session is not runnable");
  }
  const std::uint32_t committed = committed_tokens();
  if (tokens.empty() || first_position != committed ||
      tokens.size() > std::numeric_limits<std::uint32_t>::max() ||
      tokens.size() > QwenKvBlockTable::kMaximumReservedTokens - committed) {
    state_ = QwenBf16SequenceSessionState::kPoisoned;
    return Status::InvalidArgument(
        "Qwen sequence step does not extend the committed prefix exactly");
  }
  const auto target =
      committed + static_cast<std::uint32_t>(tokens.size());
  auto append = lease_.prepare_append(target);
  if (!append.ok()) {
    state_ = QwenBf16SequenceSessionState::kPoisoned;
    return append.status();
  }
  state_ = QwenBf16SequenceSessionState::kRunning;
  auto token = backend_->execute_and_read_token(tokens, first_position,
                                                lease_.block_table(), *append);
  if (!token.ok() || *token < 0 || *token >= kVocabularySize) {
    const Status rollback = lease_.rollback_append(*append);
    state_ = QwenBf16SequenceSessionState::kPoisoned;
    if (!rollback.ok()) {
      return Status::Internal("Qwen sequence append rollback failed");
    }
    if (!token.ok()) return token.status();
    return Status::Internal(
        "Qwen sequence backend produced a token outside the vocabulary");
  }
  const Status committed_status = lease_.commit_append(*append);
  if (!committed_status.ok()) {
    state_ = QwenBf16SequenceSessionState::kPoisoned;
    return committed_status;
  }
  state_ = QwenBf16SequenceSessionState::kReady;
  return token;
}

Status QwenBf16SequenceSession::release(
    QwenKvCompletionEvent last_use_event) {
  if (state_ == QwenBf16SequenceSessionState::kReleased) {
    return Status::FailedPrecondition("Qwen sequence session is released");
  }
  const Status status = lease_.release(last_use_event);
  if (!status.ok()) {
    state_ = QwenBf16SequenceSessionState::kPoisoned;
    return status;
  }
  state_ = QwenBf16SequenceSessionState::kReleased;
  return Status::Ok();
}

}  // namespace pih
