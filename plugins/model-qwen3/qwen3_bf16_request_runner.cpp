#include "pih/model/qwen3_bf16_request_runner.h"

#include <limits>

namespace pih {

Result<QwenBf16RequestRunner> QwenBf16RequestRunner::Create(
    QwenKvSlotPool& pool, QwenBf16SequenceBackend& backend,
    QwenBf16CompletionIdentityProvider& completion,
    QwenBf16KvRecycler& recycler, std::int64_t eos_token_id,
    std::uint32_t maximum_step_tokens) {
  if (!pool.ready() || pool.failed() || eos_token_id < 0 ||
      eos_token_id >= QwenBf16SequenceSession::kVocabularySize ||
      maximum_step_tokens == 0 || maximum_step_tokens > 4096) {
    return Status::FailedPrecondition(
        "Qwen request runner dependencies are not ready");
  }
  return QwenBf16RequestRunner(pool, backend, completion, recycler,
                              eos_token_id, maximum_step_tokens);
}

Result<std::vector<std::int64_t>> QwenBf16RequestRunner::generate(
    std::span<const std::int64_t> prompt,
    std::uint32_t maximum_new_tokens) {
  if (state_ != QwenBf16RequestRunnerState::kReady) {
    return Status::FailedPrecondition("Qwen request runner is not ready");
  }
  if (prompt.empty() || maximum_new_tokens == 0 ||
      prompt.size() >= QwenBf16GreedyDecode::kMaximumPositions ||
      maximum_new_tokens >
          QwenBf16GreedyDecode::kMaximumPositions - prompt.size()) {
    return Status::InvalidArgument(
        "Qwen request exceeds its bounded context");
  }
  for (const std::int64_t token : prompt) {
    if (token < 0 || token >= QwenBf16SequenceSession::kVocabularySize) {
      return Status::InvalidArgument(
          "Qwen request prompt contains a token outside the vocabulary");
    }
  }
  const auto reserved_tokens = static_cast<std::uint32_t>(
      prompt.size() + maximum_new_tokens);
  auto sequence_generation = pool_->acquire_sequence_generation();
  if (!sequence_generation.ok()) {
    state_ = QwenBf16RequestRunnerState::kPoisoned;
    return sequence_generation.status();
  }
  auto session = QwenBf16SequenceSession::Admit(
      *pool_, 0, *sequence_generation, reserved_tokens, *backend_);
  if (!session.ok()) return session.status();
  const auto handles = std::vector<QwenKvBlockHandle>(
      session->reserved_handles().begin(), session->reserved_handles().end());

  state_ = QwenBf16RequestRunnerState::kRunning;
  std::vector<std::int64_t> output;
  output.reserve(maximum_new_tokens);
  std::size_t offset = 0;
  while (offset < prompt.size()) {
    const std::size_t remaining = prompt.size() - offset;
    const std::size_t count =
        remaining >= maximum_step_tokens_ ? maximum_step_tokens_ : 1;
    auto token = session->execute(prompt.subspan(offset, count), offset);
    if (!token.ok()) {
      state_ = QwenBf16RequestRunnerState::kPoisoned;
      return token.status();
    }
    offset += count;
    if (offset == prompt.size()) output.push_back(*token);
  }
  while (output.back() != eos_token_id_ &&
         output.size() < maximum_new_tokens) {
    const std::int64_t previous = output.back();
    auto token = session->execute(
        std::span<const std::int64_t>(&previous, 1),
        prompt.size() + output.size() - 1);
    if (!token.ok()) {
      state_ = QwenBf16RequestRunnerState::kPoisoned;
      return token.status();
    }
    output.push_back(*token);
  }
  auto event = completion_->last_completion_event();
  if (!event.ok()) {
    state_ = QwenBf16RequestRunnerState::kPoisoned;
    return event.status();
  }
  const Status released = session->release(*event);
  if (!released.ok()) {
    state_ = QwenBf16RequestRunnerState::kPoisoned;
    return released;
  }
  const Status recycled = recycler_->recycle(*pool_, handles, *event);
  if (!recycled.ok()) {
    state_ = QwenBf16RequestRunnerState::kPoisoned;
    return recycled;
  }
  state_ = QwenBf16RequestRunnerState::kReady;
  return output;
}

Status QwenBf16RequestRunner::close() {
  if (state_ == QwenBf16RequestRunnerState::kRunning) {
    return Status::FailedPrecondition(
        "Qwen request runner cannot close while running");
  }
  state_ = QwenBf16RequestRunnerState::kClosed;
  return Status::Ok();
}

}  // namespace pih
