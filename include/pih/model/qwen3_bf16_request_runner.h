#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/qwen3_bf16_sequence_session.h"

namespace pih {

class QwenBf16CompletionIdentityProvider {
 public:
  virtual ~QwenBf16CompletionIdentityProvider() = default;
  virtual Result<QwenKvCompletionEvent> last_completion_event() const = 0;
};

class QwenBf16KvRecycler {
 public:
  virtual ~QwenBf16KvRecycler() = default;
  virtual Status recycle(QwenKvSlotPool& pool,
                         std::span<const QwenKvBlockHandle> handles,
                         QwenKvCompletionEvent last_use_event) = 0;
};

enum class QwenBf16RequestRunnerState : std::uint8_t {
  kReady,
  kRunning,
  kPoisoned,
  kClosed,
};

class QwenBf16RequestRunner final {
 public:
  static Result<QwenBf16RequestRunner> Create(
      QwenKvSlotPool& pool, QwenBf16SequenceBackend& backend,
      QwenBf16CompletionIdentityProvider& completion,
      QwenBf16KvRecycler& recycler, std::int64_t eos_token_id,
      std::uint32_t maximum_step_tokens = 4096);

  Result<std::vector<std::int64_t>> generate(
      std::span<const std::int64_t> prompt,
      std::uint32_t maximum_new_tokens);
  Status close();

  [[nodiscard]] QwenBf16RequestRunnerState state() const noexcept {
    return state_;
  }

 private:
  QwenBf16RequestRunner(QwenKvSlotPool& pool,
                        QwenBf16SequenceBackend& backend,
                        QwenBf16CompletionIdentityProvider& completion,
                        QwenBf16KvRecycler& recycler,
                        std::int64_t eos_token_id,
                        std::uint32_t maximum_step_tokens)
      : pool_(&pool), backend_(&backend), completion_(&completion),
        recycler_(&recycler), eos_token_id_(eos_token_id),
        maximum_step_tokens_(maximum_step_tokens) {}

  QwenKvSlotPool* pool_;
  QwenBf16SequenceBackend* backend_;
  QwenBf16CompletionIdentityProvider* completion_;
  QwenBf16KvRecycler* recycler_;
  std::int64_t eos_token_id_;
  std::uint32_t maximum_step_tokens_;
  QwenBf16RequestRunnerState state_ = QwenBf16RequestRunnerState::kReady;
};

}  // namespace pih
