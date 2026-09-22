#pragma once

#include <vector>

#include "pih/model/qwen3_bf16_packed_batch_transaction.h"
#include "pih/model/qwen3_bf16_synchronous_backend.h"

namespace pih {

enum class QwenInt4SerializedPackedBackendState : std::uint8_t {
  kReady = 1,
  kRunning = 2,
  kPoisoned = 3,
};

// Compatibility backend for M2: preserves packed controller/KV commit
// semantics while executing the existing INT4 sequence kernel path serially
// on one CUDA stream. It intentionally accepts greedy sampling only.
class QwenInt4SerializedPackedBackend final
    : public QwenBf16PackedBatchBackend {
 public:
  static Result<QwenInt4SerializedPackedBackend> Create(
      QwenBf16SequenceBackend& sequence_backend,
      QwenBf16CompletionIdentityProvider& completion_provider,
      std::uint32_t maximum_sequences);

  QwenInt4SerializedPackedBackend(
      const QwenInt4SerializedPackedBackend&) = delete;
  QwenInt4SerializedPackedBackend(
      QwenInt4SerializedPackedBackend&&) noexcept = default;

  Result<QwenBf16PackedBatchExecutionView> execute_packed(
      const PackedTokenPlan& plan, PackedTokenMetadataView metadata,
      QwenBf16PackedKvMetadataView kv_metadata,
      std::span<const QwenBf16PackedKvBinding> bindings,
      std::span<const QwenKvAppendPlan> append_plans,
      std::span<const Qwen3SamplingDescriptor> sampling) override;

  [[nodiscard]] QwenInt4SerializedPackedBackendState state() const noexcept {
    return state_;
  }

 private:
  QwenInt4SerializedPackedBackend(
      QwenBf16SequenceBackend& sequence_backend,
      QwenBf16CompletionIdentityProvider& completion_provider,
      std::uint32_t maximum_sequences)
      : sequence_backend_(&sequence_backend),
        completion_provider_(&completion_provider) {
    sampled_tokens_.reserve(maximum_sequences);
    sampling_receipts_.reserve(maximum_sequences);
  }

  QwenBf16SequenceBackend* sequence_backend_;
  QwenBf16CompletionIdentityProvider* completion_provider_;
  std::vector<std::uint32_t> sampled_tokens_;
  std::vector<QwenBf16PackedSampleReceipt> sampling_receipts_;
  QwenInt4SerializedPackedBackendState state_ =
      QwenInt4SerializedPackedBackendState::kReady;
};

}  // namespace pih
