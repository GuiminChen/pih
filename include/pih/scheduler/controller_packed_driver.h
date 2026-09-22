#pragma once

#include <cstdint>
#include <vector>

#include "pih/model/qwen3_bf16_packed_batch_transaction.h"
#include "pih/model/qwen3_bf16_packed_kv_metadata.h"
#include "pih/model/qwen3_bf16_request_runner.h"

namespace pih::qwen_plugin { class ControllerClient; }

namespace pih {

enum class ControllerPackedDriverState : std::uint8_t {
  kReady = 1,
  kCompletionPending = 2,
  kPoisoned = 3,
};

enum class ControllerPackedDriverStep : std::uint8_t {
  kIdle = 1,
  kCompleted = 2,
  kOutputBackpressured = 3,
};

struct ControllerPackedSamplingReceipt final {
  std::uint64_t request_generation;
  std::uint64_t token_ordinal;
  QwenBf16PackedSampleReceipt sample;
};

class ControllerPackedDriver final {
 public:
  static Result<ControllerPackedDriver> Create(
      qwen_plugin::ControllerClient& runtime, QwenBf16PackedBatchBackend& backend,
      std::uint32_t maximum_sequences,
      QwenBf16PackedKvMetadataLimits kv_metadata_limits,
      std::uint32_t receipt_capacity = 128);

  Status bind_sequence(std::uint64_t sequence_generation,
                       QwenKvBlockTable& block_table,
                       QwenKvCompletionEvent initial_last_use_event);
  Result<ControllerPackedDriverStep> execute_next(std::int64_t now_ns);
  Result<std::optional<ControllerPackedSamplingReceipt>>
  try_take_sampling_receipt();
  Status drain_sequence(std::uint64_t sequence_generation,
                        QwenKvSlotPool& pool,
                        QwenBf16KvRecycler& recycler);

  [[nodiscard]] ControllerPackedDriverState state() const noexcept {
    return state_;
  }

 private:
  ControllerPackedDriver(qwen_plugin::ControllerClient& runtime,
                         QwenBf16PackedBatchBackend& backend,
                         QwenBf16PackedBatchTransaction transaction,
                         QwenBf16PackedKvMetadataArena metadata,
                         std::uint32_t maximum_sequences,
                         std::uint32_t receipt_capacity)
      : runtime_(&runtime), backend_(&backend),
        transaction_(std::move(transaction)), metadata_(std::move(metadata)),
        registry_(maximum_sequences), active_bindings_(maximum_sequences),
        active_sampling_(maximum_sequences),
        pending_tokens_(maximum_sequences),
        pending_receipts_(maximum_sequences), receipts_(receipt_capacity),
        last_use_events_(maximum_sequences), reclaimed_(maximum_sequences) {}

  Result<ControllerPackedDriverStep> publish_pending(std::int64_t now_ns);

  qwen_plugin::ControllerClient* runtime_;
  QwenBf16PackedBatchBackend* backend_;
  QwenBf16PackedBatchTransaction transaction_;
  QwenBf16PackedKvMetadataArena metadata_;
  std::vector<QwenBf16PackedKvBinding> registry_;
  std::vector<QwenBf16PackedKvBinding> active_bindings_;
  std::vector<Qwen3SamplingDescriptor> active_sampling_;
  std::vector<std::uint32_t> pending_tokens_;
  std::vector<ControllerPackedSamplingReceipt> pending_receipts_;
  std::vector<ControllerPackedSamplingReceipt> receipts_;
  std::vector<QwenKvCompletionEvent> last_use_events_;
  std::vector<bool> reclaimed_;
  std::uint32_t registry_count_ = 0;
  std::uint32_t pending_token_count_ = 0;
  std::uint32_t receipt_head_ = 0;
  std::uint32_t receipt_size_ = 0;
  ControllerPackedDriverState state_ = ControllerPackedDriverState::kReady;
};

}  // namespace pih
