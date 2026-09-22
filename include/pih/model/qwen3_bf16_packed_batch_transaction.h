#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/packed_token_metadata_arena.h"
#include "pih/model/qwen3_bf16_packed_result_layout.h"
#include "pih/model/qwen3_kv_block_table.h"
#include "pih/model/qwen3_sampler.h"

namespace pih {

class QwenBf16PackedKvMetadataArena;
struct QwenBf16PackedKvMetadataView;

struct QwenBf16PackedKvBinding final {
  std::uint64_t sequence_generation;
  QwenKvBlockTable* block_table;
};

struct QwenBf16PackedBatchExecutionView final {
  std::span<const std::uint32_t> sampled_token_ids;
  QwenKvCompletionEvent completion_event;
  std::span<const QwenBf16PackedSampleReceipt> sampling_receipts;
};

class QwenBf16PackedBatchBackend {
 public:
  virtual ~QwenBf16PackedBatchBackend() = default;
  virtual Result<QwenBf16PackedBatchExecutionView> execute_packed(
      const PackedTokenPlan& plan, PackedTokenMetadataView metadata,
      QwenBf16PackedKvMetadataView kv_metadata,
      std::span<const QwenBf16PackedKvBinding> bindings,
      std::span<const QwenKvAppendPlan> append_plans,
      std::span<const Qwen3SamplingDescriptor> sampling) = 0;
};

enum class QwenBf16PackedBatchTransactionState : std::uint8_t {
  kReady = 1,
  kPoisoned = 2,
};

class QwenBf16PackedBatchTransaction final {
 public:
  static Result<QwenBf16PackedBatchTransaction> Create(
      std::uint32_t maximum_sequences);

  Result<QwenBf16PackedBatchExecutionView> execute(
      const PackedTokenPlan& plan, PackedTokenMetadataView metadata,
      std::span<const QwenBf16PackedKvBinding> bindings,
      std::span<const Qwen3SamplingDescriptor> sampling,
      QwenBf16PackedKvMetadataArena& kv_metadata_arena,
      QwenBf16PackedBatchBackend& backend);

  [[nodiscard]] QwenBf16PackedBatchTransactionState state() const noexcept {
    return state_;
  }

 private:
  explicit QwenBf16PackedBatchTransaction(std::uint32_t maximum_sequences)
      : append_plans_(maximum_sequences) {}

  std::vector<QwenKvAppendPlan> append_plans_;
  QwenBf16PackedBatchTransactionState state_ =
      QwenBf16PackedBatchTransactionState::kReady;
};

}  // namespace pih
