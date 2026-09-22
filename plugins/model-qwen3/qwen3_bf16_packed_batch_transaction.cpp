#include "pih/model/qwen3_bf16_packed_batch_transaction.h"

#include <limits>

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_bf16_packed_kv_metadata.h"

namespace pih {

Result<QwenBf16PackedBatchTransaction>
QwenBf16PackedBatchTransaction::Create(std::uint32_t maximum_sequences) {
  if (maximum_sequences == 0 || maximum_sequences > 4096) {
    return Status::InvalidArgument("packed batch sequence bound is invalid");
  }
  return QwenBf16PackedBatchTransaction(maximum_sequences);
}

Result<QwenBf16PackedBatchExecutionView>
QwenBf16PackedBatchTransaction::execute(
    const PackedTokenPlan& plan, PackedTokenMetadataView metadata,
    std::span<const QwenBf16PackedKvBinding> bindings,
    std::span<const Qwen3SamplingDescriptor> sampling,
    QwenBf16PackedKvMetadataArena& kv_metadata_arena,
    QwenBf16PackedBatchBackend& backend) {
  if (state_ != QwenBf16PackedBatchTransactionState::kReady) {
    return Status::FailedPrecondition("packed batch transaction is poisoned");
  }
  const auto sequence_count = plan.sequence_count();
  if (sequence_count == 0 || sequence_count > append_plans_.size() ||
      bindings.size() != sequence_count ||
      sampling.size() != sequence_count ||
      metadata.real_token_count != plan.total_real_tokens() ||
      metadata.input_token_ids.size() != plan.execution_bucket_tokens() ||
      metadata.positions.size() != metadata.input_token_ids.size() ||
      metadata.request_index.size() != metadata.input_token_ids.size() ||
      metadata.query_start_offsets.size() != sequence_count + 1 ||
      metadata.query_start_offsets.back() != plan.total_real_tokens() ||
      plan.packed_offsets().size() != metadata.query_start_offsets.size()) {
    return Status::InvalidArgument("packed batch metadata shape drifted");
  }
  for (std::size_t i = 0; i < metadata.query_start_offsets.size(); ++i) {
    if (metadata.query_start_offsets[i] != plan.packed_offsets()[i]) {
      return Status::InvalidArgument("packed batch prefix sums drifted");
    }
  }
  std::size_t previous_sample_sequence = 0;
  bool has_previous_sample = false;
  for (const auto sample_row : metadata.sample_row_index) {
    bool matched = false;
    for (std::size_t i = 0; i < sequence_count; ++i) {
      if (sample_row == plan.packed_offsets()[i + 1] - 1) {
        if (has_previous_sample && i <= previous_sample_sequence) {
          return Status::InvalidArgument(
              "packed batch sample rows are not canonical");
        }
        previous_sample_sequence = i;
        has_previous_sample = true;
        matched = true;
        break;
      }
    }
    if (!matched) {
      return Status::InvalidArgument("packed batch sample row is invalid");
    }
  }
  if (plan.phase() != PackedTokenPhase::kPrefill &&
      metadata.sample_row_index.size() != sequence_count) {
    return Status::InvalidArgument(
        "decode-family batch must sample every sequence");
  }
  for (std::size_t i = 0; i < sequence_count; ++i) {
    const auto begin = plan.packed_offsets()[i];
    const auto end = plan.packed_offsets()[i + 1];
    for (std::uint32_t packed = begin; packed < end; ++packed) {
      const auto local = packed - begin;
      if (metadata.input_token_ids[packed] >= 151936 ||
          metadata.request_index[packed] != i ||
          metadata.positions[packed] !=
              plan.committed_start_positions()[i] + local) {
        return Status::InvalidArgument(
            "packed batch real-token metadata drifted");
      }
    }
    auto input_digest = packed_token_input_digest(
        metadata.input_token_ids.subspan(begin, end - begin));
    if (!input_digest.ok()) return input_digest.status();
    if (*input_digest != plan.input_digests()[i]) {
      return Status::InvalidArgument("packed batch input digest drifted");
    }
  }
  for (std::uint32_t packed = plan.total_real_tokens();
       packed < plan.execution_bucket_tokens(); ++packed) {
    if (metadata.input_token_ids[packed] != 0 ||
        metadata.positions[packed] != 0 ||
        metadata.request_index[packed] != kInvalidPackedRequestIndex) {
      return Status::InvalidArgument("packed batch dummy padding drifted");
    }
  }
  for (std::size_t i = 0; i < sequence_count; ++i) {
    const auto& binding = bindings[i];
    if (binding.block_table == nullptr ||
        binding.sequence_generation !=
            plan.ordered_sequence_generations()[i] ||
        binding.block_table->descriptor().active != 1 ||
        binding.block_table->descriptor().committed_tokens !=
            plan.committed_start_positions()[i]) {
      return Status::FailedPrecondition("packed batch KV binding drifted");
    }
    for (std::size_t previous = 0; previous < i; ++previous) {
      if (bindings[previous].block_table == binding.block_table ||
          bindings[previous].block_table->descriptor().owner_sequence_index ==
              binding.block_table->descriptor().owner_sequence_index) {
        return Status::InvalidArgument(
            "packed batch KV bindings are not unique");
      }
    }
    auto target = checked_add_u64(plan.committed_start_positions()[i],
                                  plan.real_token_counts()[i]);
    if (!target.ok() || *target > std::numeric_limits<std::uint32_t>::max()) {
      return Status::InvalidArgument("packed batch KV target overflowed");
    }
    auto append = binding.block_table->prepare_append(
        static_cast<std::uint32_t>(*target));
    if (!append.ok()) return append.status();
    append_plans_[i] = *append;
  }

  const auto active_appends =
      std::span(append_plans_).first(sequence_count);
  auto kv_metadata = kv_metadata_arena.materialize(
      plan, metadata, bindings, active_appends);
  if (!kv_metadata.ok()) return kv_metadata.status();
  auto execution = backend.execute_packed(plan, metadata, *kv_metadata,
                                          bindings, active_appends, sampling);
  if (!execution.ok()) {
    state_ = QwenBf16PackedBatchTransactionState::kPoisoned;
    return execution.status();
  }
  if (execution->sampled_token_ids.size() !=
          metadata.sample_row_index.size() ||
      execution->completion_event.handle == 0 ||
      execution->completion_event.generation == 0) {
    state_ = QwenBf16PackedBatchTransactionState::kPoisoned;
    return Status::Internal("packed batch backend result shape is invalid");
  }
  for (const auto token : execution->sampled_token_ids) {
    if (token >= 151936) {
      state_ = QwenBf16PackedBatchTransactionState::kPoisoned;
      return Status::Internal("packed batch backend token is invalid");
    }
  }
  for (std::size_t i = 0; i < sequence_count; ++i) {
    const auto validation = bindings[i].block_table->rollback_append(
        append_plans_[i]);
    if (!validation.ok()) {
      state_ = QwenBf16PackedBatchTransactionState::kPoisoned;
      return Status::Internal("packed batch KV precommit validation failed");
    }
  }
  for (std::size_t i = 0; i < sequence_count; ++i) {
    const auto committed =
        bindings[i].block_table->commit_append(append_plans_[i]);
    if (!committed.ok()) {
      state_ = QwenBf16PackedBatchTransactionState::kPoisoned;
      return Status::Internal("packed batch KV commit became partial");
    }
  }
  return *execution;
}

}  // namespace pih
