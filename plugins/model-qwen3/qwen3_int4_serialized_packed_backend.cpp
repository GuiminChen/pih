#include "pih/model/qwen3_int4_serialized_packed_backend.h"

#include <algorithm>

#include "pih/model/qwen3_bf16_packed_kv_metadata.h"

namespace pih {

Result<QwenInt4SerializedPackedBackend>
QwenInt4SerializedPackedBackend::Create(
    QwenBf16SequenceBackend& sequence_backend,
    QwenBf16CompletionIdentityProvider& completion_provider,
    std::uint32_t maximum_sequences) {
  if (maximum_sequences == 0 || maximum_sequences > 4096) {
    return Status::InvalidArgument(
        "INT4 serialized packed sequence bound is invalid");
  }
  return QwenInt4SerializedPackedBackend(
      sequence_backend, completion_provider, maximum_sequences);
}

Result<QwenBf16PackedBatchExecutionView>
QwenInt4SerializedPackedBackend::execute_packed(
    const PackedTokenPlan& plan, PackedTokenMetadataView metadata,
    QwenBf16PackedKvMetadataView,
    std::span<const QwenBf16PackedKvBinding> bindings,
    std::span<const QwenKvAppendPlan> append_plans,
    std::span<const Qwen3SamplingDescriptor> sampling) {
  if (state_ != QwenInt4SerializedPackedBackendState::kReady) {
    return Status::FailedPrecondition(
        "INT4 serialized packed backend is not ready");
  }
  const auto count = plan.sequence_count();
  if (count == 0 || bindings.size() != count ||
      append_plans.size() != count || sampling.size() != count ||
      metadata.query_start_offsets.size() != count + 1 ||
      metadata.sample_row_index.size() > count) {
    return Status::InvalidArgument(
        "INT4 serialized packed sequence inputs drifted");
  }
  for (const auto& descriptor : sampling) {
    if (descriptor.mode != Qwen3SamplingMode::kGreedy ||
        descriptor.temperature != 0.0F || descriptor.top_p != 1.0F ||
        descriptor.top_k.has_value() || descriptor.top_logprobs_count != 0 ||
        descriptor.suppressed_token_count != 0) {
      return Status::FailedPrecondition(
          "INT4 serialized packed backend supports greedy sampling only");
    }
  }

  state_ = QwenInt4SerializedPackedBackendState::kRunning;
  const auto fail = [this](Status status)
      -> Result<QwenBf16PackedBatchExecutionView> {
    state_ = QwenInt4SerializedPackedBackendState::kPoisoned;
    return status;
  };
  sampled_tokens_.clear();
  sampling_receipts_.clear();
  std::size_t sample_index = 0;
  for (std::size_t sequence = 0; sequence < count; ++sequence) {
    const auto begin = plan.packed_offsets()[sequence];
    const auto end = plan.packed_offsets()[sequence + 1];
    if (begin >= end || end > metadata.input_token_ids.size() ||
        bindings[sequence].block_table == nullptr) {
      return fail(Status::InvalidArgument(
          "INT4 serialized packed token slice is invalid"));
    }
    std::vector<std::int64_t> tokens;
    tokens.reserve(end - begin);
    for (auto packed = begin; packed < end; ++packed) {
      tokens.push_back(metadata.input_token_ids[packed]);
    }
    auto token = sequence_backend_->execute_and_read_token(
        tokens, plan.committed_start_positions()[sequence],
        *bindings[sequence].block_table, append_plans[sequence]);
    if (!token.ok()) return fail(token.status());
    const bool produces_token =
        sample_index < metadata.sample_row_index.size() &&
        metadata.sample_row_index[sample_index] == end - 1;
    if (produces_token) {
      if (*token < 0 || *token >= QwenBf16PackedResultLayout::kVocabularySize) {
        return fail(Status::Internal(
            "INT4 serialized packed backend produced an invalid token"));
      }
      sampled_tokens_.push_back(static_cast<std::uint32_t>(*token));
      QwenBf16PackedSampleReceipt receipt{};
      receipt.token_id = static_cast<std::uint32_t>(*token);
      sampling_receipts_.push_back(receipt);
      ++sample_index;
    }
  }
  if (sample_index != metadata.sample_row_index.size()) {
    return fail(Status::InvalidArgument(
        "INT4 serialized packed sample rows drifted"));
  }
  auto completion = completion_provider_->last_completion_event();
  if (!completion.ok()) return fail(completion.status());
  state_ = QwenInt4SerializedPackedBackendState::kReady;
  return QwenBf16PackedBatchExecutionView{
      sampled_tokens_, *completion, sampling_receipts_};
}

}  // namespace pih
