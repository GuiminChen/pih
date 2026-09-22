#include "pih/model/deepseek_chunked_prefill_plan.h"

#include <algorithm>
#include <set>

#include "pih/core/canonical_hash.h"

namespace pih {

Result<DeepSeekChunkedPrefillPlan> DeepSeekChunkedPrefillPlan::Compile(
    std::uint64_t plan_sequence, std::uint32_t maximum_chunk_tokens,
    std::span<const DeepSeekChunkedPrefillSequence> sequences) {
  if (plan_sequence == 0 || maximum_chunk_tokens == 0 || sequences.empty() ||
      sequences.size() > 4096) {
    return Status::InvalidArgument(
        "DeepSeek chunked prefill plan input is invalid");
  }
  std::set<std::uint64_t> request_ids;
  for (const auto& sequence : sequences) {
    if (sequence.request_id == 0 || sequence.request_generation == 0 ||
        sequence.prompt_token_count == 0 ||
        sequence.consumed_token_count > sequence.prompt_token_count) {
      return Status::InvalidArgument(
          "DeepSeek chunked prefill sequence is invalid");
    }
    if (!request_ids.insert(sequence.request_id).second) {
      return Status::InvalidArgument(
          "DeepSeek chunked prefill request ID is duplicate");
    }
  }
  DeepSeekChunkedPrefillPlan result;
  result.plan_sequence_ = plan_sequence;
  std::uint32_t budget = maximum_chunk_tokens;
  std::vector<std::uint32_t> projected;
  projected.reserve(sequences.size());
  for (std::uint32_t index = 0; index < sequences.size(); ++index) {
    const auto& sequence = sequences[index];
    projected.push_back(sequence.consumed_token_count);
    const auto remaining =
        sequence.prompt_token_count - sequence.consumed_token_count;
    const auto count = std::min(remaining, budget);
    if (count != 0) {
      result.slices_.push_back({index, sequence.request_id,
                                sequence.request_generation,
                                sequence.consumed_token_count, count});
      projected.back() += count;
      result.token_count_ += count;
      budget -= count;
    }
    if (budget == 0) {
      for (++index; index < sequences.size(); ++index) {
        projected.push_back(sequences[index].consumed_token_count);
      }
      break;
    }
  }
  if (result.slices_.empty()) {
    return Status::FailedPrecondition(
        "DeepSeek chunked prefill is already complete");
  }
  result.publishes_sampled_token_ = true;
  for (std::uint32_t index = 0; index < sequences.size(); ++index) {
    result.publishes_sampled_token_ =
        result.publishes_sampled_token_ &&
        projected[index] == sequences[index].prompt_token_count;
  }
  const auto field_count =
      static_cast<std::uint32_t>(4 + sequences.size() * 4);
  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek-chunked-prefill-plan:v1", field_count);
  if (!hash.ok()) return hash.status();
  Status status = hash->add_u64(1, plan_sequence);
  if (status.ok()) status = hash->add_u32(2, maximum_chunk_tokens);
  if (status.ok()) status = hash->add_u32(3, result.token_count_);
  if (status.ok()) status = hash->add_u32(4, result.publishes_sampled_token_);
  std::uint16_t field = 5;
  for (const auto& sequence : sequences) {
    if (status.ok()) status = hash->add_u64(field++, sequence.request_id);
    if (status.ok()) {
      status = hash->add_u64(field++, sequence.request_generation);
    }
    if (status.ok()) {
      status = hash->add_u32(field++, sequence.prompt_token_count);
    }
    if (status.ok()) {
      status = hash->add_u32(field++, sequence.consumed_token_count);
    }
  }
  if (!status.ok()) return status;
  auto identity = hash->finalize();
  if (!identity.ok()) return identity.status();
  result.identity_ = *identity;
  return result;
}

}  // namespace pih
