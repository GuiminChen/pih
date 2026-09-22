#include "pih/model/packed_token_plan.h"

#include <array>
#include <limits>
#include <span>
#include <string_view>
#include <unordered_set>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Status update_u64(Sha256& digest, std::uint64_t value) {
  std::array<std::byte, 8> wire{};
  for (std::size_t index = 0; index < wire.size(); ++index) {
    wire[index] = static_cast<std::byte>(value >> (index * 8U));
  }
  return digest.update(wire);
}

Status update_bytes(Sha256& digest, std::string_view value) {
  Status status = update_u64(digest, value.size());
  if (!status.ok()) return status;
  return digest.update(std::as_bytes(std::span(value)));
}

bool valid_phase(PackedTokenPhase phase) {
  return phase == PackedTokenPhase::kPrefill ||
         phase == PackedTokenPhase::kDecode ||
         phase == PackedTokenPhase::kVerify;
}

bool nonzero(const Sha256Digest& digest) {
  for (const std::byte value : digest.bytes) {
    if (value != std::byte{0}) return true;
  }
  return false;
}

}  // namespace

Result<PackedTokenPlan> PackedTokenPlan::Create(
    std::uint64_t epoch, std::uint64_t plan_sequence,
    PackedTokenPhase phase, std::string profile_revision,
    Sha256Digest resource_vector_hash,
    std::span<const PackedSequenceInput> sequences,
    std::uint32_t execution_bucket_tokens,
    PackedTokenPlanLimits limits) {
  if (epoch == 0 || plan_sequence == 0 || !valid_phase(phase) ||
      profile_revision.empty() || profile_revision.size() > 128 ||
      !nonzero(resource_vector_hash) ||
      sequences.empty() || limits.maximum_sequences == 0 ||
      limits.maximum_real_tokens == 0 ||
      limits.maximum_execution_bucket_tokens == 0 ||
      sequences.size() > limits.maximum_sequences ||
      execution_bucket_tokens == 0 ||
      execution_bucket_tokens > limits.maximum_execution_bucket_tokens) {
    return Status::InvalidArgument("packed token plan identity is invalid");
  }

  std::vector<std::uint64_t> generations;
  std::vector<std::uint64_t> starts;
  std::vector<std::uint32_t> counts;
  std::vector<std::uint32_t> offsets;
  std::vector<std::uint64_t> state_generations;
  std::vector<Sha256Digest> input_digests;
  generations.reserve(sequences.size());
  starts.reserve(sequences.size());
  counts.reserve(sequences.size());
  offsets.reserve(sequences.size() + 1);
  state_generations.reserve(sequences.size());
  input_digests.reserve(sequences.size());
  offsets.push_back(0);
  std::unordered_set<std::uint64_t> unique;
  unique.reserve(sequences.size());
  std::uint64_t total = 0;

  for (const auto& sequence : sequences) {
    if (sequence.sequence_generation == 0 || sequence.state_generation == 0 ||
        !nonzero(sequence.input_digest) ||
        sequence.real_token_count == 0 ||
        !unique.insert(sequence.sequence_generation).second ||
        (phase == PackedTokenPhase::kDecode &&
         sequence.real_token_count != 1) ||
        (phase == PackedTokenPhase::kVerify &&
         sequence.real_token_count > 5)) {
      return Status::InvalidArgument("packed token sequence is invalid");
    }
    auto next = checked_add_u64(total, sequence.real_token_count);
    if (!next.ok() || *next > limits.maximum_real_tokens ||
        *next > std::numeric_limits<std::uint32_t>::max()) {
      return Status::ResourceExhausted("packed real token prefix exceeds its bound");
    }
    total = *next;
    generations.push_back(sequence.sequence_generation);
    starts.push_back(sequence.committed_start_position);
    counts.push_back(sequence.real_token_count);
    offsets.push_back(static_cast<std::uint32_t>(total));
    state_generations.push_back(sequence.state_generation);
    input_digests.push_back(sequence.input_digest);
  }
  if (execution_bucket_tokens < total) {
    return Status::InvalidArgument("execution bucket is smaller than real tokens");
  }
  return PackedTokenPlan(
      epoch, plan_sequence, phase, std::move(profile_revision),
      resource_vector_hash, std::move(generations), std::move(starts),
      std::move(counts), std::move(offsets), std::move(state_generations),
      std::move(input_digests), static_cast<std::uint32_t>(total),
      execution_bucket_tokens);
}

Result<Sha256Digest> PackedTokenPlan::semantic_digest() const {
  Sha256 digest;
  Status status = update_bytes(digest, kAbi);
  for (const std::uint64_t value : {
           epoch_, plan_sequence_, static_cast<std::uint64_t>(phase_),
           static_cast<std::uint64_t>(total_real_tokens_),
           static_cast<std::uint64_t>(execution_bucket_tokens_),
           static_cast<std::uint64_t>(sequence_count())}) {
    if (status.ok()) status = update_u64(digest, value);
  }
  if (status.ok()) status = update_bytes(digest, profile_revision_);
  if (status.ok()) status = digest.update(resource_vector_hash_.bytes);
  for (std::size_t index = 0; index < sequence_count() && status.ok(); ++index) {
    status = update_u64(digest, sequence_generations_[index]);
    if (status.ok()) status = update_u64(digest, committed_start_positions_[index]);
    if (status.ok()) status = update_u64(digest, real_token_counts_[index]);
    if (status.ok()) status = update_u64(digest, packed_offsets_[index]);
    if (status.ok()) status = update_u64(digest, state_generations_[index]);
    if (status.ok()) status = digest.update(input_digests_[index].bytes);
  }
  if (status.ok()) status = update_u64(digest, packed_offsets_.back());
  if (!status.ok()) return status;
  return digest.finalize();
}

}  // namespace pih
