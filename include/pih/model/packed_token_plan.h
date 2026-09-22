#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

enum class PackedTokenPhase : std::uint8_t {
  kPrefill = 1,
  kDecode = 2,
  kVerify = 3,
};

struct PackedSequenceInput final {
  std::uint64_t sequence_generation;
  std::uint64_t committed_start_position;
  std::uint32_t real_token_count;
  std::uint64_t state_generation;
  Sha256Digest input_digest;
};

struct PackedTokenPlanLimits final {
  std::uint32_t maximum_sequences;
  std::uint32_t maximum_real_tokens;
  std::uint32_t maximum_execution_bucket_tokens;
};

class PackedTokenPlan final {
 public:
  static constexpr std::string_view kAbi =
      "homogeneous_phase_real_tokens_v1";

  static Result<PackedTokenPlan> Create(
      std::uint64_t epoch, std::uint64_t plan_sequence,
      PackedTokenPhase phase, std::string profile_revision,
      Sha256Digest resource_vector_hash,
      std::span<const PackedSequenceInput> sequences,
      std::uint32_t execution_bucket_tokens,
      PackedTokenPlanLimits limits);

  [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }
  [[nodiscard]] std::uint64_t plan_sequence() const noexcept {
    return plan_sequence_;
  }
  [[nodiscard]] PackedTokenPhase phase() const noexcept { return phase_; }
  [[nodiscard]] std::string_view profile_revision() const noexcept {
    return profile_revision_;
  }
  [[nodiscard]] std::size_t sequence_count() const noexcept {
    return sequence_generations_.size();
  }
  [[nodiscard]] std::uint32_t total_real_tokens() const noexcept {
    return total_real_tokens_;
  }
  [[nodiscard]] std::uint32_t execution_bucket_tokens() const noexcept {
    return execution_bucket_tokens_;
  }
  [[nodiscard]] const std::vector<std::uint64_t>&
  ordered_sequence_generations() const noexcept {
    return sequence_generations_;
  }
  [[nodiscard]] const std::vector<std::uint64_t>&
  committed_start_positions() const noexcept {
    return committed_start_positions_;
  }
  [[nodiscard]] const std::vector<std::uint32_t>& real_token_counts() const noexcept {
    return real_token_counts_;
  }
  [[nodiscard]] const std::vector<std::uint32_t>& packed_offsets() const noexcept {
    return packed_offsets_;
  }
  [[nodiscard]] const std::vector<std::uint64_t>& state_generations() const noexcept {
    return state_generations_;
  }
  [[nodiscard]] const std::vector<Sha256Digest>& input_digests() const noexcept {
    return input_digests_;
  }
  [[nodiscard]] const Sha256Digest& resource_vector_hash() const noexcept {
    return resource_vector_hash_;
  }
  [[nodiscard]] Result<Sha256Digest> semantic_digest() const;

 private:
  PackedTokenPlan(
      std::uint64_t epoch, std::uint64_t plan_sequence,
      PackedTokenPhase phase, std::string profile_revision,
      Sha256Digest resource_vector_hash,
      std::vector<std::uint64_t> sequence_generations,
      std::vector<std::uint64_t> committed_start_positions,
      std::vector<std::uint32_t> real_token_counts,
      std::vector<std::uint32_t> packed_offsets,
      std::vector<std::uint64_t> state_generations,
      std::vector<Sha256Digest> input_digests,
      std::uint32_t total_real_tokens,
      std::uint32_t execution_bucket_tokens)
      : epoch_(epoch),
        plan_sequence_(plan_sequence),
        phase_(phase),
        profile_revision_(std::move(profile_revision)),
        resource_vector_hash_(resource_vector_hash),
        sequence_generations_(std::move(sequence_generations)),
        committed_start_positions_(std::move(committed_start_positions)),
        real_token_counts_(std::move(real_token_counts)),
        packed_offsets_(std::move(packed_offsets)),
        state_generations_(std::move(state_generations)),
        input_digests_(std::move(input_digests)),
        total_real_tokens_(total_real_tokens),
        execution_bucket_tokens_(execution_bucket_tokens) {}

  std::uint64_t epoch_ = 0;
  std::uint64_t plan_sequence_ = 0;
  PackedTokenPhase phase_ = PackedTokenPhase::kPrefill;
  std::string profile_revision_;
  Sha256Digest resource_vector_hash_{};
  std::vector<std::uint64_t> sequence_generations_;
  std::vector<std::uint64_t> committed_start_positions_;
  std::vector<std::uint32_t> real_token_counts_;
  std::vector<std::uint32_t> packed_offsets_;
  std::vector<std::uint64_t> state_generations_;
  std::vector<Sha256Digest> input_digests_;
  std::uint32_t total_real_tokens_ = 0;
  std::uint32_t execution_bucket_tokens_ = 0;
};

}  // namespace pih
