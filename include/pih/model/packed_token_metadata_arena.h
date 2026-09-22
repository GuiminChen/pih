#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"
#include "pih/model/packed_token_plan.h"

namespace pih {

inline constexpr std::uint32_t kInvalidPackedRequestIndex =
    std::numeric_limits<std::uint32_t>::max();

struct PackedSequenceTokens final {
  std::uint64_t sequence_generation;
  std::span<const std::uint32_t> token_ids;
  bool produces_logits;
};

struct PackedTokenMetadataLimits final {
  std::uint32_t maximum_sequences;
  std::uint32_t maximum_execution_bucket_tokens;
};

struct PackedTokenMetadataView final {
  std::uint64_t generation;
  std::span<const std::uint32_t> input_token_ids;
  std::span<const std::uint64_t> positions;
  std::span<const std::uint32_t> request_index;
  std::span<const std::uint32_t> query_start_offsets;
  std::span<const std::uint32_t> sample_row_index;
  std::uint32_t real_token_count;
};

Result<Sha256Digest> packed_token_input_digest(
    std::span<const std::uint32_t> token_ids);

class PackedTokenMetadataArena final {
 public:
  static Result<PackedTokenMetadataArena> Create(
      PackedTokenMetadataLimits limits);

  Result<PackedTokenMetadataView> materialize(
      const PackedTokenPlan& plan,
      std::span<const PackedSequenceTokens> sequences);

 private:
  PackedTokenMetadataArena(std::uint32_t maximum_sequences,
                           std::uint32_t maximum_tokens);

  std::uint32_t maximum_sequences_ = 0;
  std::uint32_t maximum_tokens_ = 0;
  std::uint64_t generation_ = 0;
  std::vector<std::uint32_t> token_ids_;
  std::vector<std::uint64_t> positions_;
  std::vector<std::uint32_t> request_indices_;
  std::vector<std::uint32_t> query_offsets_;
  std::vector<std::uint32_t> sample_rows_;
};

}  // namespace pih
