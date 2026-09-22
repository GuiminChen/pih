#include "pih/model/packed_token_metadata_arena.h"

#include <array>
#include <cstddef>

#include "pih/core/checked_math.h"

namespace pih {
namespace {
constexpr std::string_view kDigestDomain = "pih-packed-u32-tokens-v1";

Status update_u32(Sha256& digest, std::uint32_t value) {
  const std::array wire{static_cast<std::byte>(value),
                        static_cast<std::byte>(value >> 8U),
                        static_cast<std::byte>(value >> 16U),
                        static_cast<std::byte>(value >> 24U)};
  return digest.update(wire);
}
}  // namespace

Result<Sha256Digest> packed_token_input_digest(
    std::span<const std::uint32_t> token_ids) {
  if (token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted("packed token digest input is too large");
  }
  Sha256 digest;
  Status status = digest.update(std::as_bytes(std::span(kDigestDomain)));
  if (status.ok()) {
    status = update_u32(digest, static_cast<std::uint32_t>(token_ids.size()));
  }
  for (const auto token : token_ids) {
    if (status.ok()) status = update_u32(digest, token);
  }
  if (!status.ok()) return status;
  return digest.finalize();
}

PackedTokenMetadataArena::PackedTokenMetadataArena(
    std::uint32_t maximum_sequences, std::uint32_t maximum_tokens)
    : maximum_sequences_(maximum_sequences),
      maximum_tokens_(maximum_tokens),
      token_ids_(maximum_tokens),
      positions_(maximum_tokens),
      request_indices_(maximum_tokens, kInvalidPackedRequestIndex),
      query_offsets_(static_cast<std::size_t>(maximum_sequences) + 1),
      sample_rows_(maximum_sequences) {}

Result<PackedTokenMetadataArena> PackedTokenMetadataArena::Create(
    PackedTokenMetadataLimits limits) {
  if (limits.maximum_sequences == 0 ||
      limits.maximum_execution_bucket_tokens == 0) {
    return Status::InvalidArgument("packed metadata limits must be nonzero");
  }
  return PackedTokenMetadataArena(limits.maximum_sequences,
                                  limits.maximum_execution_bucket_tokens);
}

Result<PackedTokenMetadataView> PackedTokenMetadataArena::materialize(
    const PackedTokenPlan& plan,
    std::span<const PackedSequenceTokens> sequences) {
  if (sequences.size() != plan.sequence_count() ||
      sequences.size() > maximum_sequences_ ||
      plan.execution_bucket_tokens() > maximum_tokens_ ||
      generation_ == std::numeric_limits<std::uint64_t>::max()) {
    return Status::ResourceExhausted("packed metadata arena cannot represent plan");
  }

  // Validate the complete payload before publishing any bytes.
  for (std::size_t i = 0; i < sequences.size(); ++i) {
    if (sequences[i].sequence_generation !=
            plan.ordered_sequence_generations()[i] ||
        sequences[i].token_ids.size() != plan.real_token_counts()[i]) {
      return Status::InvalidArgument("packed token payload does not match plan");
    }
    auto digest = packed_token_input_digest(sequences[i].token_ids);
    if (!digest.ok()) return digest.status();
    if (*digest != plan.input_digests()[i]) {
      return Status::InvalidArgument("packed token payload digest mismatch");
    }
    if (!sequences[i].token_ids.empty()) {
      auto end = checked_add_u64(plan.committed_start_positions()[i],
                                 sequences[i].token_ids.size() - 1);
      if (!end.ok()) return end.status();
    }
  }

  std::uint32_t sample_count = 0;
  for (std::size_t i = 0; i < sequences.size(); ++i) {
    const auto begin = plan.packed_offsets()[i];
    query_offsets_[i] = begin;
    for (std::size_t local = 0; local < sequences[i].token_ids.size(); ++local) {
      const auto packed = begin + static_cast<std::uint32_t>(local);
      token_ids_[packed] = sequences[i].token_ids[local];
      positions_[packed] = plan.committed_start_positions()[i] + local;
      request_indices_[packed] = static_cast<std::uint32_t>(i);
    }
    if (sequences[i].produces_logits) {
      sample_rows_[sample_count++] = plan.packed_offsets()[i + 1] - 1;
    }
  }
  query_offsets_[sequences.size()] = plan.total_real_tokens();
  for (std::uint32_t i = plan.total_real_tokens();
       i < plan.execution_bucket_tokens(); ++i) {
    token_ids_[i] = 0;
    positions_[i] = 0;
    request_indices_[i] = kInvalidPackedRequestIndex;
  }
  ++generation_;
  return PackedTokenMetadataView{
      generation_,
      std::span(token_ids_).first(plan.execution_bucket_tokens()),
      std::span(positions_).first(plan.execution_bucket_tokens()),
      std::span(request_indices_).first(plan.execution_bucket_tokens()),
      std::span(query_offsets_).first(sequences.size() + 1),
      std::span(sample_rows_).first(sample_count), plan.total_real_tokens()};
}

}  // namespace pih
