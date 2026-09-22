#pragma once
#include "config.h"
#include <span>
#include <vector>

namespace pih::deepseek_v41 {
struct EngramLayerBuckets final {
  std::array<std::uint32_t, 24> primes{};
  std::array<std::uint32_t, 24> offsets{};
  std::uint32_t rows = 0;
};
struct EngramTokenHashes final {
  // Layers 1 and 14; columns are n-grams 2, 3, 4, each with eight heads.
  std::array<std::array<std::uint32_t, 24>, 2> ids{};
};
class EngramHashState final {
 public:
  // Map bytes are exactly 129280 little-endian U32 entries. The expected digest
  // must be independently admitted; hashing cannot prove tokenizer normalization
  // equivalence. Includes dense first-appearance compressed-ID validation.
  static Result<EngramHashState> Create(const FlashConfig& config,
      std::span<const std::byte> compressed_token_map, const Sha256Digest& expected_map_sha256);
  // One sequence, contiguous append only; use one state per batch member.
  // Optional 0/1 participation mask: 0 is a DEAD/image barrier. Up to 4096 tokens
  // per append, 1048576 total. Invalid requests do not change sequence state.
  Result<std::vector<EngramTokenHashes>> Append(std::uint64_t start_position,
      std::span<const std::uint32_t> token_ids, std::span<const std::uint8_t> participation = {});
  void Reset() noexcept;
  std::uint64_t position() const noexcept { return position_; }
  const std::array<EngramLayerBuckets, 2>& buckets() const noexcept { return buckets_; }
 private:
  EngramHashState() = default;
  std::vector<std::uint32_t> token_map_;
  std::array<EngramLayerBuckets, 2> buckets_{};
  std::array<std::uint32_t, 3> history_{UINT32_MAX, UINT32_MAX, UINT32_MAX};
  std::uint32_t pad_id_ = 0;
  std::uint64_t position_ = 0;
};
}  // namespace pih::deepseek_v41
