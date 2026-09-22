#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/sha256.h"
#include "pih/model/deepseek_fixed_state_layout.h"
#include "pih/model/deepseek_v4_config.h"

namespace pih {

enum class DeepSeekAttentionPhysicalExtentKind : std::uint8_t {
  kRatio4Main = 0,
  kRatio4Index = 1,
  kRatio128Main = 2,
};

struct DeepSeekAttentionPhysicalExtent final {
  DeepSeekAttentionPhysicalExtentKind kind =
      DeepSeekAttentionPhysicalExtentKind::kRatio4Main;
  std::uint64_t offset_bytes = 0;
  std::uint64_t bytes = 0;
  std::uint64_t alignment_bytes = 0;
  std::uint64_t slot_count = 0;
  std::uint64_t page_bytes = 0;
};

class DeepSeekAttentionPhysicalLayout final {
 public:
  static Result<DeepSeekAttentionPhysicalLayout> Compile(
      DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
      std::uint32_t reserved_tokens_per_sequence);

  [[nodiscard]] std::uint32_t version() const noexcept { return 1; }
  [[nodiscard]] std::span<const DeepSeekAttentionPhysicalExtent> extents()
      const noexcept { return extents_; }
  [[nodiscard]] const DeepSeekAttentionPhysicalExtent& extent(
      DeepSeekAttentionPhysicalExtentKind kind) const noexcept {
    return extents_[static_cast<std::size_t>(kind)];
  }
  [[nodiscard]] std::uint64_t allocation_bytes() const noexcept {
    return allocation_bytes_;
  }
  [[nodiscard]] std::uint32_t ratio4_page_pairs_per_sequence() const noexcept {
    return ratio4_page_pairs_per_sequence_;
  }
  [[nodiscard]] std::uint32_t ratio128_pages_per_sequence() const noexcept {
    return ratio128_pages_per_sequence_;
  }
  [[nodiscard]] std::uint32_t ratio4_pair_count() const noexcept {
    return ratio4_pair_count_;
  }
  [[nodiscard]] std::uint32_t ratio128_page_count() const noexcept {
    return ratio128_page_count_;
  }
  [[nodiscard]] const DeepSeekFixedStateLayout& fixed_layout() const noexcept {
    return fixed_layout_;
  }
  [[nodiscard]] const Sha256Digest& identity() const noexcept {
    return identity_;
  }

 private:
  DeepSeekFixedStateLayout fixed_layout_;
  std::vector<DeepSeekAttentionPhysicalExtent> extents_;
  std::uint64_t allocation_bytes_ = 0;
  std::uint32_t ratio4_page_pairs_per_sequence_ = 0;
  std::uint32_t ratio128_pages_per_sequence_ = 0;
  std::uint32_t ratio4_pair_count_ = 0;
  std::uint32_t ratio128_page_count_ = 0;
  Sha256Digest identity_{};
};

}  // namespace pih
