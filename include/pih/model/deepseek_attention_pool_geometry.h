#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/core/result.h"

namespace pih {

enum class DeepSeekStatePoolKind : std::uint16_t {
  kRecentBf16 = 1,
  kRatio4MainBf16 = 2,
  kRatio4IndexBf16 = 3,
  kRatio128MainBf16 = 4,
};

struct DeepSeekStatePoolGeometry final {
  static constexpr std::size_t kWireBytes = 48;
  static constexpr std::uint16_t kSwaOnlyLayer = 1U << 0U;
  static constexpr std::uint16_t kC4aLayer = 1U << 1U;
  static constexpr std::uint16_t kC128aLayer = 1U << 2U;
  static constexpr std::uint16_t kDsparkLayer = 1U << 3U;

  DeepSeekStatePoolKind pool_kind = DeepSeekStatePoolKind::kRecentBf16;
  std::uint16_t layer_type_mask = 0;
  std::uint32_t logical_ratio = 0;
  std::uint32_t logical_coverage_tokens = 0;
  std::uint32_t storage_units_per_page = 0;
  std::uint32_t bytes_per_storage_unit = 0;
  std::uint64_t raw_page_bytes = 0;
  std::uint32_t page_alignment_bytes = 0;
  std::uint64_t physical_page_bytes = 0;
  std::uint16_t handle_bytes = 0;
  std::uint16_t handles_per_logical_page = 0;
  std::uint16_t paired_commit_group = 0;
  std::uint16_t reserved = 0;

  Status validate() const;
  Result<std::array<std::byte, kWireBytes>> encode() const;
  static Result<DeepSeekStatePoolGeometry> Decode(
      std::span<const std::byte> wire);
  static std::array<DeepSeekStatePoolGeometry, 4> CanonicalBf16();
};

}  // namespace pih
