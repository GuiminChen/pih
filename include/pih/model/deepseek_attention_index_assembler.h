#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

struct DeepSeekSparseIndexMatrix final {
  std::vector<std::int32_t> values;
  std::uint32_t query_count = 0;
  std::uint32_t row_width = 0;
};

class DeepSeekAttentionIndexAssembler final {
 public:
  static Result<DeepSeekSparseIndexMatrix> RecentOnly(
      std::span<const std::uint32_t> query_positions,
      std::int32_t recent_physical_offset);

  static Result<DeepSeekSparseIndexMatrix> Ratio4(
      std::span<const std::uint32_t> query_positions,
      std::span<const std::vector<std::uint32_t>> selected_compressed,
      std::uint32_t compressed_slot_count,
      std::int32_t recent_physical_offset,
      std::int32_t compressed_physical_offset);

  static Result<DeepSeekSparseIndexMatrix> Ratio128(
      std::span<const std::uint32_t> query_positions,
      std::uint32_t compressed_slot_count,
      std::int32_t recent_physical_offset,
      std::int32_t compressed_physical_offset);
};

}  // namespace pih
