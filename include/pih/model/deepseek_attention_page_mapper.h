#pragma once

#include <cstdint>

#include "pih/core/result.h"
#include "pih/model/deepseek_attention_pool_geometry.h"

namespace pih {

struct DeepSeekCompressedPageLocation final {
  std::uint64_t slot_ordinal = 0;
  std::uint64_t page_ordinal = 0;
  std::uint32_t slot_in_page = 0;
};

struct DeepSeekCompressedCoverage final {
  std::uint64_t source_tokens = 0;
  std::uint64_t stored_slots = 0;
  std::uint64_t physical_pages = 0;
  std::uint32_t compressor_remainder_tokens = 0;
};

class DeepSeekAttentionPageMapper final {
 public:
  static Result<DeepSeekCompressedCoverage> Coverage(
      std::uint64_t source_tokens,
      const DeepSeekStatePoolGeometry& geometry);
  static Result<DeepSeekCompressedPageLocation> LocateStoredSlot(
      std::uint64_t slot_ordinal,
      const DeepSeekStatePoolGeometry& geometry);
};

}  // namespace pih
