#pragma once

#include <cstdint>

#include "pih/core/result.h"
#include "pih/model/deepseek_attention_pool_geometry.h"

namespace pih {

struct DeepSeekCompressionAppendPlan final {
  std::uint64_t source_begin = 0;
  std::uint64_t source_end = 0;
  std::uint64_t first_new_slot = 0;
  std::uint64_t new_slot_count = 0;
  std::uint64_t first_touched_page = 0;
  std::uint64_t touched_page_count = 0;
  std::uint32_t remainder_before = 0;
  std::uint32_t remainder_after = 0;
};

class DeepSeekCompressionAppendPlanner final {
 public:
  static Result<DeepSeekCompressionAppendPlan> Plan(
      std::uint64_t committed_source_tokens,
      std::uint32_t appended_source_tokens,
      const DeepSeekStatePoolGeometry& geometry);
};

}  // namespace pih
