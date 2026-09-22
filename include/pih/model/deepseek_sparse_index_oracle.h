#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

class DeepSeekSparseIndexOracle final {
 public:
  static constexpr std::uint32_t kRecentWindow = 128;
  static constexpr std::uint32_t kRatio4TopK = 512;
  static constexpr std::uint32_t kRatio128MaximumSlots = 8192;
  static constexpr std::uint32_t kMaximumContextTokens = 1048576;

  // Returns physical circular-buffer slots in chronological order, followed by
  // -1 padding until the fixed 128-entry recent-window width is reached.
  static Result<std::vector<std::int32_t>> RecentWindow(
      std::uint32_t query_position, std::int32_t physical_offset = 0);

  // A ratio-128 slot is visible only after all 128 source tokens, including
  // the current query position, have been consumed.
  static Result<std::vector<std::int32_t>> Ratio128(
      std::uint32_t query_position, std::int32_t physical_offset = 0);

  // selection_scores contains the already reduced ReLU/index-head score for
  // every compressed ratio-4 slot. Only visible_slot_count candidates may be
  // selected. Exact ties at the k/k+1 boundary are rejected because upstream
  // top-k does not define a portable tie order.
  static Result<std::vector<std::int32_t>> Ratio4TopK(
      std::span<const float> selection_scores,
      std::uint32_t visible_slot_count,
      std::int32_t physical_offset = 0);
};

}  // namespace pih
