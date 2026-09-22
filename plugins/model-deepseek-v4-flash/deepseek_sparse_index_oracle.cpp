#include "pih/model/deepseek_sparse_index_oracle.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace pih {

namespace {

bool offset_fits(std::int32_t offset, std::uint32_t maximum_ordinal) {
  return offset >= 0 &&
         static_cast<std::uint64_t>(offset) + maximum_ordinal <=
             static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
}

}  // namespace

Result<std::vector<std::int32_t>> DeepSeekSparseIndexOracle::RecentWindow(
    std::uint32_t query_position, std::int32_t physical_offset) {
  if (query_position >= kMaximumContextTokens ||
      !offset_fits(physical_offset, kRecentWindow - 1)) {
    return Status::InvalidArgument("DeepSeek recent-window offset is invalid");
  }
  const auto valid = std::min<std::uint32_t>(query_position + 1U,
                                             kRecentWindow);
  const auto first_logical = query_position + 1U - valid;
  std::vector<std::int32_t> indices(kRecentWindow, -1);
  for (std::uint32_t ordinal = 0; ordinal < valid; ++ordinal) {
    indices[ordinal] = physical_offset + static_cast<std::int32_t>(
                                             (first_logical + ordinal) %
                                             kRecentWindow);
  }
  return indices;
}

Result<std::vector<std::int32_t>> DeepSeekSparseIndexOracle::Ratio128(
    std::uint32_t query_position, std::int32_t physical_offset) {
  if (query_position >= kMaximumContextTokens) {
    return Status::InvalidArgument("DeepSeek ratio-128 position is invalid");
  }
  const auto count = (static_cast<std::uint64_t>(query_position) + 1U) / 128U;
  if (count > kRatio128MaximumSlots ||
      !offset_fits(physical_offset,
                   count == 0 ? 0U : static_cast<std::uint32_t>(count - 1U))) {
    return Status::InvalidArgument("DeepSeek ratio-128 index range is invalid");
  }
  std::vector<std::int32_t> indices(static_cast<std::size_t>(count));
  std::iota(indices.begin(), indices.end(), physical_offset);
  return indices;
}

Result<std::vector<std::int32_t>> DeepSeekSparseIndexOracle::Ratio4TopK(
    std::span<const float> selection_scores,
    std::uint32_t visible_slot_count, std::int32_t physical_offset) {
  if (visible_slot_count > selection_scores.size() ||
      !offset_fits(physical_offset,
                   visible_slot_count == 0 ? 0U : visible_slot_count - 1U)) {
    return Status::InvalidArgument("DeepSeek ratio-4 index range is invalid");
  }
  for (std::uint32_t index = 0; index < visible_slot_count; ++index) {
    if (!std::isfinite(selection_scores[index])) {
      return Status::InvalidArgument(
          "DeepSeek ratio-4 selection score is nonfinite");
    }
  }
  std::vector<std::uint32_t> order(visible_slot_count);
  std::iota(order.begin(), order.end(), 0U);
  std::stable_sort(order.begin(), order.end(), [&](const auto left,
                                                   const auto right) {
    return selection_scores[left] > selection_scores[right];
  });
  const auto selected = std::min<std::uint32_t>(visible_slot_count,
                                                 kRatio4TopK);
  if (visible_slot_count > kRatio4TopK &&
      selection_scores[order[kRatio4TopK - 1U]] ==
          selection_scores[order[kRatio4TopK]]) {
    return Status::InvalidArgument(
        "DeepSeek ratio-4 top-k boundary has no portable tie order");
  }
  std::vector<std::int32_t> indices(selected);
  for (std::uint32_t ordinal = 0; ordinal < selected; ++ordinal) {
    indices[ordinal] =
        physical_offset + static_cast<std::int32_t>(order[ordinal]);
  }
  return indices;
}

}  // namespace pih
