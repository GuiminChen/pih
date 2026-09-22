#include "pih/model/deepseek_attention_index_assembler.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "pih/model/deepseek_sparse_index_oracle.h"
#include "pih/model/deepseek_sparse_attention_oracle.h"

namespace pih {
namespace {

Result<DeepSeekSparseIndexMatrix> create_matrix(std::uint32_t query_count,
                                                 std::uint32_t compressed_width) {
  const auto row_width =
      DeepSeekSparseIndexOracle::kRecentWindow + compressed_width;
  if (query_count == 0 || row_width >
                              DeepSeekSparseAttentionOracle::kMaximumIndices) {
    return Status::InvalidArgument("DeepSeek sparse index matrix is invalid");
  }
  DeepSeekSparseIndexMatrix matrix;
  matrix.query_count = query_count;
  matrix.row_width = row_width;
  matrix.values.assign(static_cast<std::size_t>(query_count) * row_width, -1);
  return matrix;
}

Status copy_recent(DeepSeekSparseIndexMatrix& matrix, std::uint32_t row,
                   std::uint32_t position, std::int32_t offset) {
  auto recent = DeepSeekSparseIndexOracle::RecentWindow(position, offset);
  if (!recent.ok()) return recent.status();
  std::copy(recent->begin(), recent->end(),
            matrix.values.begin() + static_cast<std::size_t>(row) *
                                        matrix.row_width);
  return Status::Ok();
}

bool compressed_offset_fits(std::int32_t offset, std::uint32_t slots) {
  return offset >= 0 &&
         static_cast<std::uint64_t>(offset) + (slots == 0 ? 0U : slots - 1U) <=
             static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
}

bool physical_ranges_disjoint(std::int32_t recent_offset,
                              std::int32_t compressed_offset,
                              std::uint32_t compressed_slots) {
  if (recent_offset < 0 || compressed_slots == 0) return recent_offset >= 0;
  const auto recent_begin = static_cast<std::uint64_t>(recent_offset);
  const auto recent_end = recent_begin +
                          DeepSeekSparseIndexOracle::kRecentWindow;
  const auto compressed_begin = static_cast<std::uint64_t>(compressed_offset);
  const auto compressed_end = compressed_begin + compressed_slots;
  return recent_end <= compressed_begin || compressed_end <= recent_begin;
}

}  // namespace

Result<DeepSeekSparseIndexMatrix> DeepSeekAttentionIndexAssembler::RecentOnly(
    std::span<const std::uint32_t> query_positions,
    std::int32_t recent_physical_offset) {
  auto matrix = create_matrix(static_cast<std::uint32_t>(query_positions.size()), 0);
  if (!matrix.ok()) return matrix.status();
  for (std::uint32_t row = 0; row < query_positions.size(); ++row) {
    auto status = copy_recent(*matrix, row, query_positions[row], recent_physical_offset);
    if (!status.ok()) return status;
  }
  return matrix;
}

Result<DeepSeekSparseIndexMatrix> DeepSeekAttentionIndexAssembler::Ratio4(
    std::span<const std::uint32_t> query_positions,
    std::span<const std::vector<std::uint32_t>> selected_compressed,
    std::uint32_t compressed_slot_count, std::int32_t recent_physical_offset,
    std::int32_t compressed_physical_offset) {
  if (query_positions.size() != selected_compressed.size() ||
      compressed_slot_count > 262144 ||
      !compressed_offset_fits(compressed_physical_offset,
                              compressed_slot_count) ||
      !physical_ranges_disjoint(recent_physical_offset,
                                compressed_physical_offset,
                                compressed_slot_count)) {
    return Status::InvalidArgument("DeepSeek ratio-4 index input is invalid");
  }
  const auto compressed_width = std::min(
      compressed_slot_count, DeepSeekSparseIndexOracle::kRatio4TopK);
  auto matrix = create_matrix(static_cast<std::uint32_t>(query_positions.size()),
                              compressed_width);
  if (!matrix.ok()) return matrix.status();
  for (std::uint32_t row = 0; row < query_positions.size(); ++row) {
    auto status = copy_recent(*matrix, row, query_positions[row],
                              recent_physical_offset);
    if (!status.ok()) return status;
    if (selected_compressed[row].size() > compressed_width) {
      return Status::InvalidArgument(
          "DeepSeek ratio-4 selection exceeds matrix width");
    }
    std::unordered_set<std::uint32_t> unique;
    const auto causally_visible = (query_positions[row] + 1U) / 4U;
    auto destination = matrix->values.begin() +
                       static_cast<std::size_t>(row) * matrix->row_width +
                       DeepSeekSparseIndexOracle::kRecentWindow;
    for (std::uint32_t ordinal = 0;
         ordinal < selected_compressed[row].size(); ++ordinal) {
      const auto logical = selected_compressed[row][ordinal];
      if (logical >= compressed_slot_count || logical >= causally_visible ||
          !unique.insert(logical).second) {
        return Status::InvalidArgument(
            "DeepSeek ratio-4 selection is invalid");
      }
      destination[ordinal] = compressed_physical_offset +
                             static_cast<std::int32_t>(logical);
    }
  }
  return matrix;
}

Result<DeepSeekSparseIndexMatrix> DeepSeekAttentionIndexAssembler::Ratio128(
    std::span<const std::uint32_t> query_positions,
    std::uint32_t compressed_slot_count, std::int32_t recent_physical_offset,
    std::int32_t compressed_physical_offset) {
  if (compressed_slot_count >
          DeepSeekSparseIndexOracle::kRatio128MaximumSlots ||
      !compressed_offset_fits(compressed_physical_offset,
                              compressed_slot_count) ||
      !physical_ranges_disjoint(recent_physical_offset,
                                compressed_physical_offset,
                                compressed_slot_count)) {
    return Status::InvalidArgument("DeepSeek ratio-128 index input is invalid");
  }
  auto matrix = create_matrix(static_cast<std::uint32_t>(query_positions.size()),
                              compressed_slot_count);
  if (!matrix.ok()) return matrix.status();
  for (std::uint32_t row = 0; row < query_positions.size(); ++row) {
    auto status = copy_recent(*matrix, row, query_positions[row],
                              recent_physical_offset);
    if (!status.ok()) return status;
    auto compressed = DeepSeekSparseIndexOracle::Ratio128(
        query_positions[row], compressed_physical_offset);
    if (!compressed.ok() || compressed->size() > compressed_slot_count) {
      return Status::InvalidArgument(
          "DeepSeek ratio-128 visibility exceeds available slots");
    }
    std::copy(compressed->begin(), compressed->end(),
              matrix->values.begin() +
                  static_cast<std::size_t>(row) * matrix->row_width +
                  DeepSeekSparseIndexOracle::kRecentWindow);
  }
  return matrix;
}

}  // namespace pih
