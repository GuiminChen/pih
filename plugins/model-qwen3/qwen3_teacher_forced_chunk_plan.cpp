#include "pih/model/qwen3_teacher_forced_chunk_plan.h"

#include <algorithm>

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenTeacherForcedChunkPlan> QwenTeacherForcedChunkPlan::Create(
    std::span<const std::uint64_t, kCategoryCount> category_rows,
    std::uint32_t maximum_rows_per_chunk) {
  if (maximum_rows_per_chunk == 0 ||
      maximum_rows_per_chunk > kMaximumRowsPerChunk)
    return Status::InvalidArgument("Qwen teacher-forced chunk width is invalid");
  std::array<std::uint64_t, kCategoryCount> frozen{};
  std::uint64_t total = 0, chunk_count = 0;
  for (std::size_t index = 0; index < frozen.size(); ++index) {
    const auto rows = category_rows[index];
    if (rows < kMinimumRowsPerCategory || rows > kMaximumRowsPerCategory)
      return Status::InvalidArgument("Qwen teacher-forced category row count is invalid");
    frozen[index] = rows;
    auto next = checked_add_u64(total, rows);
    if (!next.ok()) return next.status();
    total = *next;
    auto rounded = checked_add_u64(rows, maximum_rows_per_chunk - 1U);
    if (!rounded.ok()) return rounded.status();
    next = checked_add_u64(chunk_count, *rounded / maximum_rows_per_chunk);
    if (!next.ok()) return next.status();
    chunk_count = *next;
  }
  if (chunk_count > 2000000U)
    return Status::ResourceExhausted("Qwen teacher-forced chunk count is unbounded");
  std::vector<QwenTeacherForcedChunk> chunks;
  chunks.reserve(static_cast<std::size_t>(chunk_count));
  for (std::size_t category = 0; category < frozen.size(); ++category) {
    for (std::uint64_t begin = 0; begin < frozen[category];) {
      const auto rows = static_cast<std::uint32_t>((std::min)(
          static_cast<std::uint64_t>(maximum_rows_per_chunk),
          frozen[category] - begin));
      auto elements = checked_mul_u64(rows, kVocabularySize);
      if (!elements.ok()) return elements.status();
      auto bytes = checked_mul_u64(*elements, sizeof(float));
      if (!bytes.ok()) return bytes.status();
      chunks.push_back({static_cast<QwenTeacherForcedCategory>(category),
                        begin, rows, *bytes});
      begin += rows;
    }
  }
  return QwenTeacherForcedChunkPlan(frozen, std::move(chunks), total,
                                    maximum_rows_per_chunk);
}

}  // namespace pih
