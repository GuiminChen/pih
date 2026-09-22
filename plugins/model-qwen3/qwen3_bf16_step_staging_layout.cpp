#include "pih/model/qwen3_bf16_step_staging_layout.h"

#include <cstring>

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_bf16_linear_shape.h"

namespace pih {
namespace {

Result<QwenBf16ArenaSpan> append_span(std::uint64_t& cursor,
                                      std::uint64_t count,
                                      std::uint64_t element_bytes) {
  auto offset = checked_align_up_u64(cursor,
                                     QwenBf16StepStagingLayout::kAlignment);
  auto bytes = checked_mul_u64(count, element_bytes);
  if (!offset.ok()) return offset.status();
  if (!bytes.ok()) return bytes.status();
  auto end = checked_add_u64(*offset, *bytes);
  if (!end.ok()) return end.status();
  cursor = *end;
  return QwenBf16ArenaSpan{*offset, *bytes};
}

template <typename T>
void copy_span(std::span<std::byte> destination, QwenBf16ArenaSpan span,
               std::span<const T> source) {
  std::memcpy(destination.data() + span.offset_bytes, source.data(),
              static_cast<std::size_t>(span.size_bytes));
}

}  // namespace

Result<QwenBf16StepStagingLayout> QwenBf16StepStagingLayout::Create(
    const QwenBf16StepInputPlan& input) {
  if (input.tokens().empty() || input.visible_handles().empty()) {
    return Status::InvalidArgument("Qwen step staging input is empty");
  }
  return CreateBounded(input.tokens().size(), input.visible_handles().size());
}

Result<QwenBf16StepStagingLayout> QwenBf16StepStagingLayout::CreateBounded(
    std::uint64_t token_count, std::uint64_t visible_handle_count) {
  if (token_count == 0 ||
      token_count > QwenBf16LinearShape::kMaximumTokensPerPlan ||
      visible_handle_count == 0 ||
      visible_handle_count > QwenKvSlotPool::kMaximumSlots) {
    return Status::InvalidArgument("Qwen step staging bounds are invalid");
  }
  QwenBf16StepStagingLayout result;
  result.token_count_ = token_count;
  result.visible_handle_count_ = visible_handle_count;
  std::uint64_t cursor = 0;
  auto token_ids = append_span(cursor, result.token_count_, sizeof(std::int64_t));
  auto positions = append_span(cursor, result.token_count_, sizeof(std::int64_t));
  auto append_handles =
      append_span(cursor, result.token_count_, sizeof(QwenKvBlockHandle));
  auto offsets = append_span(cursor, result.token_count_, sizeof(std::uint16_t));
  auto visible = append_span(cursor, result.visible_handle_count_,
                             sizeof(QwenKvBlockHandle));
  if (!token_ids.ok()) return token_ids.status();
  if (!positions.ok()) return positions.status();
  if (!append_handles.ok()) return append_handles.status();
  if (!offsets.ok()) return offsets.status();
  if (!visible.ok()) return visible.status();
  auto total = checked_align_up_u64(cursor, kAlignment);
  if (!total.ok()) return total.status();
  result.token_ids_ = *token_ids;
  result.positions_ = *positions;
  result.append_handles_ = *append_handles;
  result.token_offsets_ = *offsets;
  result.visible_handles_ = *visible;
  result.total_bytes_ = *total;
  return result;
}

Status QwenBf16StepStagingLayout::materialize(
    const QwenBf16StepInputPlan& input,
    std::span<std::byte> backing) const {
  if (input.tokens().size() != token_count_ ||
      input.positions().size() != token_count_ ||
      input.append_handles().size() != token_count_ ||
      input.token_offsets().size() != token_count_ ||
      input.visible_handles().size() != visible_handle_count_ ||
      backing.size() < total_bytes_) {
    return Status::InvalidArgument(
        "Qwen step staging backing or input identity drifted");
  }
  copy_span(backing, token_ids_, input.tokens());
  copy_span(backing, positions_, input.positions());
  copy_span(backing, append_handles_, input.append_handles());
  copy_span(backing, token_offsets_, input.token_offsets());
  copy_span(backing, visible_handles_, input.visible_handles());
  return Status::Ok();
}

}  // namespace pih
