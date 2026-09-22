#include "pih/model/deepseek_compression_append_plan.h"

#include <limits>

#include "pih/model/deepseek_attention_page_mapper.h"

namespace pih {

Result<DeepSeekCompressionAppendPlan> DeepSeekCompressionAppendPlanner::Plan(
    std::uint64_t committed_source_tokens,
    std::uint32_t appended_source_tokens,
    const DeepSeekStatePoolGeometry& geometry) {
  if (appended_source_tokens == 0 ||
      committed_source_tokens > 1048576ULL ||
      appended_source_tokens > 1048576ULL - committed_source_tokens) {
    return Status::InvalidArgument(
        "DeepSeek compression append range is invalid");
  }
  auto before = DeepSeekAttentionPageMapper::Coverage(
      committed_source_tokens, geometry);
  if (!before.ok()) return before.status();
  const auto source_end = committed_source_tokens + appended_source_tokens;
  auto after = DeepSeekAttentionPageMapper::Coverage(source_end, geometry);
  if (!after.ok()) return after.status();
  const auto new_slots = after->stored_slots - before->stored_slots;
  std::uint64_t first_page = 0;
  std::uint64_t touched_pages = 0;
  if (new_slots != 0) {
    auto first = DeepSeekAttentionPageMapper::LocateStoredSlot(
        before->stored_slots, geometry);
    auto last = DeepSeekAttentionPageMapper::LocateStoredSlot(
        after->stored_slots - 1U, geometry);
    if (!first.ok()) return first.status();
    if (!last.ok()) return last.status();
    first_page = first->page_ordinal;
    touched_pages = last->page_ordinal - first_page + 1U;
  }
  return DeepSeekCompressionAppendPlan{
      committed_source_tokens,
      source_end,
      before->stored_slots,
      new_slots,
      first_page,
      touched_pages,
      before->compressor_remainder_tokens,
      after->compressor_remainder_tokens};
}

}  // namespace pih
