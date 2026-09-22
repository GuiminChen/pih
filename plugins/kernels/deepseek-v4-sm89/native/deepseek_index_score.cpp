#include "pih/backend/cuda/deepseek_index_score.h"

#include <limits>

namespace pih {

Status validate_deepseek_index_score_launch(
    const DeepSeekIndexScoreLaunch& launch) {
  const auto blocks = static_cast<std::uint64_t>(launch.query_count) *
                      launch.slot_count;
  const auto paged = launch.page_slots_u32 != 0;
  const auto logical_end = static_cast<std::uint64_t>(launch.slot_base) +
                           launch.slot_count;
  if (launch.query_bf16 == 0 || launch.index_kv_bf16 == 0 ||
      launch.head_weight_f32 == 0 || launch.score_f32 == 0 ||
      launch.error_flag_u32 == 0 || launch.stream == 0 ||
      launch.query_count == 0 || launch.query_count > 4096 ||
      launch.head_count == 0 ||
      launch.head_count > DeepSeekIndexScoreLaunch::kMaximumHeads ||
      launch.slot_count == 0 ||
      launch.slot_count > DeepSeekIndexScoreLaunch::kMaximumSlotTile ||
      (paged && (launch.logical_page_count == 0 ||
                 launch.physical_page_count == 0 ||
                 logical_end >
                     static_cast<std::uint64_t>(launch.logical_page_count) *
                         64U)) ||
      (!paged && (launch.slot_base != 0 ||
                  launch.logical_page_count != 0 ||
                  launch.physical_page_count != 0)) ||
      blocks > static_cast<std::uint64_t>(
                   std::numeric_limits<std::int32_t>::max())) {
    return Status::InvalidArgument("DeepSeek index score launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
