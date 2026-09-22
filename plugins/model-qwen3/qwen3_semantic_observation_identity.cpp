#include "pih/model/qwen3_semantic_observation_identity.h"

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenSemanticObservationIdentityReservation>
reserve_qwen_semantic_observation_identity(
    std::uint64_t first_plan_id, std::uint64_t first_event_generation,
    std::uint64_t kv_slice_count) {
  if (first_plan_id == 0 || first_event_generation == 0 ||
      kv_slice_count == 0) {
    return Status::InvalidArgument(
        "Qwen semantic observation identity seed is invalid");
  }
  auto copy_count = checked_add_u64(kv_slice_count, 1);
  if (!copy_count.ok()) return copy_count.status();
  auto frontier = checked_add_u64(first_plan_id, *copy_count);
  if (!frontier.ok()) return frontier.status();
  auto next_plan = checked_add_u64(*frontier, 1);
  if (!next_plan.ok()) return next_plan.status();
  auto next_event = checked_add_u64(first_event_generation, 1);
  if (!next_event.ok()) return next_event.status();
  return QwenSemanticObservationIdentityReservation{
      first_plan_id, *copy_count, *frontier, *next_plan,
      first_event_generation, *next_event};
}

}  // namespace pih
