#include "pih/model/qwen3_teacher_forced_transaction_identity.h"

#include "pih/core/checked_math.h"

#include <utility>

namespace pih {

Result<QwenTeacherForcedTransactionIdentityReservation>
reserve_qwen_teacher_forced_transaction_identity(
    std::uint64_t first_plan_id, std::uint64_t first_event_generation) {
  if (first_plan_id == 0 || first_event_generation == 0)
    return Status::InvalidArgument(
        "Qwen teacher-forced transaction identity seed is invalid");
  auto target = checked_add_u64(first_plan_id, 1);
  if (!target.ok()) return target.status();
  auto readback = checked_add_u64(*target, 1);
  if (!readback.ok()) return readback.status();
  auto frontier = checked_add_u64(*readback, 1);
  if (!frontier.ok()) return frontier.status();
  auto next_plan = checked_add_u64(*frontier, 1);
  if (!next_plan.ok()) return next_plan.status();
  auto next_event = checked_add_u64(first_event_generation, 1);
  if (!next_event.ok()) return next_event.status();
  return QwenTeacherForcedTransactionIdentityReservation{
      first_plan_id, *target, *readback, *frontier, *next_plan,
      first_event_generation, *next_event};
}

Result<QwenTeacherForcedTransactionCompletion>
make_qwen_teacher_forced_transaction_completion(
    const QwenTeacherForcedTransactionIdentityReservation& identity,
    DriverEventHandle event, std::uintptr_t context_identity,
    std::uint64_t engine_epoch, std::uint32_t rank,
    std::uint64_t completion_frontier,
    std::uint64_t submit_ns, std::uint64_t deadline_ns) {
  if (identity.frontier_plan_id == 0 || identity.event_generation == 0 ||
      event == 0 || context_identity == 0 || engine_epoch == 0 ||
      completion_frontier == 0 || submit_ns >= deadline_ns)
    return Status::InvalidArgument(
        "Qwen teacher-forced transaction completion identity is invalid");
  auto slot = CompletionEventSlot::Create(event, context_identity);
  if (!slot.ok()) return slot.status();
  auto frontier = CudaCompletionFrontier::Create(
      {engine_epoch, rank, identity.frontier_plan_id,
       CudaCompletionPhase::kPrefill, completion_frontier},
      identity.event_generation, submit_ns, deadline_ns);
  if (!frontier.ok()) return frontier.status();
  return QwenTeacherForcedTransactionCompletion{
      std::move(*slot), std::move(*frontier)};
}

}  // namespace pih
