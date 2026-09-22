#pragma once

#include <cstdint>

#include "pih/backend/cuda/completion_event_slot.h"

namespace pih {

struct QwenTeacherForcedTransactionIdentityReservation final {
  std::uint64_t row_upload_plan_id = 0;
  std::uint64_t target_upload_plan_id = 0;
  std::uint64_t readback_plan_id = 0;
  std::uint64_t frontier_plan_id = 0;
  std::uint64_t next_plan_id = 0;
  std::uint64_t event_generation = 0;
  std::uint64_t next_event_generation = 0;
};

struct QwenTeacherForcedTransactionCompletion final {
  CompletionEventSlot slot;
  CudaCompletionFrontier frontier;
};

Result<QwenTeacherForcedTransactionIdentityReservation>
reserve_qwen_teacher_forced_transaction_identity(
    std::uint64_t first_plan_id, std::uint64_t first_event_generation);

Result<QwenTeacherForcedTransactionCompletion>
make_qwen_teacher_forced_transaction_completion(
    const QwenTeacherForcedTransactionIdentityReservation& identity,
    DriverEventHandle event, std::uintptr_t context_identity,
    std::uint64_t engine_epoch, std::uint32_t rank,
    std::uint64_t completion_frontier,
    std::uint64_t submit_ns, std::uint64_t deadline_ns);

}  // namespace pih
