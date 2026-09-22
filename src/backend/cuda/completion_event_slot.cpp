#include "pih/backend/cuda/completion_event_slot.h"

namespace pih {

Result<CompletionEventSlot> CompletionEventSlot::Create(
    DriverEventHandle event, std::uintptr_t context_identity) {
  if (event == 0 || context_identity == 0) {
    return Status::InvalidArgument(
        "CUDA completion event slot identity is invalid");
  }
  return CompletionEventSlot(event, context_identity);
}

Status CompletionEventSlot::record(CompletionEventDriver& driver,
                                   DriverStreamHandle stream,
                                   std::uint64_t event_generation) {
  if (state_ != CompletionEventSlotState::kIdle || stream == 0 ||
      event_generation == 0 || event_generation <= event_generation_) {
    return Status::FailedPrecondition(
        "CUDA completion event cannot be recorded in current generation");
  }
  const Status recorded = driver.record(event_, stream);
  if (!recorded.ok()) {
    state_ = CompletionEventSlotState::kPoisoned;
    return recorded;
  }
  event_generation_ = event_generation;
  state_ = CompletionEventSlotState::kRecorded;
  return Status::Ok();
}

Status CompletionEventSlot::poll(
    CompletionEventDriver& driver, CudaCompletionFrontier& frontier,
    CompletionEvidenceProvider& evidence_provider) {
  if (state_ != CompletionEventSlotState::kRecorded) {
    return Status::FailedPrecondition(
        "CUDA completion event must be recorded before query");
  }
  auto query = driver.query(event_);
  if (!query.ok()) {
    state_ = CompletionEventSlotState::kPoisoned;
    return query.status();
  }
  if (query.value() == CudaEventQueryResult::kNotReady) {
    return frontier.observe(event_generation_, query.value(), true, 0, false);
  }
  if (query.value() != CudaEventQueryResult::kSuccess) {
    const Status observed =
        frontier.observe(event_generation_, query.value(), true, 0, false);
    state_ = CompletionEventSlotState::kPoisoned;
    return observed;
  }
  auto evidence = evidence_provider.collect();
  if (!evidence.ok()) {
    state_ = CompletionEventSlotState::kPoisoned;
    return evidence.status();
  }
  const Status observed = frontier.observe(
      event_generation_, query.value(),
      evidence->submit_thread_last_error_clean, evidence->device_error_code,
      evidence->engine_poisoned);
  if (!observed.ok()) {
    state_ = CompletionEventSlotState::kPoisoned;
    return observed;
  }
  if (!frontier.publication_authorized()) {
    state_ = CompletionEventSlotState::kPoisoned;
    return Status::Internal(
        "CUDA event completed without publication authorization");
  }
  state_ = CompletionEventSlotState::kCompleted;
  return Status::Ok();
}

Status CompletionEventSlot::release(std::uint64_t event_generation) {
  if (state_ != CompletionEventSlotState::kCompleted ||
      event_generation != event_generation_) {
    return Status::FailedPrecondition(
        "CUDA completion event release generation is invalid");
  }
  state_ = CompletionEventSlotState::kIdle;
  return Status::Ok();
}

}  // namespace pih
