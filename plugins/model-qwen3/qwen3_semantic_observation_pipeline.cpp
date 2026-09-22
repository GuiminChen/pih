#include "pih/model/qwen3_semantic_observation_pipeline.h"

namespace pih {

Result<QwenSemanticObservationPipeline>
QwenSemanticObservationPipeline::Create(
    QwenSemanticObservationTransfer transfer, DriverEventHandle event,
    std::uint64_t epoch, std::uint32_t rank,
    std::uint64_t frontier_plan_id) {
  if (transfer.state() != QwenSemanticObservationTransferState::kPrepared ||
      epoch == 0 || rank == UINT32_MAX || frontier_plan_id == 0) {
    return Status::InvalidArgument(
        "Qwen semantic observation pipeline identity is invalid");
  }
  auto slot = CompletionEventSlot::Create(event, transfer.context_identity());
  if (!slot.ok()) return slot.status();
  return QwenSemanticObservationPipeline(
      std::move(transfer), std::move(*slot), epoch, rank, frontier_plan_id);
}

Status QwenSemanticObservationPipeline::poison(const char* message) {
  state_ = QwenSemanticObservationPipelineState::kPoisoned;
  return Status::FailedPrecondition(message);
}

Status QwenSemanticObservationPipeline::submit(
    TypedCopyDriver& copy_driver, CompletionEventDriver& event_driver,
    std::uint64_t submit_ns, std::uint64_t deadline_ns) {
  if (state_ != QwenSemanticObservationPipelineState::kPrepared) {
    return poison("Qwen semantic observation submit is out of order");
  }
  auto frontier = CudaCompletionFrontier::Create(
      {epoch_, rank_, frontier_plan_id_, CudaCompletionPhase::kCopy,
       frontier_plan_id_},
      transfer_.completion_event_generation(), submit_ns, deadline_ns);
  if (!frontier.ok()) {
    state_ = QwenSemanticObservationPipelineState::kPoisoned;
    return frontier.status();
  }
  Status status = transfer_.submit(copy_driver);
  if (status.ok()) {
    status = event_slot_.record(
        event_driver, transfer_.diagnostic_stream(),
        transfer_.completion_event_generation());
  }
  if (!status.ok()) {
    state_ = QwenSemanticObservationPipelineState::kPoisoned;
    return status;
  }
  frontier_.emplace(std::move(*frontier));
  state_ = QwenSemanticObservationPipelineState::kRecorded;
  return Status::Ok();
}

Status QwenSemanticObservationPipeline::poll(
    CompletionEventDriver& event_driver,
    CompletionEvidenceProvider& evidence_provider) {
  if (state_ != QwenSemanticObservationPipelineState::kRecorded ||
      !frontier_.has_value()) {
    return poison("Qwen semantic observation poll is out of order");
  }
  Status status =
      event_slot_.poll(event_driver, *frontier_, evidence_provider);
  if (!status.ok()) {
    if (status.code() != StatusCode::kUnavailable) {
      state_ = QwenSemanticObservationPipelineState::kPoisoned;
    }
    return status;
  }
  status = event_slot_.release(transfer_.completion_event_generation());
  if (!status.ok()) {
    state_ = QwenSemanticObservationPipelineState::kPoisoned;
    return status;
  }
  state_ = QwenSemanticObservationPipelineState::kComplete;
  return Status::Ok();
}

Status QwenSemanticObservationPipeline::await(
    CompletionEventDriver& event_driver,
    CompletionEvidenceProvider& evidence_provider,
    QwenBf16MonotonicClock& clock, QwenBf16PollWaiter& waiter) {
  if (state_ != QwenSemanticObservationPipelineState::kRecorded ||
      !frontier_.has_value()) {
    return poison("Qwen semantic observation await is out of order");
  }
  for (;;) {
    auto now = clock.now_ns();
    if (!now.ok()) {
      state_ = QwenSemanticObservationPipelineState::kPoisoned;
      return now.status();
    }
    const Status expiry = frontier_->expire(*now);
    if (expiry.code() != StatusCode::kUnavailable) {
      state_ = QwenSemanticObservationPipelineState::kPoisoned;
      return expiry;
    }
    const Status status = poll(event_driver, evidence_provider);
    if (status.ok()) return status;
    if (status.code() != StatusCode::kUnavailable) return status;
    const Status waited = waiter.wait();
    if (!waited.ok()) {
      state_ = QwenSemanticObservationPipelineState::kPoisoned;
      return waited;
    }
  }
}

Status QwenSemanticObservationPipeline::publish(
    std::span<const std::byte> logits, std::span<const std::byte> kv,
    QwenSemanticOutcomeRecorder& recorder) {
  if (state_ != QwenSemanticObservationPipelineState::kComplete ||
      logits.size() != transfer_.logits_bytes() ||
      kv.size() != transfer_.kv_bytes()) {
    return poison("Qwen semantic observation publication is invalid");
  }
  Status status = recorder.record_final_logits(logits);
  if (status.ok()) status = recorder.record_kv_state(kv);
  if (!status.ok()) {
    state_ = QwenSemanticObservationPipelineState::kPoisoned;
    return status;
  }
  state_ = QwenSemanticObservationPipelineState::kPublished;
  return Status::Ok();
}

}  // namespace pih
