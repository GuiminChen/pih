#include "pih/model/qwen3_bf16_tap_event_pipeline.h"

namespace pih {

Result<QwenBf16TapEventPipeline> QwenBf16TapEventPipeline::Create(
    QwenBf16TapEvidenceRun evidence_run,
    DriverEventHandle event,
    std::uint64_t epoch) {
  if (epoch == 0 ||
      evidence_run.state() != QwenBf16TapEvidenceRunState::kPrepared) {
    return Status::InvalidArgument("Qwen tap event pipeline identity is invalid");
  }
  auto slot = CompletionEventSlot::Create(
      event, evidence_run.identity().primary_context_identity);
  if (!slot.ok()) return slot.status();
  return QwenBf16TapEventPipeline(std::move(evidence_run), std::move(*slot),
                                  epoch);
}

Status QwenBf16TapEventPipeline::poison(const char* message) {
  state_ = QwenBf16TapEventPipelineState::kPoisoned;
  return Status::FailedPrecondition(message);
}

Result<CudaCompletionFrontier> QwenBf16TapEventPipeline::make_frontier(
    std::uint64_t plan_id, std::uint64_t event_generation,
    std::uint64_t submit_ns, std::uint64_t deadline_ns) const {
  return CudaCompletionFrontier::Create(
      {epoch_, evidence_run_.identity().rank, plan_id,
       CudaCompletionPhase::kCopy, plan_id},
      event_generation, submit_ns, deadline_ns);
}

Result<QwenNumericalTapInlineSnapshotDriver>
QwenBf16TapEventPipeline::inline_snapshot_driver(
    TypedCopyDriver& copy_driver) {
  if (state_ != QwenBf16TapEventPipelineState::kCapturing) {
    return poison("Qwen tap event pipeline is not capturing");
  }
  auto driver = evidence_run_.inline_snapshot_driver(copy_driver);
  if (!driver.ok()) {
    state_ = QwenBf16TapEventPipelineState::kPoisoned;
    return driver.status();
  }
  return driver;
}

Status QwenBf16TapEventPipeline::record_snapshot(
    CompletionEventDriver& event_driver,
    std::uint64_t submit_ns,
    std::uint64_t deadline_ns) {
  if (state_ != QwenBf16TapEventPipelineState::kCapturing ||
      evidence_run_.state() != QwenBf16TapEvidenceRunState::kCapturing) {
    return poison("Qwen tap snapshot event is out of order");
  }
  const auto& identity = evidence_run_.identity();
  auto frontier = make_frontier(identity.snapshot_copy_plan_id,
                                identity.snapshot_event_generation,
                                submit_ns, deadline_ns);
  if (!frontier.ok()) {
    state_ = QwenBf16TapEventPipelineState::kPoisoned;
    return frontier.status();
  }
  const Status recorded = event_slot_.record(
      event_driver, identity.execution_stream,
      identity.snapshot_event_generation);
  if (!recorded.ok()) {
    state_ = QwenBf16TapEventPipelineState::kPoisoned;
    return recorded;
  }
  frontier_.emplace(std::move(*frontier));
  state_ = QwenBf16TapEventPipelineState::kSnapshotRecorded;
  return Status::Ok();
}

Status QwenBf16TapEventPipeline::poll_snapshot(
    CompletionEventDriver& event_driver,
    CompletionEvidenceProvider& evidence_provider) {
  if (state_ != QwenBf16TapEventPipelineState::kSnapshotRecorded ||
      !frontier_.has_value()) {
    return poison("Qwen tap snapshot poll is out of order");
  }
  const Status polled =
      event_slot_.poll(event_driver, *frontier_, evidence_provider);
  if (!polled.ok()) {
    if (polled.code() != StatusCode::kUnavailable) {
      state_ = QwenBf16TapEventPipelineState::kPoisoned;
    }
    return polled;
  }
  Status status = evidence_run_.complete_snapshot_batch(*frontier_);
  if (status.ok()) {
    status = event_slot_.release(
        evidence_run_.identity().snapshot_event_generation);
  }
  if (!status.ok()) {
    state_ = QwenBf16TapEventPipelineState::kPoisoned;
    return status;
  }
  frontier_.reset();
  state_ = QwenBf16TapEventPipelineState::kHostReady;
  return Status::Ok();
}

Status QwenBf16TapEventPipeline::submit_host_and_record(
    TypedCopyDriver& copy_driver,
    CompletionEventDriver& event_driver,
    std::uint64_t submit_ns,
    std::uint64_t deadline_ns) {
  if (state_ != QwenBf16TapEventPipelineState::kHostReady) {
    return poison("Qwen tap host event is out of order");
  }
  const auto& identity = evidence_run_.identity();
  auto frontier = make_frontier(identity.host_copy_plan_id,
                                identity.host_event_generation,
                                submit_ns, deadline_ns);
  if (!frontier.ok()) {
    state_ = QwenBf16TapEventPipelineState::kPoisoned;
    return frontier.status();
  }
  Status status = evidence_run_.submit_host_batch(copy_driver);
  if (status.ok()) {
    status = event_slot_.record(event_driver, identity.diagnostic_stream,
                                identity.host_event_generation);
  }
  if (!status.ok()) {
    state_ = QwenBf16TapEventPipelineState::kPoisoned;
    return status;
  }
  frontier_.emplace(std::move(*frontier));
  state_ = QwenBf16TapEventPipelineState::kHostRecorded;
  return Status::Ok();
}

Status QwenBf16TapEventPipeline::poll_host(
    CompletionEventDriver& event_driver,
    CompletionEvidenceProvider& evidence_provider) {
  if (state_ != QwenBf16TapEventPipelineState::kHostRecorded ||
      !frontier_.has_value()) {
    return poison("Qwen tap host poll is out of order");
  }
  const Status polled =
      event_slot_.poll(event_driver, *frontier_, evidence_provider);
  if (!polled.ok()) {
    if (polled.code() != StatusCode::kUnavailable) {
      state_ = QwenBf16TapEventPipelineState::kPoisoned;
    }
    return polled;
  }
  Status status = evidence_run_.complete_host_batch(*frontier_);
  if (status.ok()) {
    status = event_slot_.release(
        evidence_run_.identity().host_event_generation);
  }
  if (!status.ok()) {
    state_ = QwenBf16TapEventPipelineState::kPoisoned;
    return status;
  }
  frontier_.reset();
  state_ = QwenBf16TapEventPipelineState::kHostComplete;
  return Status::Ok();
}

Status QwenBf16TapEventPipeline::expire(std::uint64_t now_ns) {
  if ((state_ != QwenBf16TapEventPipelineState::kSnapshotRecorded &&
       state_ != QwenBf16TapEventPipelineState::kHostRecorded) ||
      !frontier_.has_value()) {
    return poison("Qwen tap event pipeline has no expirable frontier");
  }
  const Status status = frontier_->expire(now_ns);
  if (status.code() != StatusCode::kUnavailable) {
    state_ = QwenBf16TapEventPipelineState::kPoisoned;
  }
  return status;
}

Result<QwenNumericalTapRunReceipt> QwenBf16TapEventPipeline::seal(
    const QwenNumericalTapArenas& arenas) {
  if (state_ != QwenBf16TapEventPipelineState::kHostComplete) {
    return poison("Qwen tap event pipeline cannot be sealed");
  }
  auto receipt = evidence_run_.seal(arenas);
  if (!receipt.ok()) {
    state_ = QwenBf16TapEventPipelineState::kPoisoned;
    return receipt.status();
  }
  state_ = QwenBf16TapEventPipelineState::kSealed;
  return receipt;
}

}  // namespace pih
