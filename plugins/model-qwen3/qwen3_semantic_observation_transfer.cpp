#include "pih/model/qwen3_semantic_observation_transfer.h"

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Result<CudaCopyEndpoint> with_offset(CudaCopyEndpoint endpoint,
                                     std::uint64_t relative) {
  auto offset = checked_add_u64(endpoint.offset, relative);
  if (!offset.ok()) return offset.status();
  endpoint.offset = *offset;
  return endpoint;
}

}  // namespace

Result<QwenSemanticObservationTransfer>
QwenSemanticObservationTransfer::Create(
    CudaCopyEndpoint logits_source, CudaCopyEndpoint logits_destination,
    CudaCopyEndpoint kv_source, CudaCopyEndpoint kv_destination,
    const QwenKvSemanticObservationPlan& kv_plan,
    QwenSemanticObservationTransferIdentity identity) {
  const auto slices = kv_plan.slices();
  if (slices.empty() || identity.first_plan_id == 0 ||
      identity.context_identity == 0 || identity.diagnostic_stream == 0 ||
      identity.completion_event_generation == 0 ||
      slices.size() > UINT64_MAX - identity.first_plan_id) {
    return Status::InvalidArgument(
        "Qwen semantic observation transfer identity is invalid");
  }
  auto logits = CudaTypedCopyPlan::Create(
      identity.first_plan_id, CudaCopyPurpose::kDiagnostic,
      CudaCopyKind::kDeviceToHost, logits_source, logits_destination,
      kFinalLogitsBytes, 4, identity.context_identity,
      identity.diagnostic_stream, identity.completion_event_generation);
  if (!logits.ok()) return logits.status();
  std::vector<CudaTypedCopyPlan> plans;
  plans.reserve(1 + slices.size());
  plans.push_back(std::move(*logits));
  for (std::size_t index = 0; index < slices.size(); ++index) {
    auto source = with_offset(kv_source, slices[index].source_offset);
    if (!source.ok()) return source.status();
    auto destination =
        with_offset(kv_destination, slices[index].destination_offset);
    if (!destination.ok()) return destination.status();
    auto plan = CudaTypedCopyPlan::Create(
        identity.first_plan_id + index + 1,
        CudaCopyPurpose::kDiagnostic, CudaCopyKind::kDeviceToHost, *source,
        *destination, slices[index].bytes, 2, identity.context_identity,
        identity.diagnostic_stream, identity.completion_event_generation);
    if (!plan.ok()) return plan.status();
    plans.push_back(std::move(*plan));
  }
  return QwenSemanticObservationTransfer(
      std::move(plans), kv_plan.payload_bytes(),
      identity.completion_event_generation, identity.diagnostic_stream,
      identity.context_identity);
}

Status QwenSemanticObservationTransfer::submit(TypedCopyDriver& driver) {
  if (state_ != QwenSemanticObservationTransferState::kPrepared) {
    state_ = QwenSemanticObservationTransferState::kPoisoned;
    return Status::FailedPrecondition(
        "Qwen semantic observation transfer is not reusable");
  }
  for (auto& plan : plans_) {
    const Status status = plan.submit(driver);
    if (!status.ok()) {
      state_ = QwenSemanticObservationTransferState::kPoisoned;
      return status;
    }
  }
  state_ = QwenSemanticObservationTransferState::kSubmitted;
  return Status::Ok();
}

}  // namespace pih
