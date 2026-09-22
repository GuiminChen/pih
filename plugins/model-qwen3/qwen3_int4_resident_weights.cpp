#include "pih/model/qwen3_int4_resident_weights.h"

#include <array>
#include "pih/core/checked_math.h"

namespace pih {

Result<QwenInt4ResidentWeights> QwenInt4ResidentWeights::Create(
    const QwenInt4ArtifactLayout& layout,
    const QwenInt4LinearShapeLedger& ledger, std::uint64_t engine_epoch,
    Allocator& allocator,
    CudaCopyEndpoint canonical_file, std::int32_t owning_rank,
    std::uint64_t destination_owner_id,
    std::uintptr_t primary_context_identity, DriverStreamHandle stream,
    DriverEventHandle completion_event,
    std::uint64_t completion_event_generation,
    std::uint64_t first_plan_id, std::uint64_t submit_ns,
    std::uint64_t deadline_ns) {
  if (engine_epoch == 0 || owning_rank < 0 || canonical_file.rank !=
          static_cast<std::uint32_t>(owning_rank) ||
      destination_owner_id == 0 ||
      destination_owner_id == canonical_file.owner_id || submit_ns == 0 ||
      deadline_ns <= submit_ns) {
    return Status::InvalidArgument("Qwen INT4 resident identity is invalid");
  }
  auto backing = Buffer::Allocate(
      allocator,layout.logical_payload_bytes(),256);
  if (!backing.ok()) return backing.status();
  if (backing->device().type() != DeviceType::kCuda ||
      backing->device().index() != owning_rank || backing->generation() == 0) {
    return Status::InvalidArgument("Qwen INT4 resident backing is not CUDA");
  }
  const std::array<std::int64_t,1> shape{
      static_cast<std::int64_t>(layout.logical_payload_bytes())};
  auto pool = backing->view(DType::kUInt8,shape);
  if (!pool.ok()) return pool.status();
  auto resources = QwenInt4WeightResourceSet::Create(
      layout,ledger,*pool,owning_rank);
  if (!resources.ok()) return resources.status();
  CudaCopyEndpoint destination{
      reinterpret_cast<std::uintptr_t>(backing->data()),backing->size_bytes(),
      0,destination_owner_id,backing->generation(),CudaCopyMemoryType::kDevice,
      static_cast<std::uint32_t>(owning_rank),backing->device().index()};
  auto upload = QwenInt4WeightUploadPlan::Create(
      layout,canonical_file,destination,primary_context_identity,stream,
      completion_event_generation,first_plan_id);
  if (!upload.ok()) return upload.status();
  auto slot=CompletionEventSlot::Create(completion_event,
                                        primary_context_identity);
  if(!slot.ok())return slot.status();
  auto final_plan_id=checked_add_u64(first_plan_id,
      static_cast<std::uint64_t>(layout.payload_record_count()-1));
  if(!final_plan_id.ok())return final_plan_id.status();
  auto frontier=CudaCompletionFrontier::Create(
      {engine_epoch,static_cast<std::uint32_t>(owning_rank),
       first_plan_id,CudaCompletionPhase::kCopy,*final_plan_id},
      completion_event_generation,submit_ns,deadline_ns);
  if(!frontier.ok())return frontier.status();
  return QwenInt4ResidentWeights(
      std::move(*backing),std::move(*resources),std::move(*upload),
      std::move(*slot),std::move(*frontier),stream,
      completion_event_generation);
}

Status QwenInt4ResidentWeights::submit(
    TypedCopyDriver& copy_driver, CompletionEventDriver& event_driver) {
  if(state_!=QwenInt4ResidentWeightState::kPrepared)
    return Status::FailedPrecondition("Qwen INT4 resident is not submit-ready");
  auto status=upload_.submit(copy_driver);
  if(status.ok()) status=event_slot_.record(event_driver,stream_,event_generation_);
  if(!status.ok()) { state_=QwenInt4ResidentWeightState::kPoisoned; return status; }
  state_=QwenInt4ResidentWeightState::kSubmitted;
  return Status::Ok();
}

Status QwenInt4ResidentWeights::poll(
    CompletionEventDriver& event_driver,
    CompletionEvidenceProvider& evidence_provider) {
  if(state_!=QwenInt4ResidentWeightState::kSubmitted)
    return Status::FailedPrecondition("Qwen INT4 resident is not poll-ready");
  const auto status=event_slot_.poll(event_driver,frontier_,evidence_provider);
  if(!status.ok()) {
    if(event_slot_.state()==CompletionEventSlotState::kPoisoned)
      state_=QwenInt4ResidentWeightState::kPoisoned;
    return status;
  }
  if(event_slot_.state()!=CompletionEventSlotState::kCompleted ||
     !frontier_.publication_authorized()) {
    state_=QwenInt4ResidentWeightState::kPoisoned;
    return Status::Internal("Qwen INT4 publication frontier drifted");
  }
  state_=QwenInt4ResidentWeightState::kPublished;
  return Status::Ok();
}

Status QwenInt4ResidentWeights::expire(std::uint64_t now_ns) {
  if(state_!=QwenInt4ResidentWeightState::kSubmitted)
    return Status::FailedPrecondition("Qwen INT4 resident is not awaiting completion");
  const Status status=frontier_.expire(now_ns);
  if(status.code()!=StatusCode::kUnavailable)
    state_=QwenInt4ResidentWeightState::kPoisoned;
  return status;
}

Result<const QwenInt4WeightResourceSet*>
QwenInt4ResidentWeights::resources() const {
  if(state_!=QwenInt4ResidentWeightState::kPublished)
    return Status::FailedPrecondition("Qwen INT4 weights are not published");
  return &pending_resources_;
}

}  // namespace pih
