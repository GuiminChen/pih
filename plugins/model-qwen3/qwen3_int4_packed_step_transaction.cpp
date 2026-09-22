#include "pih/model/qwen3_int4_packed_step_transaction.h"

#include <cstring>

namespace pih {

Result<QwenInt4PackedStepTransaction> QwenInt4PackedStepTransaction::Create(
    QwenBf16PackedStepUpload upload,
    QwenInt4PackedPreparedExecution& compute,
    QwenBf16PackedReadback readback,QwenBf16PackedResultLayout layout,
    std::uint32_t sample_count,CompletionEventSlot slot,
    CudaCompletionFrontier frontier,std::span<std::byte> result,
    DriverStreamHandle stream,std::uint64_t event_generation,
    QwenBf16StepHealthProvider& health) {
  if(sample_count>layout.sample_capacity() || stream==0 ||
     event_generation==0 ||
     upload.state()!=QwenBf16PackedStepUploadState::kPrepared ||
     readback.state()!=QwenBf16PackedReadbackState::kPrepared ||
     slot.state()!=CompletionEventSlotState::kIdle ||
     upload.stream()!=stream || readback.stream()!=stream ||
     upload.event_generation()!=event_generation ||
     readback.event_generation()!=event_generation ||
     frontier.event_generation()!=event_generation ||
     upload.context_identity()!=readback.context_identity() ||
     upload.context_identity()!=slot.context_identity() ||
     result.size()!=layout.total_bytes()) {
    return Status::InvalidArgument(
        "packed INT4 transaction identity or state is invalid");
  }
  auto initialized=layout.initialize(result);if(!initialized.ok())return initialized;
  return QwenInt4PackedStepTransaction(std::move(upload),compute,
      std::move(readback),layout,sample_count,std::move(slot),
      std::move(frontier),result,stream,event_generation,health);
}

Status QwenInt4PackedStepTransaction::submit(
    TypedCopyDriver& copy_driver,
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenInt4LmHeadExecutionDriver& lm_head_driver,
    CompletionEventDriver& event_driver) {
  if(state_!=QwenInt4PackedStepTransactionState::kPrepared)
    return Status::FailedPrecondition("packed INT4 step is not submit-ready");
  auto status=upload_.submit(copy_driver);
  if(status.ok())status=compute_->run(clear_driver,kernel_driver,
                                      lm_head_driver,stream_);
  if(status.ok())status=readback_.submit(copy_driver);
  if(status.ok())status=completion_slot_.record(
      event_driver,stream_,event_generation_);
  if(!status.ok()){state_=QwenInt4PackedStepTransactionState::kPoisoned;return status;}
  state_=QwenInt4PackedStepTransactionState::kSubmitted;return Status::Ok();
}

Result<CompletionPublicationEvidence> QwenInt4PackedStepTransaction::collect(){
  auto health=health_provider_->collect();if(!health.ok())return health.status();
  std::uint32_t error=UINT32_MAX;
  std::memcpy(&error,pinned_result_backing_.data()+
      result_layout_.device_error().offset_bytes,sizeof(error));
  return CompletionPublicationEvidence{health->submit_thread_last_error_clean,
                                       error,health->engine_poisoned};
}

Result<std::vector<QwenBf16PackedSampleReceipt>>
QwenInt4PackedStepTransaction::poll_sampling(CompletionEventDriver& driver){
  if(state_!=QwenInt4PackedStepTransactionState::kSubmitted)
    return Status::FailedPrecondition(
        "packed INT4 step is not awaiting completion");
  auto status=completion_slot_.poll(driver,completion_frontier_,*this);
  if(!status.ok()){
    if(status.code()!=StatusCode::kUnavailable)
      state_=QwenInt4PackedStepTransactionState::kPoisoned;
    return status;
  }
  auto receipts=result_layout_.parse_sampling(
      pinned_result_backing_,sample_count_,
      completion_frontier_.publication_authorized());
  if(!receipts.ok()){
    state_=QwenInt4PackedStepTransactionState::kPoisoned;
    return receipts.status();
  }
  state_=QwenInt4PackedStepTransactionState::kCompleted;return receipts;
}

Status QwenInt4PackedStepTransaction::expire(std::uint64_t now_ns){
  if(state_!=QwenInt4PackedStepTransactionState::kSubmitted)
    return Status::FailedPrecondition(
        "packed INT4 step is not awaiting completion");
  auto status=completion_frontier_.expire(now_ns);
  if(status.code()!=StatusCode::kUnavailable)
    state_=QwenInt4PackedStepTransactionState::kPoisoned;
  return status;
}

Status QwenInt4PackedStepTransaction::release_completion(){
  if(state_!=QwenInt4PackedStepTransactionState::kCompleted)
    return Status::FailedPrecondition(
        "packed INT4 completion cannot be released before publication");
  return completion_slot_.release(event_generation_);
}

}  // namespace pih
