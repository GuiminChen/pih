#include "pih/model/qwen3_int4_step_transaction.h"

#include <cstring>

namespace pih {

Status QwenInt4PreparedStepCompute::submit(
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenInt4LmHeadExecutionDriver& lm_head_driver,
    DriverStreamHandle stream) {
  return execution_.run(clear_driver,kernel_driver,lm_head_driver,stream);
}

Result<QwenInt4StepTransaction> QwenInt4StepTransaction::Create(
    QwenBf16StepUpload upload,QwenInt4StepCompute& compute,
    QwenBf16StepReadback readback,CompletionEventSlot slot,
    CudaCompletionFrontier frontier,std::span<std::byte> result,
    DriverStreamHandle stream,std::uint64_t event_generation,
    QwenBf16StepHealthProvider& health) {
  if(stream==0 || event_generation==0 ||
     upload.state()!=QwenBf16StepUploadState::kPrepared ||
     readback.state()!=QwenBf16StepReadbackState::kPrepared ||
     slot.state()!=CompletionEventSlotState::kIdle ||
     upload.stream()!=stream || readback.stream()!=stream ||
     upload.event_generation()!=event_generation ||
     readback.event_generation()!=event_generation ||
     frontier.event_generation()!=event_generation ||
     upload.context_identity()!=readback.context_identity() ||
     upload.context_identity()!=slot.context_identity() ||
     result.size()!=QwenBf16StepResultLayout::kTotalBytes)
    return Status::InvalidArgument("Qwen INT4 transaction identity is invalid");
  const Status initialized=QwenBf16StepResultLayout::initialize(result);
  if(!initialized.ok())return initialized;
  return QwenInt4StepTransaction(std::move(upload),compute,std::move(readback),
      std::move(slot),std::move(frontier),result,stream,event_generation,health);
}

Status QwenInt4StepTransaction::submit(
    TypedCopyDriver& copies,QwenBf16DeviceErrorClearDriver& clear,
    KernelLaunchDriver& kernels,QwenInt4LmHeadExecutionDriver& lm_head,
    CompletionEventDriver& events) {
  if(state_!=QwenInt4StepTransactionState::kPrepared)
    return Status::FailedPrecondition("Qwen INT4 transaction is not submit-ready");
  Status status=upload_.submit(copies);
  if(status.ok())status=compute_->submit(clear,kernels,lm_head,stream_);
  if(status.ok())status=readback_.submit(copies);
  if(status.ok())status=completion_slot_.record(events,stream_,event_generation_);
  if(!status.ok()){state_=QwenInt4StepTransactionState::kPoisoned;return status;}
  state_=QwenInt4StepTransactionState::kSubmitted; return Status::Ok();
}

Result<CompletionPublicationEvidence> QwenInt4StepTransaction::collect() {
  auto health=health_provider_->collect(); if(!health.ok())return health.status();
  std::uint32_t error=UINT32_MAX;
  std::memcpy(&error,pinned_result_backing_.data()+
      QwenBf16StepResultLayout::device_error().offset_bytes,sizeof(error));
  return CompletionPublicationEvidence{health->submit_thread_last_error_clean,
                                       error,health->engine_poisoned};
}

Result<std::int64_t> QwenInt4StepTransaction::poll(CompletionEventDriver& events) {
  if(state_!=QwenInt4StepTransactionState::kSubmitted)
    return Status::FailedPrecondition("Qwen INT4 transaction is not awaiting completion");
  const Status status=completion_slot_.poll(events,completion_frontier_,*this);
  if(!status.ok()){
    if(status.code()!=StatusCode::kUnavailable)state_=QwenInt4StepTransactionState::kPoisoned;
    return status;
  }
  auto token=QwenBf16StepResultLayout::parse(
      pinned_result_backing_,completion_frontier_.publication_authorized());
  if(!token.ok()){state_=QwenInt4StepTransactionState::kPoisoned;return token.status();}
  state_=QwenInt4StepTransactionState::kCompleted; return token;
}

Status QwenInt4StepTransaction::expire(std::uint64_t now_ns) {
  if(state_!=QwenInt4StepTransactionState::kSubmitted)
    return Status::FailedPrecondition("Qwen INT4 transaction is not awaiting completion");
  const Status status=completion_frontier_.expire(now_ns);
  if(status.code()!=StatusCode::kUnavailable)state_=QwenInt4StepTransactionState::kPoisoned;
  return status;
}

Status QwenInt4StepTransaction::release_completion() {
  if(state_!=QwenInt4StepTransactionState::kCompleted)
    return Status::FailedPrecondition("Qwen INT4 completion is not publishable");
  return completion_slot_.release(event_generation_);
}

}  // namespace pih
