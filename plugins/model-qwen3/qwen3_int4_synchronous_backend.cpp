#include "pih/model/qwen3_int4_synchronous_backend.h"

#include <cmath>
#include <limits>

#include "pih/core/checked_math.h"

namespace pih {
namespace {
bool endpoint_matches(const CudaCopyEndpoint& endpoint,std::uintptr_t base,
                      std::uint64_t bytes,CudaCopyMemoryType type,
                      std::int32_t rank) {
  return endpoint.allocation_base==base && endpoint.allocation_bytes>=bytes &&
      endpoint.offset==0 && endpoint.owner_id!=0 && endpoint.generation!=0 &&
      endpoint.memory_type==type && endpoint.rank==static_cast<std::uint32_t>(rank) &&
      endpoint.device_or_numa>=0;
}
}

Result<QwenInt4SynchronousBackend> QwenInt4SynchronousBackend::Create(
    const QwenBf16CommandBuffer& commands,
    std::span<const ResolvedKernelFunction> functions,
    const QwenInt4WeightResourceSet& weights,
    QwenInt4SynchronousBackendArenas arenas,
    QwenInt4SynchronousBackendIdentity identity,
    QwenInt4SynchronousBackendDrivers drivers) {
  const bool drivers_valid=drivers.copy&&drivers.clear&&drivers.kernels&&
      drivers.lm_head&&drivers.events&&drivers.health&&drivers.clock&&drivers.waiter;
  if(!drivers_valid || identity.epoch==0 || identity.first_request_generation==0 ||
     identity.first_event_generation==0 || identity.first_plan_id==0 ||
     identity.timeout_ns==0 || identity.context_identity==0 || identity.stream==0 ||
     identity.event==0 || identity.owning_rank<0 || identity.slot_count==0 ||
     identity.slot_count>QwenKvSlotPool::kMaximumSlots ||
     !std::isfinite(identity.rms_epsilon) || identity.rms_epsilon<=0 ||
     !std::isfinite(identity.attention_scale) || identity.attention_scale<=0 ||
     functions.size()!=QwenInt4KernelBundle::kFunctionCount ||
     weights.owning_rank()!=identity.owning_rank ||
     drivers.copy->context_identity()!=identity.context_identity ||
     arenas.pinned_result_backing.size()!=QwenBf16StepResultLayout::kTotalBytes ||
     arenas.pinned_staging_backing.empty())
    return Status::InvalidArgument("Qwen INT4 synchronous backend configuration is invalid");
  const auto staging_base=reinterpret_cast<std::uintptr_t>(arenas.pinned_staging_backing.data());
  const auto result_base=reinterpret_cast<std::uintptr_t>(arenas.pinned_result_backing.data());
  if(!endpoint_matches(arenas.pinned_staging,staging_base,arenas.pinned_staging_backing.size(),CudaCopyMemoryType::kRegisteredPinnedHost,identity.owning_rank) ||
     !endpoint_matches(arenas.pinned_result,result_base,arenas.pinned_result_backing.size(),CudaCopyMemoryType::kRegisteredPinnedHost,identity.owning_rank) ||
     !endpoint_matches(arenas.device_staging,arenas.device.step_staging.base,arenas.device.step_staging.bytes,CudaCopyMemoryType::kDevice,identity.owning_rank) ||
     !endpoint_matches(arenas.sampled_token,arenas.device.sampled_token.base,8,CudaCopyMemoryType::kDevice,identity.owning_rank) ||
     !endpoint_matches(arenas.device_error,arenas.device.device_error.base,4,CudaCopyMemoryType::kDevice,identity.owning_rank))
    return Status::InvalidArgument("Qwen INT4 synchronous backend arenas mismatch");
  return QwenInt4SynchronousBackend(commands,functions,weights,arenas,identity,drivers);
}

Result<std::int64_t> QwenInt4SynchronousBackend::execute_and_read_token(
    std::span<const std::int64_t> tokens,std::uint64_t first_position,
    const QwenKvBlockTable& block_table,const QwenKvAppendPlan& append_plan) {
  if(state_!=QwenInt4SynchronousBackendState::kReady)
    return Status::FailedPrecondition("Qwen INT4 backend is not ready");
  if(next_request_generation_==UINT64_MAX || next_event_generation_==UINT64_MAX ||
     next_plan_id_>UINT64_MAX-7){state_=QwenInt4SynchronousBackendState::kPoisoned;
    return Status::ResourceExhausted("Qwen INT4 backend identity exhausted");}
  state_=QwenInt4SynchronousBackendState::kRunning;
  const auto fail=[this](Status status)->Result<std::int64_t>{
    state_=QwenInt4SynchronousBackendState::kPoisoned;return status;};
  const auto request_generation=next_request_generation_++;
  const auto event_generation=next_event_generation_++;
  const auto upload_plan=next_plan_id_;const auto readback_plan=next_plan_id_+5;
  next_plan_id_+=7;
  auto owners=arenas_.device;owners.device_error.generation=request_generation;
  auto built=QwenInt4StepBuilder::Create(tokens,first_position,block_table,append_plan,
      request_generation,identity_.owning_rank,identity_.slot_count,
      identity_.rms_epsilon,identity_.attention_scale,*commands_,functions_,
      *weights_,owners,arenas_.pinned_staging_backing);
  if(!built.ok())return fail(built.status());
  QwenInt4PreparedStepCompute compute(std::move(built->execution()));
  auto device_staging=arenas_.device_staging;device_staging.generation=owners.step_staging.generation;
  auto upload=QwenBf16StepUpload::Create(built->staging_layout(),arenas_.pinned_staging,
      device_staging,identity_.context_identity,identity_.stream,event_generation,upload_plan);
  if(!upload.ok())return fail(upload.status());
  auto sampled=arenas_.sampled_token;sampled.generation=owners.sampled_token.generation;
  auto error=arenas_.device_error;error.generation=request_generation;
  auto readback=QwenBf16StepReadback::Create(sampled,error,arenas_.pinned_result,
      identity_.context_identity,identity_.stream,event_generation,readback_plan);
  if(!readback.ok())return fail(readback.status());
  auto slot=CompletionEventSlot::Create(identity_.event,identity_.context_identity);
  if(!slot.ok())return fail(slot.status());
  auto submit_ns=drivers_.clock->now_ns();if(!submit_ns.ok())return fail(submit_ns.status());
  auto deadline=checked_add_u64(*submit_ns,identity_.timeout_ns);if(!deadline.ok())return fail(deadline.status());
  const auto phase=tokens.size()==1&&first_position!=0?CudaCompletionPhase::kDecode:CudaCompletionPhase::kPrefill;
  auto frontier=CudaCompletionFrontier::Create({identity_.epoch,static_cast<std::uint32_t>(identity_.owning_rank),request_generation,phase,append_plan.target_committed_tokens},event_generation,*submit_ns,*deadline);
  if(!frontier.ok())return fail(frontier.status());
  auto transaction=QwenInt4StepTransaction::Create(std::move(*upload),compute,
      std::move(*readback),std::move(*slot),std::move(*frontier),
      arenas_.pinned_result_backing,identity_.stream,event_generation,*drivers_.health);
  if(!transaction.ok())return fail(transaction.status());
  Status submitted=transaction->submit(*drivers_.copy,*drivers_.clear,*drivers_.kernels,
                                        *drivers_.lm_head,*drivers_.events);
  if(!submitted.ok())return fail(submitted);
  for(;;){
    auto now=drivers_.clock->now_ns();if(!now.ok())return fail(now.status());
    const Status expiry=transaction->expire(*now);
    if(expiry.code()!=StatusCode::kUnavailable)return fail(expiry);
    auto token=transaction->poll(*drivers_.events);
    if(token.ok()){
      const Status released=transaction->release_completion();if(!released.ok())return fail(released);
      last_completion_event_={identity_.event,event_generation};state_=QwenInt4SynchronousBackendState::kReady;return token;
    }
    if(token.status().code()!=StatusCode::kUnavailable)return fail(token.status());
    const Status waited=drivers_.waiter->wait();if(!waited.ok())return fail(waited);
  }
}

Result<QwenKvCompletionEvent> QwenInt4SynchronousBackend::last_completion_event() const {
  if(state_!=QwenInt4SynchronousBackendState::kReady || last_completion_event_.handle==0 || last_completion_event_.generation==0)
    return Status::FailedPrecondition("Qwen INT4 backend has no completion event");
  return last_completion_event_;
}

}  // namespace pih
