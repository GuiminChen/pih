#include "pih/model/qwen3_int4_engine_resource_plan.h"
#include "pih/core/checked_math.h"
namespace pih{
Result<QwenInt4EngineResourcePlan> QwenInt4EngineResourcePlan::Create(
 std::uint32_t step,std::uint32_t sequence,std::uint32_t slots,
 std::uint64_t workspace,std::uint32_t maximum_batch_sequences){
 auto runtime=QwenBf16EngineResourcePlan::Create(
  step,sequence,slots,workspace,maximum_batch_sequences);
 if(!runtime.ok())return runtime.status();
 auto steady=checked_add_u64(runtime->step_staging_bytes(),runtime->pinned_result_bytes());
 if(!steady.ok())return steady.status();
 auto startup=checked_add_u64(*steady,QwenInt4ArtifactLayout::kOfficialFileBytes);
 if(!startup.ok())return startup.status();
 if(runtime->total_device_bytes()<runtime->resident_weight_bytes())
  return Status::Internal("Qwen INT4 runtime accounting underflow");
 const auto arena_bytes=runtime->total_device_bytes()-runtime->resident_weight_bytes();
 auto device=checked_add_u64(arena_bytes,QwenInt4ArtifactLayout::kOfficialLogicalPayloadBytes);
 if(!device.ok())return device.status();
 return QwenInt4EngineResourcePlan(std::move(*runtime),*steady,*startup,*device);
}
Status QwenInt4EngineResourcePlan::admit(std::uint64_t available_device_bytes,
 std::uint64_t available_pinned_bytes,std::uint64_t device_reserve_bytes)const{
 auto required=checked_add_u64(total_device_bytes_,device_reserve_bytes);
 if(!required.ok())return required.status();
 if(*required>available_device_bytes)
  return Status::ResourceExhausted("Qwen INT4 device capacity is unfunded");
 if(startup_pinned_peak_bytes_>available_pinned_bytes)
  return Status::ResourceExhausted("Qwen INT4 pinned startup capacity is unfunded");
 return Status::Ok();
}}
