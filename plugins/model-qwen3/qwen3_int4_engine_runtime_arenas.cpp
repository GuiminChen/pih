#include "pih/model/qwen3_int4_engine_runtime_arenas.h"
namespace pih{
Result<QwenInt4EngineRuntimeArenas> QwenInt4EngineRuntimeArenas::Allocate(
 const QwenInt4EngineBootstrapPlan& plan,Allocator& device_allocator,
 RegisteredPinnedAllocator& pinned_allocator,PinnedPlacementVerifier& verifier,
 std::int32_t numa_node,std::int32_t owning_rank){
 return Allocate(plan.resources(),device_allocator,pinned_allocator,verifier,
                 numa_node,owning_rank);
}
Result<QwenInt4EngineRuntimeArenas> QwenInt4EngineRuntimeArenas::Allocate(
 const QwenInt4EngineResourcePlan& plan,Allocator& device_allocator,
 RegisteredPinnedAllocator& pinned_allocator,PinnedPlacementVerifier& verifier,
 std::int32_t numa_node,std::int32_t owning_rank){
 if(numa_node<0||owning_rank<0)return Status::InvalidArgument("Qwen INT4 arena owner is invalid");
 auto device=QwenBf16EngineDeviceArenas::Allocate(plan.runtime(),device_allocator,owning_rank);
 if(!device.ok())return device.status();
 auto pinned=QwenBf16PinnedHostArenas::AllocateVerified(plan.runtime(),pinned_allocator,verifier,numa_node);
 if(!pinned.ok())return pinned.status();
 return QwenInt4EngineRuntimeArenas(std::move(*device),std::move(*pinned));
}
Result<QwenInt4SynchronousBackendArenas> QwenInt4EngineRuntimeArenas::backend_arenas(
 QwenInt4BackendArenaOwnerIds ids,std::int32_t rank){
 if(rank<0||ids.pinned_staging==0||ids.device_staging==0||ids.sampled_token==0||
    ids.device_error==0||ids.pinned_result==0)
  return Status::InvalidArgument("Qwen INT4 backend arena identity is invalid");
 const auto owners=device_.step_owners();
 const auto endpoint=[rank](const QwenBf16DeviceArenaOwner& owner,std::uint64_t id){
  return CudaCopyEndpoint{owner.base,owner.bytes,0,id,owner.generation,
   CudaCopyMemoryType::kDevice,static_cast<std::uint32_t>(rank),rank};};
 auto staging=pinned_.staging_endpoint(ids.pinned_staging,static_cast<std::uint32_t>(rank));
 if(!staging.ok())return staging.status();
 auto result=pinned_.result_endpoint(ids.pinned_result,static_cast<std::uint32_t>(rank));
 if(!result.ok())return result.status();
 return QwenInt4SynchronousBackendArenas{owners,*staging,
  endpoint(owners.step_staging,ids.device_staging),
  endpoint(owners.sampled_token,ids.sampled_token),
  endpoint(owners.device_error,ids.device_error),*result,pinned_.staging(),pinned_.result()};
}}
