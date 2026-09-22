#include "pih/model/nvidia_qwen3_int4_capacity_probe.h"
#include <cuda.h>
#include "pih/backend/cuda/cuda_driver_status.h"
namespace pih{
Result<QwenInt4EngineBootstrapPlan> NvidiaQwenInt4CapacityProbe::Admit(
 const Qwen3Config& config,std::int32_t ordinal,std::uint32_t step,
 std::uint32_t sequence,std::uint32_t slots,std::uint64_t workspace,
 std::uint64_t pinned,std::uint64_t reserve,std::uint32_t maximum_batch_sequences){
 if(ordinal<0||pinned==0)return Status::InvalidArgument("NVIDIA Qwen INT4 capacity identity is invalid");
 auto model=QwenInt4EngineModelPlan::Create(config);if(!model.ok())return model.status();
 return Admit(std::move(*model),ordinal,step,sequence,slots,workspace,pinned,
              reserve,maximum_batch_sequences);
}
Result<QwenInt4EngineBootstrapPlan> NvidiaQwenInt4CapacityProbe::Admit(
 QwenInt4EngineModelPlan model,std::int32_t ordinal,std::uint32_t step,
 std::uint32_t sequence,std::uint32_t slots,std::uint64_t workspace,
 std::uint64_t pinned,std::uint64_t reserve,std::uint32_t maximum_batch_sequences){
 if(ordinal<0||pinned==0)return Status::InvalidArgument("NVIDIA Qwen INT4 capacity identity is invalid");
 CUcontext context=nullptr;CUresult result=cuCtxGetCurrent(&context);
 if(result!=CUDA_SUCCESS)return cuda_driver_status(result,"cuCtxGetCurrent");
 if(context==nullptr)return Status::FailedPrecondition("NVIDIA Qwen INT4 capacity probe requires current context");
 CUdevice device=0;result=cuCtxGetDevice(&device);
 if(result!=CUDA_SUCCESS)return cuda_driver_status(result,"cuCtxGetDevice");
 if(device!=ordinal)return Status::FailedPrecondition("current CUDA device differs from capacity owner");
 std::size_t free_bytes=0,total_bytes=0;result=cuMemGetInfo(&free_bytes,&total_bytes);
 if(result!=CUDA_SUCCESS)return cuda_driver_status(result,"cuMemGetInfo");
 if(free_bytes==0||total_bytes==0||free_bytes>total_bytes)
  return Status::Internal("CUDA memory capacity observation is invalid");
 return QwenInt4EngineBootstrapPlan::Create(std::move(model),step,sequence,slots,workspace,
  static_cast<std::uint64_t>(free_bytes),pinned,reserve,maximum_batch_sequences);
}}
