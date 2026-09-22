#include "pih/model/nvidia_qwen3_int4_execution_driver.h"

#include <algorithm>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_driver_status.h"

#if CUDA_VERSION < 13020
#error "PIH Qwen INT4 execution driver requires CUDA Toolkit 13.2 or newer"
#endif

namespace pih {

Result<NvidiaQwenInt4ExecutionDriver> NvidiaQwenInt4ExecutionDriver::Create(
    void* workspace,std::uint64_t workspace_bytes,
    std::uint64_t maximum_workspace_bytes,std::int32_t owning_rank,
    std::uint32_t maximum_lm_head_rows) {
  if(owning_rank<0 || maximum_workspace_bytes>workspace_bytes ||
     (workspace_bytes!=0 && workspace==nullptr) ||
     maximum_lm_head_rows==0 || maximum_lm_head_rows>32)
    return Status::InvalidArgument("NVIDIA Qwen INT4 workspace identity is invalid");
  CUcontext context=nullptr;CUresult result=cuCtxGetCurrent(&context);
  if(result!=CUDA_SUCCESS)return cuda_driver_status(result,"cuCtxGetCurrent");
  if(context==nullptr)return Status::FailedPrecondition("NVIDIA Qwen INT4 driver requires current context");
  CUdevice device=0;result=cuCtxGetDevice(&device);
  if(result!=CUDA_SUCCESS)return cuda_driver_status(result,"cuCtxGetDevice");
  if(device!=owning_rank)return Status::FailedPrecondition("current CUDA device differs from Qwen INT4 owner");
  std::vector<std::unique_ptr<GemmPlan>> plans;plans.reserve(maximum_lm_head_rows);
  std::uint64_t required=0;
  for(std::uint32_t rows=1;rows<=maximum_lm_head_rows;++rows){
    auto plan=GemmPlan::Create(rows,151936,1024,maximum_workspace_bytes,
                               DType::kFloat32);
    if(!plan.ok())return plan.status();
    required=(std::max)(required,(*plan)->workspace_bytes());
    plans.push_back(std::move(*plan));
  }
  if(required>workspace_bytes || (required!=0 && workspace==nullptr))
    return Status::ResourceExhausted("NVIDIA Qwen INT4 LM head workspace is too small");
  return NvidiaQwenInt4ExecutionDriver(std::move(plans),required,workspace,
      workspace_bytes,owning_rank,reinterpret_cast<std::uintptr_t>(context));
}

Status NvidiaQwenInt4ExecutionDriver::require_current_owner() const {
  CUcontext context=nullptr;CUresult result=cuCtxGetCurrent(&context);
  if(result!=CUDA_SUCCESS)return cuda_driver_status(result,"cuCtxGetCurrent");
  if(reinterpret_cast<std::uintptr_t>(context)!=context_identity_)
    return Status::FailedPrecondition("current CUDA context differs from Qwen INT4 owner");
  CUdevice device=0;result=cuCtxGetDevice(&device);
  if(result!=CUDA_SUCCESS)return cuda_driver_status(result,"cuCtxGetDevice");
  if(device!=owning_rank_)return Status::FailedPrecondition("current CUDA device differs from Qwen INT4 owner");
  return Status::Ok();
}


Status NvidiaQwenInt4ExecutionDriver::execute(
    const QwenInt4LmHeadBinding& binding,DriverStreamHandle stream) {
  if(stream==0)return Status::InvalidArgument("NVIDIA Qwen INT4 LM head requires explicit stream");
  const Status owner=require_current_owner();if(!owner.ok())return owner;
  if(binding.input().rank()!=2 || binding.input().dim(0)==0 ||
     binding.input().dim(0)>plans_.size())
    return Status::InvalidArgument("NVIDIA Qwen INT4 LM head row count is invalid");
  return plans_[binding.input().dim(0)-1]->execute(
      binding.input(),binding.weight(),binding.output(),
      workspace_,workspace_bytes_,reinterpret_cast<cudaStream_t>(stream));
}

}  // namespace pih
