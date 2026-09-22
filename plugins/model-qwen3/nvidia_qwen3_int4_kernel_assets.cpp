#include "pih/model/nvidia_qwen3_int4_kernel_assets.h"
#include <cuda.h>
#include "pih/backend/cuda/cuda_driver_status.h"
namespace pih{
Result<QwenInt4KernelBundle>NvidiaQwenInt4KernelAssets::Load(
 KernelModuleDriver& driver,const std::filesystem::path& root,std::int32_t ordinal,
 std::uint64_t maximum){
 if(root.empty()||ordinal<0||maximum==0)return Status::InvalidArgument("NVIDIA Qwen INT4 kernel identity is invalid");
 CUcontext context=nullptr;CUresult result=cuCtxGetCurrent(&context);
 if(result!=CUDA_SUCCESS)return cuda_driver_status(result,"cuCtxGetCurrent");
 if(context==nullptr)return Status::FailedPrecondition("NVIDIA Qwen INT4 kernels require current context");
 CUdevice device=0;result=cuCtxGetDevice(&device);
 if(result!=CUDA_SUCCESS)return cuda_driver_status(result,"cuCtxGetDevice");
 if(device!=ordinal)return Status::FailedPrecondition("current CUDA device differs from kernel owner");
 int major=0,minor=0;result=cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device);
 if(result==CUDA_SUCCESS)result=cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device);
 if(result!=CUDA_SUCCESS)return cuda_driver_status(result,"cuDeviceGetAttribute");
 const int sm=major*10+minor;if(sm!=89&&sm!=90)return Status::Unavailable("Qwen INT4 requires SM89 or SM90");
 const auto dir=root/("sm_"+std::to_string(sm));
 return QwenInt4KernelBundle::Load(driver,dir/"qwen_bf16_primitives.cubin.json",
  dir/"qwen_bf16_primitives.cubin",static_cast<std::uint32_t>(major),
  static_cast<std::uint32_t>(minor),maximum);
}
Result<QwenBf16KernelBundle>NvidiaQwenInt4KernelAssets::LoadPacked(
 KernelModuleDriver& driver,const std::filesystem::path& root,
 std::int32_t ordinal,std::uint64_t maximum){
 if(root.empty()||ordinal<0||maximum==0)return Status::InvalidArgument(
  "NVIDIA packed Qwen INT4 kernel identity is invalid");
 CUcontext context=nullptr;CUresult result=cuCtxGetCurrent(&context);
 if(result!=CUDA_SUCCESS)return cuda_driver_status(result,"cuCtxGetCurrent");
 if(context==nullptr)return Status::FailedPrecondition(
  "NVIDIA packed Qwen INT4 kernels require current context");
 CUdevice device=0;result=cuCtxGetDevice(&device);
 if(result!=CUDA_SUCCESS)return cuda_driver_status(result,"cuCtxGetDevice");
 if(device!=ordinal)return Status::FailedPrecondition(
  "current CUDA device differs from packed kernel owner");
 int major=0,minor=0;
 result=cuDeviceGetAttribute(&major,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device);
 if(result==CUDA_SUCCESS)result=cuDeviceGetAttribute(&minor,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device);
 if(result!=CUDA_SUCCESS)return cuda_driver_status(result,"cuDeviceGetAttribute");
 const int sm=major*10+minor;if(sm!=89&&sm!=90)return Status::Unavailable(
  "packed Qwen INT4 requires SM89 or SM90");
 const auto dir=root/("sm_"+std::to_string(sm));
 return QwenBf16KernelBundle::Load(driver,
  dir/"qwen_bf16_primitives.cubin.json",
  dir/"qwen_bf16_primitives.cubin",static_cast<std::uint32_t>(major),
  static_cast<std::uint32_t>(minor),maximum);
}}
