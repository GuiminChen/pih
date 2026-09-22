#include "pih/backend/cuda/nvidia_deepseek_dspark_moe_stage_operations.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/backend/cuda/cuda_status.h"

namespace pih {

Result<NvidiaDeepSeekDsparkMoeStageOperations>
NvidiaDeepSeekDsparkMoeStageOperations::Create() {
  CUcontext context = nullptr;
  const auto result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuCtxGetCurrent DeepSeek DSpark MoE");
  }
  if (context == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark MoE requires a current CUDA context");
  }
  return NvidiaDeepSeekDsparkMoeStageOperations(
      reinterpret_cast<std::uint64_t>(context));
}

Status NvidiaDeepSeekDsparkMoeStageOperations::require_context() const {
  CUcontext context = nullptr;
  const auto result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuCtxGetCurrent DeepSeek DSpark MoE");
  }
  return reinterpret_cast<std::uint64_t>(context) == context_identity_
      ? Status::Ok()
      : Status::FailedPrecondition(
            "DeepSeek DSpark MoE CUDA context identity changed");
}

Status NvidiaDeepSeekDsparkMoeStageOperations::validate_host_error(
    std::uint32_t* host_error) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (host_error == nullptr) {
    return Status::InvalidArgument("DeepSeek DSpark MoE host error is null");
  }
  cudaPointerAttributes attributes{};
  const auto result = cudaPointerGetAttributes(&attributes, host_error);
  if (result != cudaSuccess) {
    return cuda_status(result,
                       "cudaPointerGetAttributes DeepSeek DSpark MoE error");
  }
  return attributes.type == cudaMemoryTypeHost
      ? Status::Ok()
      : Status::FailedPrecondition(
            "DeepSeek DSpark MoE error is not pinned host memory");
}

Status NvidiaDeepSeekDsparkMoeStageOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  return zero_bytes_async(device, sizeof(std::uint32_t), stream);
}

Status NvidiaDeepSeekDsparkMoeStageOperations::zero_bytes_async(
    std::uintptr_t device, std::uint64_t bytes, std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (device == 0 || bytes == 0 || stream == 0 || bytes > SIZE_MAX) {
    return Status::InvalidArgument(
        "DeepSeek DSpark MoE memset identity is invalid");
  }
  return cuda_status(
      cudaMemsetAsync(reinterpret_cast<void*>(device), 0,
                      static_cast<std::size_t>(bytes),
                      reinterpret_cast<cudaStream_t>(stream)),
      "cudaMemsetAsync DeepSeek DSpark MoE");
}

#define PIH_DSPARK_MOE_LAUNCH(method, launch_function, type) \
  Status NvidiaDeepSeekDsparkMoeStageOperations::method(type launch) { \
    auto status = require_context();                                  \
    return status.ok() ? launch_function(launch) : status;            \
  }

PIH_DSPARK_MOE_LAUNCH(
    mhc_pre, launch_deepseek_mhc_pre, DeepSeekMhcPreLaunch)
PIH_DSPARK_MOE_LAUNCH(
    router_gemm, launch_deepseek_router_bf16_gemm,
    DeepSeekRouterBf16GemmLaunch)
PIH_DSPARK_MOE_LAUNCH(
    quant, launch_deepseek_fp8_activation_quant,
    DeepSeekFp8ActivationQuantLaunch)
PIH_DSPARK_MOE_LAUNCH(
    fp4_gemm, launch_deepseek_fp4_gemm, DeepSeekFp4GemmLaunch)
PIH_DSPARK_MOE_LAUNCH(
    shared_swiglu, launch_deepseek_shared_expert_swiglu,
    DeepSeekSharedExpertSwiGluLaunch)
PIH_DSPARK_MOE_LAUNCH(
    finalize, launch_deepseek_expert_finalize,
    DeepSeekExpertFinalizeLaunch)
PIH_DSPARK_MOE_LAUNCH(
    mhc_post, launch_deepseek_mhc_post, DeepSeekMhcPostLaunch)

#undef PIH_DSPARK_MOE_LAUNCH

Status NvidiaDeepSeekDsparkMoeStageOperations::copy_d2h_async(
    void* host, std::uintptr_t device, std::uint64_t bytes,
    std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (host == nullptr || device == 0 || bytes == 0 ||
      bytes > SIZE_MAX || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark MoE D2H identity is invalid");
  }
  return cuda_status(
      cudaMemcpyAsync(host, reinterpret_cast<const void*>(device),
                      static_cast<std::size_t>(bytes), cudaMemcpyDeviceToHost,
                      reinterpret_cast<cudaStream_t>(stream)),
      "cudaMemcpyAsync DeepSeek DSpark MoE D2H");
}

Status NvidiaDeepSeekDsparkMoeStageOperations::record_event(
    std::uintptr_t event, std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (event == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark MoE event identity is invalid");
  }
  return cuda_status(
      cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                      reinterpret_cast<cudaStream_t>(stream)),
      "cudaEventRecord DeepSeek DSpark MoE");
}

Result<DeepSeekExpertAsyncStatus>
NvidiaDeepSeekDsparkMoeStageOperations::query_event(std::uintptr_t event) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (event == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark MoE event is null");
  }
  const auto result = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event));
  if (result == cudaSuccess) return DeepSeekExpertAsyncStatus::kSuccess;
  if (result == cudaErrorNotReady) {
    return DeepSeekExpertAsyncStatus::kInProgress;
  }
  return cuda_status(result, "cudaEventQuery DeepSeek DSpark MoE");
}

Status NvidiaDeepSeekDsparkMoeStageOperations::synchronize_stream(
    std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark MoE stream is null");
  }
  return cuda_status(
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
      "cudaStreamSynchronize DeepSeek DSpark MoE cancel");
}

}  // namespace pih
