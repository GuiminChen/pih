#include "pih/backend/cuda/nvidia_deepseek_dspark_embed_operations.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/backend/cuda/cuda_status.h"

namespace pih {
Result<NvidiaDeepSeekDsparkEmbedOperations>
NvidiaDeepSeekDsparkEmbedOperations::Create() {
  CUcontext context = nullptr;
  const auto result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS)
    return cuda_driver_status(result, "cuCtxGetCurrent DeepSeek DSpark embed");
  if (context == nullptr)
    return Status::FailedPrecondition(
        "DeepSeek DSpark embed requires a current CUDA context");
  return NvidiaDeepSeekDsparkEmbedOperations(
      reinterpret_cast<std::uint64_t>(context));
}
Status NvidiaDeepSeekDsparkEmbedOperations::require_context() const {
  CUcontext context = nullptr;
  const auto result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS)
    return cuda_driver_status(result, "cuCtxGetCurrent DeepSeek DSpark embed");
  return reinterpret_cast<std::uint64_t>(context) == context_identity_
      ? Status::Ok()
      : Status::FailedPrecondition(
            "DeepSeek DSpark embed CUDA context identity changed");
}
Status NvidiaDeepSeekDsparkEmbedOperations::validate_host_error(
    std::uint32_t* host_error) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (host_error == nullptr)
    return Status::InvalidArgument("DeepSeek DSpark embed host error is null");
  cudaPointerAttributes attributes{};
  const auto result = cudaPointerGetAttributes(&attributes, host_error);
  if (result != cudaSuccess)
    return cuda_status(result,
                       "cudaPointerGetAttributes DeepSeek DSpark embed error");
  return attributes.type == cudaMemoryTypeHost
      ? Status::Ok()
      : Status::FailedPrecondition(
            "DeepSeek DSpark embed error is not pinned host memory");
}
Status NvidiaDeepSeekDsparkEmbedOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (device == 0 || stream == 0)
    return Status::InvalidArgument(
        "DeepSeek DSpark embed error clear is invalid");
  return cuda_status(cudaMemsetAsync(
      reinterpret_cast<void*>(device), 0, sizeof(std::uint32_t),
      reinterpret_cast<cudaStream_t>(stream)),
      "cudaMemsetAsync DeepSeek DSpark embed error");
}
#define PIH_CONTEXT_LAUNCH(method, function, type) \
Status NvidiaDeepSeekDsparkEmbedOperations::method(type launch) { \
  auto status = require_context(); \
  return status.ok() ? function(launch) : status; \
}
PIH_CONTEXT_LAUNCH(quant, launch_deepseek_fp8_activation_quant,
                         DeepSeekFp8ActivationQuantLaunch)
PIH_CONTEXT_LAUNCH(gemm, launch_deepseek_fp8_gemm,
                         DeepSeekFp8GemmLaunch)
PIH_CONTEXT_LAUNCH(rms, launch_deepseek_rms_norm,
                         DeepSeekRmsNormLaunch)
PIH_CONTEXT_LAUNCH(draft_init, launch_deepseek_dspark_draft_init,
                         DeepSeekDsparkDraftInitLaunch)
#undef PIH_CONTEXT_LAUNCH
Status NvidiaDeepSeekDsparkEmbedOperations::copy_error_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (host == nullptr || device == 0 || stream == 0)
    return Status::InvalidArgument(
        "DeepSeek DSpark embed error copy is invalid");
  return cuda_status(cudaMemcpyAsync(
      host, reinterpret_cast<const void*>(device), sizeof(std::uint32_t),
      cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)),
      "cudaMemcpyAsync DeepSeek DSpark embed error D2H");
}
}  // namespace pih
