#include "pih/backend/cuda/nvidia_runtime_resource_driver.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/backend/cuda/cuda_status.h"

#if CUDA_VERSION < 13020
#error "PIH runtime resources require CUDA Toolkit 13.2 or newer"
#endif

namespace pih {

Status NvidiaRuntimeResourceDriver::require_current(
    std::uintptr_t context) const {
  CUcontext current = nullptr;
  const CUresult result = cuCtxGetCurrent(&current);
  if (result != CUDA_SUCCESS) return cuda_driver_status(result, "cuCtxGetCurrent");
  if (reinterpret_cast<std::uintptr_t>(current) != context) {
    return Status::FailedPrecondition("current CUDA context differs from owner");
  }
  return Status::Ok();
}

Result<std::uintptr_t> NvidiaRuntimeResourceDriver::retain_primary_context(
    std::int32_t device_ordinal, std::uint32_t context_flags) {
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) return cuda_driver_status(result, "cuInit");
  CUdevice device{};
  result = cuDeviceGet(&device, device_ordinal);
  if (result != CUDA_SUCCESS) return cuda_driver_status(result, "cuDeviceGet");
  int active = 0;
  unsigned int effective_flags = 0;
  result = cuDevicePrimaryCtxGetState(device, &effective_flags, &active);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuDevicePrimaryCtxGetState");
  }
  if (active != 0 && effective_flags != context_flags) {
    return Status::FailedPrecondition(
        "active CUDA primary context has incompatible flags");
  }
  if (active == 0) {
    result = cuDevicePrimaryCtxSetFlags(device, context_flags);
    if (result != CUDA_SUCCESS) {
      return cuda_driver_status(result, "cuDevicePrimaryCtxSetFlags");
    }
  }
  CUcontext context = nullptr;
  result = cuDevicePrimaryCtxRetain(&context, device);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuDevicePrimaryCtxRetain");
  }
  result = cuCtxSetCurrent(context);
  if (result != CUDA_SUCCESS) {
    (void)cuDevicePrimaryCtxRelease(device);
    return cuda_driver_status(result, "cuCtxSetCurrent");
  }
  return reinterpret_cast<std::uintptr_t>(context);
}

Status NvidiaRuntimeResourceDriver::bind_runtime(
    std::int32_t device_ordinal, std::uintptr_t context) {
  Status current = require_current(context);
  if (!current.ok()) return current;
  cudaError_t result = cudaSetDevice(device_ordinal);
  if (result != cudaSuccess) return cuda_status(result, "cudaSetDevice");
  result = cudaFree(nullptr);
  if (result != cudaSuccess) return cuda_status(result, "cudaFree(nullptr)");
  return require_current(context);
}

Result<DriverStreamHandle>
NvidiaRuntimeResourceDriver::create_nonblocking_stream(
    std::uintptr_t context) {
  const Status current = require_current(context);
  if (!current.ok()) return current;
  CUstream stream = nullptr;
  const CUresult result = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
  if (result != CUDA_SUCCESS) return cuda_driver_status(result, "cuStreamCreate");
  return reinterpret_cast<DriverStreamHandle>(stream);
}

Result<DriverEventHandle>
NvidiaRuntimeResourceDriver::create_disable_timing_event(
    std::uintptr_t context) {
  const Status current = require_current(context);
  if (!current.ok()) return current;
  CUevent event = nullptr;
  const CUresult result = cuEventCreate(&event, CU_EVENT_DISABLE_TIMING);
  if (result != CUDA_SUCCESS) return cuda_driver_status(result, "cuEventCreate");
  return reinterpret_cast<DriverEventHandle>(event);
}

void NvidiaRuntimeResourceDriver::destroy_event(
    DriverEventHandle event) noexcept {
  if (event != 0) (void)cuEventDestroy(reinterpret_cast<CUevent>(event));
}

void NvidiaRuntimeResourceDriver::destroy_stream(
    DriverStreamHandle stream) noexcept {
  if (stream != 0) (void)cuStreamDestroy(reinterpret_cast<CUstream>(stream));
}

void NvidiaRuntimeResourceDriver::release_primary_context(
    std::int32_t device_ordinal, std::uintptr_t context) noexcept {
  CUdevice device{};
  if (cuDeviceGet(&device, device_ordinal) != CUDA_SUCCESS) return;
  CUcontext current = nullptr;
  if (cuCtxGetCurrent(&current) == CUDA_SUCCESS &&
      reinterpret_cast<std::uintptr_t>(current) == context) {
    (void)cuCtxSetCurrent(nullptr);
  }
  (void)cuDevicePrimaryCtxRelease(device);
}

}  // namespace pih
