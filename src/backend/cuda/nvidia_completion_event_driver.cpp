#include "pih/backend/cuda/nvidia_completion_event_driver.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/backend/cuda/cuda_status.h"

#if CUDA_VERSION < 13020
#error "PIH completion event driver requires CUDA Toolkit 13.2 or newer"
#endif

namespace pih {

Result<NvidiaCompletionEventDriver> NvidiaCompletionEventDriver::Create() {
  CUcontext context = nullptr;
  const CUresult result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuCtxGetCurrent");
  }
  if (context == nullptr) {
    return Status::FailedPrecondition(
        "NVIDIA completion event driver requires a current context");
  }
  return NvidiaCompletionEventDriver(
      reinterpret_cast<std::uintptr_t>(context));
}

Status NvidiaCompletionEventDriver::require_current_context() const {
  CUcontext current = nullptr;
  const CUresult result = cuCtxGetCurrent(&current);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuCtxGetCurrent");
  }
  if (reinterpret_cast<std::uintptr_t>(current) != context_identity_) {
    return Status::FailedPrecondition(
        "current CUDA context differs from completion event owner");
  }
  return Status::Ok();
}

Status NvidiaCompletionEventDriver::record(DriverEventHandle event,
                                           DriverStreamHandle stream) {
  if (event == 0 || stream == 0) {
    return Status::InvalidArgument(
        "CUDA completion record requires event and non-default stream");
  }
  const Status context = require_current_context();
  if (!context.ok()) return context;
  return cuda_driver_status(
      cuEventRecord(reinterpret_cast<CUevent>(event),
                    reinterpret_cast<CUstream>(stream)),
      "cuEventRecord");
}

Result<CudaEventQueryResult> NvidiaCompletionEventDriver::query(
    DriverEventHandle event) {
  if (event == 0) {
    return Status::InvalidArgument("CUDA completion query event is null");
  }
  const Status context = require_current_context();
  if (!context.ok()) return context;
  const CUresult result = cuEventQuery(reinterpret_cast<CUevent>(event));
  if (result == CUDA_SUCCESS) return CudaEventQueryResult::kSuccess;
  if (result == CUDA_ERROR_NOT_READY) return CudaEventQueryResult::kNotReady;
  return cuda_driver_status(result, "cuEventQuery");
}

Status NvidiaCompletionEventDriver::require_clean_last_error() {
  const Status context = require_current_context();
  if (!context.ok()) return context;
  return cuda_status(cudaPeekAtLastError(), "cudaPeekAtLastError");
}

}  // namespace pih
