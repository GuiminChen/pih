#include "pih/backend/cuda/nvidia_typed_copy_driver.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <limits>

#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/backend/cuda/cuda_status.h"

#if CUDA_VERSION < 13020
#error "PIH typed copy driver requires CUDA Toolkit 13.2 or newer"
#endif

namespace pih {
namespace {

Result<cudaMemcpyKind> runtime_kind(CudaCopyKind kind) {
  switch (kind) {
    case CudaCopyKind::kHostToDevice:
      return cudaMemcpyHostToDevice;
    case CudaCopyKind::kDeviceToHost:
      return cudaMemcpyDeviceToHost;
    case CudaCopyKind::kDeviceToDevice:
      return cudaMemcpyDeviceToDevice;
  }
  return Status::InvalidArgument("unknown CUDA typed copy kind");
}

}  // namespace

Result<NvidiaTypedCopyDriver> NvidiaTypedCopyDriver::Create() {
  CUcontext context = nullptr;
  const CUresult result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuCtxGetCurrent");
  }
  if (context == nullptr) {
    return Status::FailedPrecondition(
        "NVIDIA typed copy driver requires a current context");
  }
  return NvidiaTypedCopyDriver(reinterpret_cast<std::uintptr_t>(context));
}

Status NvidiaTypedCopyDriver::require_current_context() const {
  CUcontext current = nullptr;
  const CUresult result = cuCtxGetCurrent(&current);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuCtxGetCurrent");
  }
  if (reinterpret_cast<std::uintptr_t>(current) != context_identity_) {
    return Status::FailedPrecondition(
        "current CUDA context differs from typed copy owner");
  }
  return Status::Ok();
}

Status NvidiaTypedCopyDriver::copy(CudaCopyKind kind,
                                   std::uintptr_t destination,
                                   std::uintptr_t source,
                                   std::uint64_t bytes,
                                   DriverStreamHandle stream) {
  if (destination == 0 || source == 0 || bytes == 0 || stream == 0) {
    return Status::InvalidArgument(
        "CUDA async copy requires nonempty pointers, bytes, and stream");
  }
  if (bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted("CUDA async copy exceeds address space");
  }
  auto selected_kind = runtime_kind(kind);
  if (!selected_kind.ok()) return selected_kind.status();
  const Status context = require_current_context();
  if (!context.ok()) return context;
  const Status clean_before = cuda_status(
      cudaPeekAtLastError(), "cudaPeekAtLastError before async copy");
  if (!clean_before.ok()) return clean_before;
  const cudaError_t result = cudaMemcpyAsync(
      reinterpret_cast<void*>(destination),
      reinterpret_cast<const void*>(source), static_cast<std::size_t>(bytes),
      selected_kind.value(), reinterpret_cast<cudaStream_t>(stream));
  const Status clean_after = cuda_status(
      cudaPeekAtLastError(), "cudaPeekAtLastError after async copy");
  if (result != cudaSuccess) return cuda_status(result, "cudaMemcpyAsync");
  return clean_after;
}

}  // namespace pih
