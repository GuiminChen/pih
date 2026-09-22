#include "pih/backend/cuda/nvidia_cuda_context_activator.h"

#include <cuda.h>

#include "pih/backend/cuda/cuda_driver_status.h"

namespace pih {

Status NvidiaCudaContextActivator::activate(
    std::uintptr_t context_identity) {
  if (context_identity == 0) {
    return Status::InvalidArgument(
        "NVIDIA CUDA context activation identity is null");
  }
  return cuda_driver_status(
      cuCtxSetCurrent(reinterpret_cast<CUcontext>(context_identity)),
      "cuCtxSetCurrent NCCL rank");
}

}  // namespace pih
