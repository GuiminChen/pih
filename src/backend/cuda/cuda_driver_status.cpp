#include "pih/backend/cuda/cuda_driver_status.h"

#include <string>
#include <utility>

namespace pih {

Status cuda_driver_status(CUresult result, std::string_view operation) {
  if (result == CUDA_SUCCESS) return Status::Ok();
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  std::string message(operation);
  message.append(": ");
  message.append(name == nullptr ? "unknown CUDA driver error" : name);
  if (description != nullptr) {
    message.append(" (");
    message.append(description);
    message.push_back(')');
  }
  switch (result) {
    case CUDA_ERROR_OUT_OF_MEMORY:
      return Status::ResourceExhausted(std::move(message));
    case CUDA_ERROR_INVALID_VALUE:
    case CUDA_ERROR_INVALID_IMAGE:
    case CUDA_ERROR_INVALID_HANDLE:
      return Status::InvalidArgument(std::move(message));
    case CUDA_ERROR_NOT_INITIALIZED:
    case CUDA_ERROR_DEINITIALIZED:
    case CUDA_ERROR_NO_DEVICE:
    case CUDA_ERROR_NO_BINARY_FOR_GPU:
    case CUDA_ERROR_SYSTEM_NOT_READY:
    case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
      return Status::Unavailable(std::move(message));
    default:
      return Status::Internal(std::move(message));
  }
}

}  // namespace pih
