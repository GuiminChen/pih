#include "pih/backend/cuda/cuda_status.h"

#include <string>

namespace pih {

Status cuda_status(cudaError_t error, std::string_view operation) {
  if (error == cudaSuccess) {
    return Status::Ok();
  }
  std::string message(operation);
  message.append(": ");
  message.append(cudaGetErrorName(error));
  message.append(" (");
  message.append(cudaGetErrorString(error));
  message.push_back(')');

  switch (error) {
    case cudaErrorMemoryAllocation:
      return Status::ResourceExhausted(std::move(message));
    case cudaErrorInvalidDevice:
    case cudaErrorInvalidValue:
      return Status::InvalidArgument(std::move(message));
    case cudaErrorNoDevice:
    case cudaErrorDevicesUnavailable:
    case cudaErrorInsufficientDriver:
      return Status::Unavailable(std::move(message));
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchFailure:
    case cudaErrorAssert:
      return Status::Internal(std::move(message));
    default:
      return Status::Internal(std::move(message));
  }
}

}  // namespace pih
