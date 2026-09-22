#include "pih/backend/cuda/nvidia_physical_device_identity_probe.h"

#include <cuda.h>

#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/model/engine_physical_gpu_uuid_commitment.h"

namespace pih {

Result<Sha256Digest> nvidia_physical_device_identity(
    std::int32_t device_ordinal) {
  if (device_ordinal < 0)
    return Status::InvalidArgument("CUDA device ordinal is invalid");
  auto status = cuInit(0);
  if (status != CUDA_SUCCESS) return cuda_driver_status(status, "cuInit");
  CUdevice device = 0;
  status = cuDeviceGet(&device, device_ordinal);
  if (status != CUDA_SUCCESS) return cuda_driver_status(status, "cuDeviceGet");
  CUuuid uuid{};
  status = cuDeviceGetUuid_v2(&uuid, device);
  if (status != CUDA_SUCCESS)
    return cuda_driver_status(status, "cuDeviceGetUuid_v2");
  return engine_physical_gpu_uuid_commitment(
      std::as_bytes(std::span(uuid.bytes)));
}

}  // namespace pih
