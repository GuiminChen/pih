#include "pih/backend/cuda/nvidia_deepseek_physical_device_identity_probe.h"

#include <cuda.h>

#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/model/deepseek_gpu_uuid_commitment.h"

namespace pih {

Result<NvidiaDeepSeekPhysicalDeviceIdentityProbe>
NvidiaDeepSeekPhysicalDeviceIdentityProbe::Create(
    std::int32_t startup_device_ordinal) {
  if (startup_device_ordinal < 0)
    return Status::InvalidArgument("DeepSeek startup CUDA ordinal is invalid");
  return NvidiaDeepSeekPhysicalDeviceIdentityProbe(startup_device_ordinal);
}

Result<Sha256Digest>
NvidiaDeepSeekPhysicalDeviceIdentityProbe::
current_physical_device_uuid_commitment() {
  auto status = cuInit(0);
  if (status != CUDA_SUCCESS) return cuda_driver_status(status, "cuInit");
  CUdevice device = 0;
  status = cuDeviceGet(&device, ordinal_);
  if (status != CUDA_SUCCESS) return cuda_driver_status(status, "cuDeviceGet");
  CUuuid uuid{};
  status = cuDeviceGetUuid_v2(&uuid, device);
  if (status != CUDA_SUCCESS)
    return cuda_driver_status(status, "cuDeviceGetUuid_v2");
  return deepseek_gpu_uuid_commitment(std::as_bytes(std::span(uuid.bytes)));
}

}  // namespace pih
