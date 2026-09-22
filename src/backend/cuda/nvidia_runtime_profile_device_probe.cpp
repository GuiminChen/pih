#include "pih/backend/cuda/nvidia_runtime_profile_device_probe.h"

#include <cuda.h>

#include <cstring>

#include "pih/backend/cuda/cuda_driver_status.h"

namespace pih {

Result<RuntimeProfileDeviceObservation>
NvidiaRuntimeProfileDeviceProbe::observe(std::int32_t ordinal) {
  if (ordinal < 0) {
    return Status::InvalidArgument("NVIDIA device ordinal is invalid");
  }
  auto result = cuInit(0);
  if (result != CUDA_SUCCESS) return cuda_driver_status(result, "cuInit");
  int count = 0;
  result = cuDeviceGetCount(&count);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuDeviceGetCount");
  }
  if (ordinal >= count) {
    return Status::InvalidArgument("NVIDIA device ordinal is not visible");
  }
  CUdevice device{};
  result = cuDeviceGet(&device, ordinal);
  if (result != CUDA_SUCCESS) return cuda_driver_status(result, "cuDeviceGet");

  RuntimeProfileDeviceObservation observation{};
  observation.ordinal = ordinal;
  char name[256]{};
  result = cuDeviceGetName(name, sizeof(name), device);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuDeviceGetName");
  }
  observation.name = name;
  int major = 0;
  result = cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuDeviceGetAttribute major");
  }
  int minor = 0;
  result = cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuDeviceGetAttribute minor");
  }
  if (major < 0 || minor < 0) {
    return Status::Internal("NVIDIA compute capability is invalid");
  }
  observation.compute_major = static_cast<std::uint32_t>(major);
  observation.compute_minor = static_cast<std::uint32_t>(minor);
  std::size_t total_memory = 0;
  result = cuDeviceTotalMem(&total_memory, device);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuDeviceTotalMem");
  }
  observation.total_memory_bytes = total_memory;
  CUuuid uuid{};
  result = cuDeviceGetUuid(&uuid, device);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuDeviceGetUuid");
  }
  static_assert(sizeof(uuid.bytes) == 16);
  std::memcpy(observation.uuid.data(), uuid.bytes, observation.uuid.size());
  return observation;
}

}  // namespace pih
