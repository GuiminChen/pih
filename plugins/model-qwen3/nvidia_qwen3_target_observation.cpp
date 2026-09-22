#include "pih/model/nvidia_qwen3_target_observation.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <utility>

#include "pih/backend/cuda/cuda_status.h"

#if CUDA_VERSION < 13020
#error "PIH Qwen target observation requires CUDA Toolkit 13.2 or newer"
#endif

namespace pih {

Result<QwenTargetObservation> observe_nvidia_qwen_target(
    QwenNumericalRunIdentity identity) {
  int count = 0;
  cudaError_t result = cudaGetDeviceCount(&count);
  if (result != cudaSuccess) return cuda_status(result, "cudaGetDeviceCount");
  int ordinal = -1;
  result = cudaGetDevice(&ordinal);
  if (result != cudaSuccess) return cuda_status(result, "cudaGetDevice");
  cudaDeviceProp properties{};
  result = cudaGetDeviceProperties(&properties, ordinal);
  if (result != cudaSuccess) {
    return cuda_status(result, "cudaGetDeviceProperties");
  }
  int driver = 0;
  result = cudaDriverGetVersion(&driver);
  if (result != cudaSuccess) return cuda_status(result, "cudaDriverGetVersion");
  int runtime = 0;
  result = cudaRuntimeGetVersion(&runtime);
  if (result != cudaSuccess) return cuda_status(result, "cudaRuntimeGetVersion");

  QwenObservedDevice observed{};
  observed.visible_device_count = static_cast<std::uint32_t>(count);
  observed.current_ordinal = static_cast<std::uint32_t>(ordinal);
  observed.name = properties.name;
  observed.compute_major = static_cast<std::uint32_t>(properties.major);
  observed.compute_minor = static_cast<std::uint32_t>(properties.minor);
  observed.total_global_memory_bytes = properties.totalGlobalMem;
  static_assert(sizeof(properties.uuid.bytes) == 16);
  std::memcpy(observed.uuid.data(), properties.uuid.bytes, observed.uuid.size());
  observed.pci_domain = static_cast<std::uint32_t>(properties.pciDomainID);
  observed.pci_bus = static_cast<std::uint32_t>(properties.pciBusID);
  observed.pci_device = static_cast<std::uint32_t>(properties.pciDeviceID);
  observed.driver_version = static_cast<std::uint32_t>(driver);
  observed.runtime_version = static_cast<std::uint32_t>(runtime);
  return QwenTargetObservation::Create(std::move(identity), std::move(observed));
}

}  // namespace pih
