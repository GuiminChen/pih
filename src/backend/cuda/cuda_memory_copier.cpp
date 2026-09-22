#include "pih/backend/cuda/cuda_memory_copier.h"

#include <cstddef>
#include <limits>
#include <utility>

#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

class DeviceGuard final {
 public:
  static Result<DeviceGuard> Create(std::int32_t target) {
    int previous = 0;
    auto status = cudaGetDevice(&previous);
    if (status != cudaSuccess) return cuda_status(status, "cudaGetDevice");
    if (previous != target) {
      status = cudaSetDevice(target);
      if (status != cudaSuccess) return cuda_status(status, "cudaSetDevice");
    }
    return DeviceGuard(previous, target);
  }

  ~DeviceGuard() {
    if (previous_ != target_) (void)cudaSetDevice(previous_);
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
  DeviceGuard(DeviceGuard&& other) noexcept
      : previous_(std::exchange(other.previous_, other.target_)),
        target_(other.target_) {}

 private:
  DeviceGuard(int previous, int target) : previous_(previous), target_(target) {}
  int previous_;
  int target_;
};

}  // namespace

Result<std::unique_ptr<CudaMemoryCopier>> CudaMemoryCopier::Create(
    std::int32_t device_index) {
  if (device_index < 0) {
    return Status::InvalidArgument("CUDA device index must be nonnegative");
  }
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) return cuda_status(status, "cudaGetDeviceCount");
  if (device_index >= count) {
    return Status::InvalidArgument("CUDA device index is out of range");
  }
  return std::unique_ptr<CudaMemoryCopier>(
      new CudaMemoryCopier(device_index));
}

Status CudaMemoryCopier::copy(void* destination, Device destination_device,
                              const void* source, Device source_device,
                              std::uint64_t bytes) {
  if (source_device != Device::Cpu() ||
      destination_device.type() != DeviceType::kCuda ||
      destination_device.index() != device_index_) {
    return Status::InvalidArgument(
        "CUDA weight copier requires CPU source and its exact CUDA destination");
  }
  if (bytes == 0) return Status::Ok();
  if (destination == nullptr || source == nullptr) {
    return Status::InvalidArgument("nonempty copy requires source and destination");
  }
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted("copy exceeds address space");
  }
  auto guard = DeviceGuard::Create(device_index_);
  if (!guard.ok()) return guard.status();
  return cuda_status(cudaMemcpy(destination, source, static_cast<std::size_t>(bytes),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(H2D)");
}

}  // namespace pih
