#include "pih/backend/cuda/cuda_allocator.h"

#include <limits>
#include <memory>
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
    if (status != cudaSuccess) {
      return cuda_status(status, "cudaGetDevice");
    }
    if (previous != target) {
      status = cudaSetDevice(target);
      if (status != cudaSuccess) {
        return cuda_status(status, "cudaSetDevice");
      }
    }
    return DeviceGuard(previous, target);
  }

  ~DeviceGuard() {
    if (previous_ != target_) {
      (void)cudaSetDevice(previous_);
    }
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

bool valid_alignment(std::uint64_t alignment) {
  return alignment != 0 && (alignment & (alignment - 1)) == 0 && alignment <= 256;
}

}  // namespace

Result<std::unique_ptr<CudaAllocator>> CudaAllocator::Create(
    std::int32_t device_index) {
  if (device_index < 0) {
    return Status::InvalidArgument("CUDA device index must be nonnegative");
  }
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    return cuda_status(status, "cudaGetDeviceCount");
  }
  if (device_index >= count) {
    return Status::InvalidArgument("CUDA device index is out of range");
  }
  return std::unique_ptr<CudaAllocator>(new CudaAllocator(device_index));
}

Result<Allocation> CudaAllocator::allocate(std::uint64_t bytes,
                                           std::uint64_t alignment) {
  if (!valid_alignment(alignment)) {
    return Status::InvalidArgument(
        "CUDA alignment must be a power of two no greater than 256");
  }
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted("allocation size exceeds addressable memory");
  }
  auto guard = DeviceGuard::Create(device_index_);
  if (!guard.ok()) {
    return guard.status();
  }
  void* data = nullptr;
  if (bytes != 0) {
    const auto status = cudaMalloc(&data, static_cast<std::size_t>(bytes));
    if (status != cudaSuccess) {
      return cuda_status(status, "cudaMalloc");
    }
  }
  const auto generation = next_generation_.fetch_add(1, std::memory_order_relaxed);
  if (generation == 0) {
    if (data != nullptr) {
      (void)cudaFree(data);
    }
    return Status::ResourceExhausted("allocation generation exhausted");
  }
  return Allocation{data, bytes, 256, generation,
                    Device::Create(DeviceType::kCuda, device_index_).value()};
}

Status CudaAllocator::release(Allocation allocation) {
  if (allocation.data == nullptr) {
    return Status::Ok();
  }
  auto guard = DeviceGuard::Create(device_index_);
  if (!guard.ok()) {
    return guard.status();
  }
  const auto status = cudaFree(allocation.data);
  return status == cudaSuccess ? Status::Ok()
                               : cuda_status(status, "cudaFree");
}

void CudaAllocator::deallocate(Allocation allocation) noexcept {
  try {
    (void)release(std::move(allocation));
  } catch (...) {
    // The compatibility Allocator interface cannot report cleanup failure.
  }
}

}  // namespace pih
