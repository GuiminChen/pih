#include "pih/backend/cuda/nvidia_pinned_host_allocator.h"

#include <limits>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

#if CUDA_VERSION < 13020
#error "PIH pinned host allocator requires CUDA Toolkit 13.2 or newer"
#endif

namespace pih {

Result<std::unique_ptr<NvidiaPinnedHostAllocator>>
NvidiaPinnedHostAllocator::Create() {
  return std::unique_ptr<NvidiaPinnedHostAllocator>(
      new NvidiaPinnedHostAllocator());
}

Result<Allocation> NvidiaPinnedHostAllocator::allocate(
    std::uint64_t bytes, std::uint64_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
      alignment > 256) {
    return Status::InvalidArgument(
        "pinned host alignment must be a power of two at most 256");
  }
  if (bytes > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted(
        "pinned host allocation exceeds addressable memory");
  }
  void* data = nullptr;
  if (bytes != 0) {
    const auto status = cudaHostAlloc(
        &data, static_cast<std::size_t>(bytes), cudaHostAllocDefault);
    if (status != cudaSuccess) return cuda_status(status, "cudaHostAlloc");
  }
  const auto generation =
      next_generation_.fetch_add(1, std::memory_order_relaxed);
  if (generation == 0) {
    if (data != nullptr) (void)cudaFreeHost(data);
    return Status::ResourceExhausted(
        "pinned host allocation generation exhausted");
  }
  if (data != nullptr &&
      reinterpret_cast<std::uintptr_t>(data) % alignment != 0) {
    (void)cudaFreeHost(data);
    return Status::Internal(
        "CUDA pinned host allocation did not meet alignment");
  }
  return Allocation{data, bytes, 256, generation, Device::Cpu()};
}

void NvidiaPinnedHostAllocator::deallocate(Allocation allocation) noexcept {
  if (allocation.data != nullptr) (void)cudaFreeHost(allocation.data);
}

}  // namespace pih
