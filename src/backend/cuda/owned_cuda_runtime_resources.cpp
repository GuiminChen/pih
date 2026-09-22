#include "pih/backend/cuda/owned_cuda_runtime_resources.h"

#include <utility>

namespace pih {

Result<OwnedCudaRuntimeResources> OwnedCudaRuntimeResources::Create(
    std::int32_t device_ordinal, std::uint32_t rank,
    std::uint64_t worker_generation, std::uint32_t context_flags,
    std::unique_ptr<CudaRuntimeResourceDriver> driver) {
  if (driver == nullptr) {
    return Status::InvalidArgument("CUDA runtime resource driver is required");
  }
  auto resources = CudaRuntimeResources::Create(
      device_ordinal, rank, worker_generation, context_flags, *driver);
  if (!resources.ok()) return resources.status();
  // Allocate the owned wrapper while the borrowed driver is still held by the
  // local unique_ptr. If allocation throws, resources can therefore unwind
  // through a live driver instead of retaining a dangling driver pointer.
  auto owned_resources =
      std::make_unique<CudaRuntimeResources>(std::move(*resources));
  OwnedCudaRuntimeResources result;
  result.driver_ = std::move(driver);
  result.resources_ = std::move(owned_resources);
  return result;
}

}  // namespace pih
