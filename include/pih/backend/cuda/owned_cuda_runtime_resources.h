#pragma once

#include <memory>

#include "pih/backend/cuda/cuda_runtime_resources.h"

namespace pih {

class OwnedCudaRuntimeResources final {
 public:
  static Result<OwnedCudaRuntimeResources> Create(
      std::int32_t device_ordinal, std::uint32_t rank,
      std::uint64_t worker_generation, std::uint32_t context_flags,
      std::unique_ptr<CudaRuntimeResourceDriver> driver);

  OwnedCudaRuntimeResources(const OwnedCudaRuntimeResources&) = delete;
  OwnedCudaRuntimeResources& operator=(const OwnedCudaRuntimeResources&) = delete;
  OwnedCudaRuntimeResources(OwnedCudaRuntimeResources&&) noexcept = default;
  OwnedCudaRuntimeResources& operator=(OwnedCudaRuntimeResources&&) = delete;

  [[nodiscard]] const CudaRuntimeResourceIdentity& identity() const noexcept {
    return resources_->identity();
  }
  [[nodiscard]] CudaRuntimeResourceDriver& driver() noexcept {
    return *driver_;
  }

 private:
  OwnedCudaRuntimeResources() = default;
  // resources_ is destroyed before the driver it borrows.
  std::unique_ptr<CudaRuntimeResourceDriver> driver_;
  std::unique_ptr<CudaRuntimeResources> resources_;
};

}  // namespace pih
