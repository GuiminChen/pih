#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

#include "pih/core/allocator.h"

namespace pih {

class CudaAllocator final : public Allocator {
 public:
  static Result<std::unique_ptr<CudaAllocator>> Create(std::int32_t device_index);

  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override;
  Status release(Allocation allocation);
  void deallocate(Allocation allocation) noexcept override;

  [[nodiscard]] std::int32_t device_index() const noexcept { return device_index_; }

 private:
  explicit CudaAllocator(std::int32_t device_index) : device_index_(device_index) {}

  std::int32_t device_index_;
  std::atomic<std::uint64_t> next_generation_{1};
};

}  // namespace pih
