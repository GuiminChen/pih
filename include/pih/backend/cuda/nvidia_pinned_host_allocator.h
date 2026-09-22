#pragma once

#include <atomic>
#include <memory>

#include "pih/backend/cuda/registered_pinned_allocator.h"

namespace pih {

class NvidiaPinnedHostAllocator final : public RegisteredPinnedAllocator {
 public:
  static Result<std::unique_ptr<NvidiaPinnedHostAllocator>> Create();

  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override;
  void deallocate(Allocation allocation) noexcept override;

 private:
  NvidiaPinnedHostAllocator() = default;

  std::atomic<std::uint64_t> next_generation_{1};
};

}  // namespace pih
