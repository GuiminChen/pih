#pragma once

#include <atomic>
#include <cstdint>

#include "pih/core/device.h"

namespace pih {

struct Allocation final {
  void* data = nullptr;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 0;
  std::uint64_t generation = 0;
  Device device = Device::Cpu();
};

class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual Result<Allocation> allocate(std::uint64_t bytes,
                                      std::uint64_t alignment) = 0;
  virtual void deallocate(Allocation allocation) noexcept = 0;
};

class CpuAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override;
  void deallocate(Allocation allocation) noexcept override;

 private:
  std::atomic<std::uint64_t> next_generation_{1};
};

}  // namespace pih
