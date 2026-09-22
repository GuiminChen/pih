#include "pih/core/allocator.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>

namespace pih {
namespace {

bool valid_alignment(std::uint64_t alignment) {
  return alignment != 0 && (alignment & (alignment - 1)) == 0;
}

}  // namespace

Result<Allocation> CpuAllocator::allocate(std::uint64_t bytes,
                                          std::uint64_t alignment) {
  if (!valid_alignment(alignment)) {
    return Status::InvalidArgument("alignment must be a nonzero power of two");
  }
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted("allocation size exceeds addressable memory");
  }

  const auto actual_alignment =
      std::max<std::uint64_t>(alignment, alignof(std::max_align_t));
  void* data = nullptr;
  if (bytes != 0) {
    data = ::operator new(static_cast<std::size_t>(bytes),
                          std::align_val_t(actual_alignment), std::nothrow);
    if (data == nullptr) {
      return Status::ResourceExhausted("CPU allocation failed");
    }
  }
  const auto generation = next_generation_.fetch_add(1, std::memory_order_relaxed);
  if (generation == 0) {
    if (data != nullptr) {
      ::operator delete(data, std::align_val_t(actual_alignment));
    }
    return Status::ResourceExhausted("allocation generation exhausted");
  }
  return Allocation{data, bytes, actual_alignment, generation, Device::Cpu()};
}

void CpuAllocator::deallocate(Allocation allocation) noexcept {
  if (allocation.data != nullptr) {
    ::operator delete(allocation.data, std::align_val_t(allocation.alignment));
  }
}

}  // namespace pih
