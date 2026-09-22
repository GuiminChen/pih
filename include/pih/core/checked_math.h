#pragma once

#include <cstdint>
#include <limits>

#include "pih/core/result.h"

namespace pih {

inline Result<std::uint64_t> checked_add_u64(std::uint64_t lhs, std::uint64_t rhs) {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
    return Status::ResourceExhausted("u64 addition overflow");
  }
  return lhs + rhs;
}

inline Result<std::uint64_t> checked_mul_u64(std::uint64_t lhs, std::uint64_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    return Status::ResourceExhausted("u64 multiplication overflow");
  }
  return lhs * rhs;
}

inline Result<std::uint64_t> checked_align_up_u64(std::uint64_t value,
                                                   std::uint64_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return Status::InvalidArgument("alignment must be a nonzero power of two");
  }
  const std::uint64_t mask = alignment - 1;
  auto sum = checked_add_u64(value, mask);
  if (!sum.ok()) {
    return sum.status();
  }
  return sum.value() & ~mask;
}

}  // namespace pih
