#include "pih/backend/cuda/elementwise_launch.h"

#include <limits>

#include "pih/core/checked_math.h"

namespace pih {

Result<ElementwiseLaunch> ElementwiseLaunch::Create(
    std::uint64_t elements, std::uint32_t block_threads) {
  if (elements == 0) {
    return Status::InvalidArgument("elementwise launch requires nonzero elements");
  }
  if (block_threads < kWarpThreads || block_threads > kMaximumBlockThreads ||
      block_threads % kWarpThreads != 0) {
    return Status::InvalidArgument(
        "elementwise block size must be a warp multiple from 32 through 1024");
  }
  auto numerator = checked_add_u64(elements, block_threads - 1U);
  if (!numerator.ok()) return numerator.status();
  const std::uint64_t blocks = numerator.value() / block_threads;
  if (blocks > kMaximumGridBlocks) {
    return Status::ResourceExhausted("elementwise launch exceeds CUDA grid.x");
  }
  return ElementwiseLaunch(elements, block_threads,
                           static_cast<std::uint32_t>(blocks));
}

}  // namespace pih
