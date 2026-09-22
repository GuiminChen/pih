#include "pih/backend/cuda/verified_kernel_launcher.h"

#include <limits>

#include "pih/core/checked_math.h"

namespace pih {

Result<KernelLaunchGeometry> KernelLaunchGeometry::Create(
    std::uint32_t grid_x, std::uint32_t grid_y, std::uint32_t grid_z,
    std::uint32_t block_x, std::uint32_t block_y, std::uint32_t block_z,
    std::uint32_t dynamic_shared_bytes) {
  if (grid_x == 0 || grid_y == 0 || grid_z == 0 || block_x == 0 ||
      block_y == 0 || block_z == 0) {
    return Status::InvalidArgument("kernel launch dimensions must be nonzero");
  }
  if (grid_x > kMaximumGridX || grid_y > kMaximumGridYZ ||
      grid_z > kMaximumGridYZ) {
    return Status::InvalidArgument("kernel grid exceeds portable CUDA bounds");
  }
  if (block_x > kMaximumBlockX || block_y > kMaximumBlockY ||
      block_z > kMaximumBlockZ) {
    return Status::InvalidArgument("kernel block dimension exceeds CUDA bounds");
  }
  auto block_xy = checked_mul_u64(block_x, block_y);
  if (!block_xy.ok()) return block_xy.status();
  auto block_threads = checked_mul_u64(block_xy.value(), block_z);
  if (!block_threads.ok()) return block_threads.status();
  if (block_threads.value() > kMaximumBlockThreads) {
    return Status::InvalidArgument("kernel block exceeds 1024 threads");
  }
  return KernelLaunchGeometry(grid_x, grid_y, grid_z, block_x, block_y,
                              block_z, dynamic_shared_bytes);
}

Status submit_verified_kernel(KernelLaunchDriver& driver,
                              const ResolvedKernelFunction& function,
                              const KernelLaunchGeometry& geometry,
                              DriverStreamHandle stream,
                              KernelArgumentPacket& arguments) {
  if (function.handle == 0) {
    return Status::InvalidArgument("verified kernel function is null");
  }
  if (stream == 0) {
    return Status::InvalidArgument("verified kernel launch forbids default stream");
  }
  if (function.logical_id != arguments.logical_id() ||
      function.parameter_abi_sha256 != arguments.parameter_abi_sha256()) {
    return Status::InvalidArgument(
        "kernel function and argument packet identities differ");
  }
  auto parameters = arguments.ready_kernel_params();
  if (!parameters.ok()) return parameters.status();
  return driver.launch(function.handle, geometry, stream, parameters.value());
}

}  // namespace pih
