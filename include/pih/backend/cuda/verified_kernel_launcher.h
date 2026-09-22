#pragma once

#include <cstdint>

#include "pih/backend/cuda/kernel_argument_packet.h"
#include "pih/backend/cuda/verified_kernel_module.h"

namespace pih {

using DriverStreamHandle = std::uintptr_t;

class KernelLaunchGeometry final {
 public:
  static constexpr std::uint32_t kMaximumGridX = 0x7fffffffU;
  static constexpr std::uint32_t kMaximumGridYZ = 65535;
  static constexpr std::uint32_t kMaximumBlockThreads = 1024;
  static constexpr std::uint32_t kMaximumBlockX = 1024;
  static constexpr std::uint32_t kMaximumBlockY = 1024;
  static constexpr std::uint32_t kMaximumBlockZ = 64;

  static Result<KernelLaunchGeometry> Create(
      std::uint32_t grid_x, std::uint32_t grid_y, std::uint32_t grid_z,
      std::uint32_t block_x, std::uint32_t block_y, std::uint32_t block_z,
      std::uint32_t dynamic_shared_bytes);

  [[nodiscard]] std::uint32_t grid_x() const noexcept { return grid_x_; }
  [[nodiscard]] std::uint32_t grid_y() const noexcept { return grid_y_; }
  [[nodiscard]] std::uint32_t grid_z() const noexcept { return grid_z_; }
  [[nodiscard]] std::uint32_t block_x() const noexcept { return block_x_; }
  [[nodiscard]] std::uint32_t block_y() const noexcept { return block_y_; }
  [[nodiscard]] std::uint32_t block_z() const noexcept { return block_z_; }
  [[nodiscard]] std::uint32_t dynamic_shared_bytes() const noexcept {
    return dynamic_shared_bytes_;
  }

 private:
  KernelLaunchGeometry(std::uint32_t grid_x, std::uint32_t grid_y,
                       std::uint32_t grid_z, std::uint32_t block_x,
                       std::uint32_t block_y, std::uint32_t block_z,
                       std::uint32_t dynamic_shared_bytes)
      : grid_x_(grid_x),
        grid_y_(grid_y),
        grid_z_(grid_z),
        block_x_(block_x),
        block_y_(block_y),
        block_z_(block_z),
        dynamic_shared_bytes_(dynamic_shared_bytes) {}

  std::uint32_t grid_x_;
  std::uint32_t grid_y_;
  std::uint32_t grid_z_;
  std::uint32_t block_x_;
  std::uint32_t block_y_;
  std::uint32_t block_z_;
  std::uint32_t dynamic_shared_bytes_;
};

class KernelLaunchDriver {
 public:
  virtual ~KernelLaunchDriver() = default;
  virtual Status launch(DriverFunctionHandle function,
                        const KernelLaunchGeometry& geometry,
                        DriverStreamHandle stream, void** kernel_params) = 0;
};

Status submit_verified_kernel(KernelLaunchDriver& driver,
                              const ResolvedKernelFunction& function,
                              const KernelLaunchGeometry& geometry,
                              DriverStreamHandle stream,
                              KernelArgumentPacket& arguments);

}  // namespace pih
