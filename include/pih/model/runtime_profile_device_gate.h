#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pih/model/runtime_profile_payload.h"

namespace pih {

struct RuntimeProfileDeviceObservation final {
  std::int32_t ordinal = -1;
  std::string name;
  std::uint32_t compute_major = 0;
  std::uint32_t compute_minor = 0;
  std::uint64_t total_memory_bytes = 0;
  std::array<std::byte, 16> uuid{};
};

class RuntimeProfileDeviceProbe {
 public:
  virtual ~RuntimeProfileDeviceProbe() = default;
  virtual Result<RuntimeProfileDeviceObservation> observe(
      std::int32_t ordinal) = 0;
};

class VerifiedRuntimeProfileDevices final {
 public:
  [[nodiscard]] RuntimeProfileGpuFamily gpu_family() const noexcept {
    return gpu_family_;
  }
  [[nodiscard]] std::span<const std::int32_t> ordinals() const noexcept {
    return ordinals_;
  }
  [[nodiscard]] const Sha256Digest& observation_root() const noexcept {
    return observation_root_;
  }

 private:
  friend Result<VerifiedRuntimeProfileDevices> verify_runtime_profile_devices(
      const VerifiedRuntimeProfile&, std::span<const std::int32_t>,
      RuntimeProfileDeviceProbe&);
  VerifiedRuntimeProfileDevices(RuntimeProfileGpuFamily gpu_family,
                                std::vector<std::int32_t> ordinals,
                                Sha256Digest observation_root) noexcept;
  RuntimeProfileGpuFamily gpu_family_ =
      RuntimeProfileGpuFamily::kRtx4090D24GiB;
  std::vector<std::int32_t> ordinals_;
  Sha256Digest observation_root_{};
};

Result<VerifiedRuntimeProfileDevices> verify_runtime_profile_devices(
    const VerifiedRuntimeProfile& profile,
    std::span<const std::int32_t> ordered_ordinals,
    RuntimeProfileDeviceProbe& probe);

}  // namespace pih
