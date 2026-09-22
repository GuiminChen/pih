#pragma once

#include <span>
#include <vector>

#include "pih/core/sha256.h"

namespace pih {

struct DeepSeekPhysicalDeviceRegistryEntry final {
  std::uint32_t rank = 0;
  std::int32_t startup_device_ordinal = -1;
  std::uint64_t registry_identity = 0;
  Sha256Digest uuid_commitment{};
};

class DeepSeekPhysicalDeviceRegistry final {
 public:
  static Result<DeepSeekPhysicalDeviceRegistry> Create(
      std::span<const DeepSeekPhysicalDeviceRegistryEntry> entries);
  [[nodiscard]] const DeepSeekPhysicalDeviceRegistryEntry* rank(
      std::uint32_t rank) const noexcept;
  [[nodiscard]] const DeepSeekPhysicalDeviceRegistryEntry* identity(
      std::uint64_t registry_identity) const noexcept;
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return static_cast<std::uint32_t>(entries_.size());
  }

 private:
  explicit DeepSeekPhysicalDeviceRegistry(
      std::vector<DeepSeekPhysicalDeviceRegistryEntry> entries) noexcept
      : entries_(std::move(entries)) {}
  std::vector<DeepSeekPhysicalDeviceRegistryEntry> entries_;
};

}  // namespace pih
