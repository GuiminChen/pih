#pragma once

#include <cstdint>

#include "pih/backend/cuda/pinned_placement_verifier.h"
#include "pih/core/result.h"

namespace pih {

class NvidiaLinuxNumaPolicyGuard final {
 public:
  static Result<NvidiaLinuxNumaPolicyGuard> Bind(std::int32_t numa_node);
  ~NvidiaLinuxNumaPolicyGuard();
  NvidiaLinuxNumaPolicyGuard(const NvidiaLinuxNumaPolicyGuard&) = delete;
  NvidiaLinuxNumaPolicyGuard& operator=(
      const NvidiaLinuxNumaPolicyGuard&) = delete;
  NvidiaLinuxNumaPolicyGuard(NvidiaLinuxNumaPolicyGuard&& other) noexcept;
  NvidiaLinuxNumaPolicyGuard& operator=(
      NvidiaLinuxNumaPolicyGuard&&) = delete;

  Status restore();

 private:
  explicit NvidiaLinuxNumaPolicyGuard(bool active) : active_(active) {}
  bool active_ = false;
};

class NvidiaLinuxPinnedPlacementVerifier final
    : public PinnedPlacementVerifier {
 public:
  static Result<std::int32_t> DeviceNumaNode(
      std::int32_t device_ordinal);
  Status verify(const void* data, std::uint64_t bytes,
                std::int32_t numa_node) override;
};

}  // namespace pih
