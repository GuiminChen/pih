#pragma once

#include "pih/model/engine_gpu_process_inventory.h"

namespace pih {

class EngineGpuProcessInventoryProvider {
 public:
  virtual ~EngineGpuProcessInventoryProvider() = default;
  virtual Result<std::vector<EngineGpuProcessInventoryEntry>> collect(
      std::span<const std::uint64_t> sorted_physical_gpu_identities) = 0;
};

class EngineGpuProcessInventoryCompiler final {
 public:
  static Result<EngineGpuProcessInventoryCompiler> Create(
      std::uint64_t engine_generation, Sha256Digest allocation_lease_digest,
      std::span<const std::uint64_t> sorted_physical_gpu_identities,
      EngineGpuProcessInventoryProvider& provider);
  Result<EngineGpuProcessInventoryReceipt> capture(
      std::uint64_t sample_started_ns,
      std::uint64_t sample_completed_ns);

 private:
  EngineGpuProcessInventoryCompiler(
      std::uint64_t generation, Sha256Digest lease_digest,
      std::vector<std::uint64_t> devices,
      EngineGpuProcessInventoryProvider& provider,
      EngineGpuProcessInventoryGate validator) noexcept
      : generation_(generation), lease_digest_(lease_digest),
        devices_(std::move(devices)), provider_(&provider),
        validator_(std::move(validator)) {}
  std::uint64_t generation_ = 0;
  Sha256Digest lease_digest_{};
  std::vector<std::uint64_t> devices_;
  EngineGpuProcessInventoryProvider* provider_ = nullptr;
  EngineGpuProcessInventoryGate validator_;
  std::uint64_t next_sample_identity_ = 1;
  std::uint64_t last_sample_completed_ns_ = 0;
  bool poisoned_ = false;
};

}  // namespace pih
