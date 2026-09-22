#include "pih/model/engine_gpu_process_inventory_compiler.h"

#include <limits>

namespace pih {

Result<EngineGpuProcessInventoryCompiler>
EngineGpuProcessInventoryCompiler::Create(
    std::uint64_t engine_generation, Sha256Digest allocation_lease_digest,
    std::span<const std::uint64_t> sorted_physical_gpu_identities,
    EngineGpuProcessInventoryProvider& provider) {
  auto validator = EngineGpuProcessInventoryGate::Create(
      engine_generation, allocation_lease_digest,
      sorted_physical_gpu_identities);
  if (!validator.ok()) return validator.status();
  return EngineGpuProcessInventoryCompiler(
      engine_generation, allocation_lease_digest,
      std::vector<std::uint64_t>(sorted_physical_gpu_identities.begin(),
                                 sorted_physical_gpu_identities.end()),
      provider, std::move(*validator));
}

Result<EngineGpuProcessInventoryReceipt>
EngineGpuProcessInventoryCompiler::capture(
    std::uint64_t sample_started_ns, std::uint64_t sample_completed_ns) {
  if (poisoned_)
    return Status::FailedPrecondition(
        "engine GPU process inventory compiler is poisoned");
  if (sample_completed_ns < sample_started_ns ||
      sample_started_ns < last_sample_completed_ns_ ||
      next_sample_identity_ == std::numeric_limits<std::uint64_t>::max()) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine GPU process inventory sample time drifted");
  }
  auto collected = provider_->collect(devices_);
  bool visible = true;
  std::vector<EngineGpuProcessInventoryEntry> entries;
  if (!collected.ok()) {
    if (collected.status().code() != StatusCode::kUnavailable) {
      poisoned_ = true;
      return collected.status();
    }
    visible = false;
    entries.reserve(devices_.size());
    for (const auto device : devices_) entries.push_back({device, {}});
  } else {
    entries = std::move(*collected);
  }
  EngineGpuProcessInventoryReceipt receipt{
      generation_, lease_digest_, next_sample_identity_, sample_started_ns,
      sample_completed_ns, visible, std::move(entries)};
  auto validated = validator_.accept(receipt);
  if (!validated.ok()) {
    poisoned_ = true;
    return validated.status();
  }
  ++next_sample_identity_;
  last_sample_completed_ns_ = sample_completed_ns;
  return receipt;
}

}  // namespace pih
