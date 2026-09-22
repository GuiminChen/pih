#include "pih/model/engine_gpu_process_inventory.h"

#include <algorithm>

namespace pih {

Result<EngineGpuProcessInventoryGate> EngineGpuProcessInventoryGate::Create(
    std::uint64_t engine_generation, Sha256Digest allocation_lease_digest,
    std::span<const std::uint64_t> sorted_physical_gpu_identities) {
  if (engine_generation == 0 || allocation_lease_digest == Sha256Digest{} ||
      sorted_physical_gpu_identities.empty() ||
      sorted_physical_gpu_identities.size() > 4)
    return Status::InvalidArgument("engine GPU process inventory is invalid");
  for (std::size_t index = 0;
       index < sorted_physical_gpu_identities.size(); ++index) {
    if (sorted_physical_gpu_identities[index] == 0 ||
        (index != 0 && sorted_physical_gpu_identities[index - 1] >=
                           sorted_physical_gpu_identities[index]))
      return Status::InvalidArgument(
          "engine GPU process inventory devices are not canonical");
  }
  return EngineGpuProcessInventoryGate(
      engine_generation, allocation_lease_digest,
      std::vector<std::uint64_t>(sorted_physical_gpu_identities.begin(),
                                 sorted_physical_gpu_identities.end()));
}

Result<EngineGpuProcessInventoryState>
EngineGpuProcessInventoryGate::accept(
    const EngineGpuProcessInventoryReceipt& receipt) {
  if (poisoned_)
    return Status::FailedPrecondition(
        "engine GPU process inventory is poisoned");
  if (receipt.engine_generation != generation_ ||
      receipt.allocation_lease_digest != lease_digest_ ||
      receipt.sample_identity == 0 ||
      receipt.sample_identity != last_sample_identity_ + 1 ||
      receipt.sample_completed_ns < receipt.sample_started_ns ||
      receipt.sample_started_ns < last_sample_completed_ns_ ||
      receipt.devices.size() != devices_.size()) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine GPU process inventory receipt drifted");
  }
  bool occupied = false;
  for (std::size_t index = 0; index < devices_.size(); ++index) {
    const auto& entry = receipt.devices[index];
    if (entry.physical_gpu_identity != devices_[index]) {
      poisoned_ = true;
      return Status::FailedPrecondition(
          "engine GPU process inventory coverage drifted");
    }
    for (std::size_t process = 0;
         process < entry.sorted_process_identities.size(); ++process) {
      if (entry.sorted_process_identities[process] == 0 ||
          (process != 0 && entry.sorted_process_identities[process - 1] >=
                               entry.sorted_process_identities[process])) {
        poisoned_ = true;
        return Status::FailedPrecondition(
            "engine GPU process inventory process set is not canonical");
      }
    }
    occupied = occupied || !entry.sorted_process_identities.empty();
  }
  last_sample_identity_ = receipt.sample_identity;
  last_sample_completed_ns_ = receipt.sample_completed_ns;
  state_ = !receipt.visibility_complete
               ? EngineGpuProcessInventoryState::kUnknown
               : occupied ? EngineGpuProcessInventoryState::kOccupied
                          : EngineGpuProcessInventoryState::kEmpty;
  return state_;
}

bool EngineGpuProcessInventoryGate::bound_to(
    std::uint64_t generation, const Sha256Digest& lease_digest) const noexcept {
  return generation == generation_ && lease_digest == lease_digest_;
}

}  // namespace pih
