#include "pih/model/engine_pidfd_reaping_ledger.h"

#include <algorithm>
#include <set>

namespace pih {

Result<EnginePidfdReapingLedger> EnginePidfdReapingLedger::Create(
    std::uint64_t engine_generation,
    std::span<const EngineSupervisedProcessIdentity> processes) {
  if (engine_generation == 0 || processes.size() < 2 || processes.size() > 5)
    return Status::InvalidArgument("engine pidfd reaping manifest is invalid");
  std::set<std::uint64_t> process_identities;
  std::set<std::uint64_t> pidfd_identities;
  for (std::size_t index = 0; index < processes.size(); ++index) {
    const auto& value = processes[index];
    const bool controller =
        index == 0 && value.role == EngineSupervisedProcessRole::kController &&
        value.rank == -1;
    const bool rank =
        index != 0 && value.role == EngineSupervisedProcessRole::kRank &&
        value.rank == static_cast<std::int32_t>(index - 1);
    if ((!controller && !rank) || value.process_identity == 0 ||
        value.pidfd_identity == 0 ||
        !process_identities.insert(value.process_identity).second ||
        !pidfd_identities.insert(value.pidfd_identity).second)
      return Status::InvalidArgument(
          "engine pidfd reaping process identity is invalid");
  }
  return EnginePidfdReapingLedger(
      engine_generation,
      std::vector<EngineSupervisedProcessIdentity>(processes.begin(),
                                                   processes.end()));
}

Status EnginePidfdReapingLedger::poison() noexcept {
  poisoned_ = true;
  return Status::FailedPrecondition("engine pidfd reaping receipt drifted");
}

Status EnginePidfdReapingLedger::accept(
    const EnginePidfdReapedReceipt& receipt) {
  if (poisoned_) return poison();
  if (receipt.engine_generation != generation_ ||
      receipt.event_identity == 0 ||
      receipt.event_identity != last_event_identity_ + 1 ||
      !receipt.exited || !receipt.reaped)
    return poison();
  const auto found = std::find(processes_.begin(), processes_.end(),
                               receipt.process);
  if (found == processes_.end()) return poison();
  const auto index = static_cast<std::size_t>(found - processes_.begin());
  if (reaped_[index]) return poison();
  reaped_[index] = true;
  last_event_identity_ = receipt.event_identity;
  return Status::Ok();
}

bool EnginePidfdReapingLedger::complete() const noexcept {
  return !poisoned_ &&
         std::all_of(reaped_.begin(), reaped_.end(), [](bool value) {
           return value;
         });
}

std::size_t EnginePidfdReapingLedger::remaining() const noexcept {
  if (poisoned_) return processes_.size();
  return static_cast<std::size_t>(
      std::count(reaped_.begin(), reaped_.end(), false));
}

}  // namespace pih
