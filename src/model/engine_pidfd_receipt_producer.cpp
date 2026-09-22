#include "pih/model/engine_pidfd_receipt_producer.h"

#include <algorithm>
#include <set>

namespace pih {
namespace {

bool same_target(const EnginePidfdReapTarget& a,
                 const EnginePidfdReapTarget& b) noexcept {
  return a.process == b.process &&
         a.creator_process_identity == b.creator_process_identity &&
         a.adopted_reaper_identity == b.adopted_reaper_identity;
}

Status validate_authorized_target(
    const EnginePidfdReaperAuthority& authority,
    const EnginePidfdReapTarget& target) {
  if (target.process.process_identity == 0 ||
      target.process.pidfd_identity == 0 ||
      target.creator_process_identity == 0)
    return Status::InvalidArgument("engine pidfd reap target is invalid");
  const bool controller =
      target.process.role == EngineSupervisedProcessRole::kController &&
      target.process.rank == -1;
  const bool rank = target.process.role == EngineSupervisedProcessRole::kRank &&
                    target.process.rank >= 0 && target.process.rank < 4;
  if (!controller && !rank)
    return Status::InvalidArgument("engine pidfd reap target role is invalid");
  if (authority.mode == EnginePidfdReaperMode::kDirectParent) {
    if (target.creator_process_identity != authority.producer_process_identity ||
        target.adopted_reaper_identity != 0)
      return Status::FailedPrecondition(
          "engine pidfd target is not a direct child");
    return Status::Ok();
  }
  if (authority.mode != EnginePidfdReaperMode::kDesignatedSubreaper ||
      !authority.kernel_subreaper_enabled ||
      target.creator_process_identity == authority.producer_process_identity ||
      target.adopted_reaper_identity != authority.producer_process_identity)
    return Status::FailedPrecondition(
        "engine pidfd target has no designated subreaper authority");
  return Status::Ok();
}

}  // namespace

Result<EnginePidfdReceiptProducer> EnginePidfdReceiptProducer::Create(
    EnginePidfdReaperAuthority authority,
    std::span<const EnginePidfdReapTarget> targets,
    EnginePidfdReapOperations& operations) {
  if (authority.engine_generation == 0 ||
      authority.producer_process_identity == 0 || targets.empty() ||
      targets.size() > 5)
    return Status::InvalidArgument("engine pidfd receipt producer is invalid");
  std::set<std::uint64_t> processes;
  std::set<std::uint64_t> pidfds;
  for (const auto& target : targets) {
    const auto status = validate_authorized_target(authority, target);
    if (!status.ok()) return status;
    if (!processes.insert(target.process.process_identity).second ||
        !pidfds.insert(target.process.pidfd_identity).second)
      return Status::InvalidArgument(
          "engine pidfd receipt producer targets are duplicated");
  }
  return EnginePidfdReceiptProducer(
      authority, std::vector<EnginePidfdReapTarget>(targets.begin(),
                                                    targets.end()),
      operations);
}

Status EnginePidfdReceiptProducer::validate_target(
    const EnginePidfdReapTarget& target) const {
  return validate_authorized_target(authority_, target);
}

Result<std::optional<EnginePidfdReapedReceipt>>
EnginePidfdReceiptProducer::poll(const EnginePidfdReapTarget& target,
                                 std::uint64_t event_identity) {
  if (poisoned_)
    return Status::FailedPrecondition(
        "engine pidfd receipt producer is poisoned");
  if (event_identity == 0) {
    poisoned_ = true;
    return Status::InvalidArgument("engine pidfd reap event identity is invalid");
  }
  const auto status = validate_target(target);
  if (!status.ok()) { poisoned_ = true; return status; }
  const auto found = std::find_if(
      targets_.begin(), targets_.end(),
      [&](const auto& value) { return same_target(value, target); });
  if (found == targets_.end()) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine pidfd reap target is not authorized");
  }
  const auto index = static_cast<std::size_t>(found - targets_.begin());
  if (produced_[index]) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine pidfd reaped receipt was already produced");
  }
  auto observed = operations_->poll_and_reap(*found);
  if (!observed.ok()) { poisoned_ = true; return observed.status(); }
  if (!observed->has_value())
    return std::optional<EnginePidfdReapedReceipt>{};
  const auto& value = **observed;
  if (value.process_identity != found->process.process_identity ||
      value.pidfd_identity != found->process.pidfd_identity ||
      !value.exited || !value.reaped) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine pidfd kernel reap observation drifted");
  }
  produced_[index] = true;
  return std::optional<EnginePidfdReapedReceipt>{EnginePidfdReapedReceipt{
      authority_.engine_generation, event_identity, found->process, true, true}};
}

}  // namespace pih
