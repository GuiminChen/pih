#include "pih/model/deepseek_boundary_credit_tracker.h"

#include <limits>

namespace pih {

Result<DeepSeekBoundaryCreditTracker>
DeepSeekBoundaryCreditTracker::Create(std::uint32_t credit_count) {
  if (credit_count != kCreditCount) {
    return Status::InvalidArgument(
        "DeepSeek boundary credit count must equal two");
  }
  return DeepSeekBoundaryCreditTracker{};
}

Result<DeepSeekBoundaryCreditHandle>
DeepSeekBoundaryCreditTracker::reserve(
    std::uint64_t pipeline_plan_sequence,
    std::uint64_t operation_ordinal) {
  if (poisoned_ || pipeline_plan_sequence == 0 || operation_ordinal == 0) {
    return Status::FailedPrecondition(
        "DeepSeek boundary credit tracker cannot reserve");
  }
  for (const auto& slot : slots_) {
    if (slot.state != DeepSeekBoundaryCreditState::kFree &&
        slot.plan_sequence == pipeline_plan_sequence &&
        slot.operation_ordinal == operation_ordinal) {
      return Status::FailedPrecondition(
          "DeepSeek boundary operation identity is already reserved");
    }
  }
  for (std::uint32_t index = 0; index < kCreditCount; ++index) {
    auto& slot = slots_[index];
    if (slot.state != DeepSeekBoundaryCreditState::kFree) continue;
    if (slot.generation == std::numeric_limits<std::uint64_t>::max()) {
      poisoned_ = true;
      return Status::ResourceExhausted(
          "DeepSeek boundary credit generation exhausted");
    }
    ++slot.generation;
    slot.plan_sequence = pipeline_plan_sequence;
    slot.operation_ordinal = operation_ordinal;
    slot.state = DeepSeekBoundaryCreditState::kPrepared;
    return DeepSeekBoundaryCreditHandle{index, slot.generation,
                                        pipeline_plan_sequence,
                                        operation_ordinal};
  }
  return Status::ResourceExhausted(
      "DeepSeek boundary receive credits are exhausted");
}

Status DeepSeekBoundaryCreditTracker::require(
    const DeepSeekBoundaryCreditHandle& handle,
    DeepSeekBoundaryCreditState expected) const {
  if (handle.credit_index >= kCreditCount ||
      handle.credit_generation == 0 || handle.pipeline_plan_sequence == 0 ||
      handle.operation_ordinal == 0) {
    return Status::InvalidArgument(
        "DeepSeek boundary credit handle is invalid");
  }
  const auto& slot = slots_[handle.credit_index];
  if (slot.state != expected || slot.generation != handle.credit_generation ||
      slot.plan_sequence != handle.pipeline_plan_sequence ||
      slot.operation_ordinal != handle.operation_ordinal) {
    return Status::FailedPrecondition(
        "DeepSeek boundary credit handle is stale or in wrong state");
  }
  return Status::Ok();
}

void DeepSeekBoundaryCreditTracker::clear(Slot& slot) noexcept {
  slot.state = DeepSeekBoundaryCreditState::kFree;
  slot.plan_sequence = 0;
  slot.operation_ordinal = 0;
}

Status DeepSeekBoundaryCreditTracker::abort_prepare(
    const DeepSeekBoundaryCreditHandle& handle) {
  const auto status = require(handle, DeepSeekBoundaryCreditState::kPrepared);
  if (!status.ok()) return status;
  clear(slots_[handle.credit_index]);
  return Status::Ok();
}

Status DeepSeekBoundaryCreditTracker::commit(
    const DeepSeekBoundaryCreditHandle& handle) {
  const auto status = require(handle, DeepSeekBoundaryCreditState::kPrepared);
  if (!status.ok()) return status;
  slots_[handle.credit_index].state = DeepSeekBoundaryCreditState::kCommitted;
  return Status::Ok();
}

Status DeepSeekBoundaryCreditTracker::complete_verified(
    const DeepSeekBoundaryCreditHandle& handle) {
  const auto status = require(handle, DeepSeekBoundaryCreditState::kCommitted);
  if (!status.ok()) return status;
  slots_[handle.credit_index].state =
      DeepSeekBoundaryCreditState::kCompleteVerified;
  return Status::Ok();
}

Status DeepSeekBoundaryCreditTracker::release(
    const DeepSeekBoundaryCreditHandle& handle) {
  const auto status = require(
      handle, DeepSeekBoundaryCreditState::kCompleteVerified);
  if (!status.ok()) return status;
  clear(slots_[handle.credit_index]);
  return Status::Ok();
}

Status DeepSeekBoundaryCreditTracker::mark_suspect(
    const DeepSeekBoundaryCreditHandle& handle) {
  const auto status = require(handle, DeepSeekBoundaryCreditState::kCommitted);
  if (!status.ok()) return status;
  slots_[handle.credit_index].state = DeepSeekBoundaryCreditState::kSuspect;
  poisoned_ = true;
  return Status::Ok();
}

}  // namespace pih
