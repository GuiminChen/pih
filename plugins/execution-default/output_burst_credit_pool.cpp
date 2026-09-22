#include "pih/scheduler/output_burst_credit_pool.h"

#include <algorithm>
#include <limits>

namespace pih {

Result<OutputBurstCreditPool> OutputBurstCreditPool::Create(
    OutputBurstCreditLimits limits) {
  if (limits.credit_count == 0 || limits.maximum_records_qwen != 1 ||
      limits.maximum_records_deepseek != 5 ||
      limits.maximum_slots_per_credit == 0 ||
      limits.maximum_bytes_per_credit == 0) {
    return Status::InvalidArgument("output burst credit limits are invalid");
  }
  try {
    return OutputBurstCreditPool(limits);
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("output burst credit allocation failed");
  }
}

Result<OutputBurstLease> OutputBurstCreditPool::acquire(
    std::uint64_t plan, OutputPlanKind kind, std::uint32_t records,
    std::uint32_t slots, std::uint64_t bytes) {
  const auto record_limit = kind == OutputPlanKind::kQwen
      ? limits_.maximum_records_qwen
      : kind == OutputPlanKind::kDeepSeek
            ? limits_.maximum_records_deepseek : 0;
  if (plan == 0 || records > record_limit || slots == 0 ||
      slots > limits_.maximum_slots_per_credit || bytes == 0 ||
      bytes > limits_.maximum_bytes_per_credit) {
    return Status::InvalidArgument("output burst request exceeds its envelope");
  }
  if (std::any_of(credits_.begin(), credits_.end(), [plan](const Credit& credit) {
        return credit.state != OutputBurstState::kFree &&
               credit.plan_sequence == plan;
      })) {
    return Status::FailedPrecondition("plan already owns an output burst credit");
  }
  for (std::uint32_t i = 0; i < credits_.size(); ++i) {
    auto& credit = credits_[i];
    if (credit.state != OutputBurstState::kFree) continue;
    if (credit.generation == std::numeric_limits<std::uint64_t>::max()) {
      return Status::FailedPrecondition("output burst generation wrapped");
    }
    ++credit.generation;
    credit.state = OutputBurstState::kPrepared;
    credit.plan_sequence = plan;
    credit.records = records;
    credit.slots = slots;
    credit.bytes = bytes;
    return OutputBurstLease{i, credit.generation, plan, records, slots, bytes};
  }
  return Status::ResourceExhausted("output burst credits are exhausted");
}

Status OutputBurstCreditPool::validate(const OutputBurstLease& lease,
                                       OutputBurstState state) const {
  if (lease.credit_index >= credits_.size())
    return Status::FailedPrecondition("output burst lease is stale");
  const auto& credit = credits_[lease.credit_index];
  if (credit.state != state || credit.generation != lease.credit_generation ||
      credit.plan_sequence != lease.plan_sequence ||
      credit.records != lease.records || credit.slots != lease.slots ||
      credit.bytes != lease.bytes) {
    return Status::FailedPrecondition("output burst lease identity drifted");
  }
  return Status::Ok();
}

Status OutputBurstCreditPool::validate_commit(
    const OutputBurstLease& lease) const {
  return validate(lease, OutputBurstState::kPrepared);
}

Status OutputBurstCreditPool::validate_abort(
    const OutputBurstLease& lease) const {
  return validate(lease, OutputBurstState::kPrepared);
}

Status OutputBurstCreditPool::validate_transfer(
    const OutputBurstLease& lease) const {
  return validate(lease, OutputBurstState::kCommitted);
}

Status OutputBurstCreditPool::commit(const OutputBurstLease& lease) {
  auto status = validate_commit(lease);
  if (status.ok()) credits_[lease.credit_index].state = OutputBurstState::kCommitted;
  return status;
}
Status OutputBurstCreditPool::abort(const OutputBurstLease& lease) {
  auto status = validate_abort(lease);
  if (status.ok()) credits_[lease.credit_index].state = OutputBurstState::kFree;
  return status;
}
Status OutputBurstCreditPool::transfer(const OutputBurstLease& lease) {
  auto status = validate_transfer(lease);
  if (status.ok()) credits_[lease.credit_index].state = OutputBurstState::kTransferred;
  return status;
}
Status OutputBurstCreditPool::release(const OutputBurstLease& lease) {
  auto status = validate(lease, OutputBurstState::kTransferred);
  if (!status.ok()) return status;
  const auto generation = credits_[lease.credit_index].generation;
  credits_[lease.credit_index] = {};
  credits_[lease.credit_index].generation = generation;
  return Status::Ok();
}
std::uint32_t OutputBurstCreditPool::available() const noexcept {
  return static_cast<std::uint32_t>(std::count_if(
      credits_.begin(), credits_.end(), [](const Credit& credit) {
        return credit.state == OutputBurstState::kFree;
      }));
}

}  // namespace pih
