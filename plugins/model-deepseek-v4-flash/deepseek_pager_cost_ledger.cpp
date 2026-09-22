#include "pih/model/deepseek_pager_cost_ledger.h"

#include "pih/core/canonical_hash.h"
#include "pih/core/checked_math.h"

namespace pih {

Result<DeepSeekPagerCostLedger> DeepSeekPagerCostLedger::Create(
    std::uint32_t maximum_entries) {
  if (maximum_entries == 0 || maximum_entries > kMaximumEntries) {
    return Status::InvalidArgument(
        "DeepSeek pager cost ledger capacity is invalid");
  }
  DeepSeekPagerCostLedger result;
  result.maximum_entries_ = maximum_entries;
  result.entries_.reserve(maximum_entries);
  Status status = result.recompute_identity();
  if (!status.ok()) return status;
  return result;
}

Result<std::uint64_t> DeepSeekPagerCostLedger::debit(
    DeepSeekPagerCostKey key, DeepSeekExpertDemandDisposition disposition,
    std::uint64_t observed_cost_ns) {
  if (key.request_id == 0 || key.request_generation == 0 ||
      key.plan_sequence == 0 || key.layer >= 43 || key.expert >= 256 ||
      disposition > DeepSeekExpertDemandDisposition::kNewMiss) {
    return Status::InvalidArgument(
        "DeepSeek pager cost identity is invalid");
  }
  if (disposition != DeepSeekExpertDemandDisposition::kNewMiss) {
    if (observed_cost_ns != 0) {
      return Status::InvalidArgument(
          "DeepSeek pager hit or join cannot carry miss cost");
    }
    return std::uint64_t{0};
  }
  if (observed_cost_ns == 0) {
    return Status::InvalidArgument(
        "DeepSeek observed pager miss cost must be nonzero");
  }
  for (const auto& entry : entries_) {
    if (entry.key == key) {
      if (entry.observed_cost_ns != observed_cost_ns) {
        return Status::FailedPrecondition(
            "DeepSeek pager cost replay drifted");
      }
      return std::uint64_t{0};
    }
  }
  if (entries_.size() == maximum_entries_) {
    return Status::ResourceExhausted(
        "DeepSeek pager cost ledger is full");
  }
  auto total = checked_add_u64(total_debit_ns_, observed_cost_ns);
  if (!total.ok()) return total.status();
  entries_.push_back({key, observed_cost_ns});
  total_debit_ns_ = *total;
  Status status = recompute_identity();
  if (!status.ok()) {
    total_debit_ns_ -= observed_cost_ns;
    entries_.pop_back();
    return status;
  }
  return observed_cost_ns;
}

Status DeepSeekPagerCostLedger::recompute_identity() {
  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek-pager-cost-ledger:v1",
      static_cast<std::uint32_t>(2 + entries_.size() * 6));
  if (!hash.ok()) return hash.status();
  Status status = hash->add_u32(1, maximum_entries_);
  if (status.ok()) status = hash->add_u64(2, total_debit_ns_);
  std::uint16_t field = 3;
  for (const auto& entry : entries_) {
    if (status.ok()) status = hash->add_u64(field++, entry.key.request_id);
    if (status.ok()) {
      status = hash->add_u64(field++, entry.key.request_generation);
    }
    if (status.ok()) status = hash->add_u64(field++, entry.key.plan_sequence);
    if (status.ok()) status = hash->add_u32(field++, entry.key.layer);
    if (status.ok()) status = hash->add_u32(field++, entry.key.expert);
    if (status.ok()) status = hash->add_u64(field++, entry.observed_cost_ns);
  }
  if (!status.ok()) return status;
  auto identity = hash->finalize();
  if (!identity.ok()) return identity.status();
  identity_ = *identity;
  return Status::Ok();
}

}  // namespace pih
