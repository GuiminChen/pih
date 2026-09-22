#include "pih/model/deepseek_stage_mapped_inventory.h"

#include <map>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Result<std::map<std::string, std::uint64_t, std::less<>>> ValidatePlan(
    const DeepSeekRankMappingPlan& plan, std::uint64_t* planned_bytes) {
  if (planned_bytes == nullptr || plan.intervals.empty()) {
    return Status::InvalidArgument(
        "DeepSeek stage mapping inventory input is empty");
  }
  std::map<std::string, std::uint64_t, std::less<>> required;
  *planned_bytes = 0;
  const DeepSeekMappedInterval* previous = nullptr;
  for (const auto& interval : plan.intervals) {
    if (interval.shard_name.empty() ||
        interval.file_begin >= interval.file_end) {
      return Status::InvalidArgument(
          "DeepSeek stage mapped interval is invalid");
    }
    if (previous != nullptr &&
        (interval.shard_name < previous->shard_name ||
         (interval.shard_name == previous->shard_name &&
          interval.file_begin <= previous->file_end))) {
      return Status::InvalidArgument(
          "DeepSeek stage mapped intervals are not canonical");
    }
    previous = &interval;
    required.emplace(interval.shard_name, 0);
    auto total = checked_add_u64(*planned_bytes,
                                 interval.file_end - interval.file_begin);
    if (!total.ok()) return total.status();
    *planned_bytes = *total;
  }
  if (*planned_bytes != plan.mapped_interval_bytes ||
      plan.shards.size() != required.size()) {
    return Status::InvalidArgument(
        "DeepSeek rank mapping ledger differs from intervals");
  }
  for (const auto& shard : plan.shards) {
    auto found = required.find(shard.shard_name);
    if (found == required.end() || shard.file_bytes == 0 ||
        found->second != 0) {
      return Status::InvalidArgument(
          "DeepSeek rank shard receipt is invalid or duplicated");
    }
    found->second = shard.file_bytes;
  }
  return required;
}

}  // namespace

DeepSeekStageMappedInventory& DeepSeekStageMappedInventory::operator=(
    DeepSeekStageMappedInventory&& other) noexcept {
  if (this == &other) return *this;

  // Capability mappings must be released while their shared leases are still
  // live. Default member-wise move assignment overwrites the lease vector
  // first because of declaration order, which can strand provider leases.
  capability_mappings_.clear();
  capability_leases_.clear();

  rank_ = other.rank_;
  mapped_interval_bytes_ = other.mapped_interval_bytes_;
  capability_leases_ = std::move(other.capability_leases_);
  capability_mappings_ = std::move(other.capability_mappings_);
  other.rank_ = 0;
  other.mapped_interval_bytes_ = 0;
  return *this;
}


Result<DeepSeekStageMappedInventory>
DeepSeekStageMappedInventory::CreateCapabilityBacked(
    const DeepSeekRankMappingPlan& plan,
    std::vector<DeepSeekCapabilityShardLease> leases) {
  if (leases.empty()) {
    return Status::InvalidArgument(
        "DeepSeek capability shard lease set is empty");
  }
  std::uint64_t planned_bytes = 0;
  auto required_result = ValidatePlan(plan, &planned_bytes);
  if (!required_result.ok()) return required_result.status();
  auto required = std::move(*required_result);
  std::map<std::string, DeepSeekVerifiedArtifactLease*, std::less<>> by_name;
  for (const auto& value : leases) {
    if (value.shard_name.empty() || value.lease == nullptr ||
        !by_name.emplace(value.shard_name, value.lease.get()).second) {
      return Status::InvalidArgument(
          "DeepSeek capability shard lease is invalid or duplicated");
    }
  }
  if (by_name.size() != required.size()) {
    return Status::InvalidArgument(
        "DeepSeek capability shard set differs from rank plan");
  }
  for (const auto& [shard_name, file_bytes] : required) {
    const auto found = by_name.find(shard_name);
    if (found == by_name.end() || found->second->file_bytes() != file_bytes) {
      return Status::InvalidArgument(
          "DeepSeek capability shard identity differs from rank receipt");
    }
  }

  DeepSeekStageMappedInventory result;
  result.rank_ = plan.rank;
  result.mapped_interval_bytes_ = planned_bytes;
  result.capability_mappings_.reserve(leases.size());
  for (const auto& value : leases) {
    auto mapping = value.lease->map_read_only(0, value.lease->file_bytes());
    if (!mapping.ok()) return mapping.status();
    result.capability_mappings_.push_back(
        {value.shard_name, std::move(*mapping)});
  }
  result.capability_leases_ = std::move(leases);
  return result;
}

Result<std::span<const std::byte>> DeepSeekStageMappedInventory::bytes(
    std::string_view shard_name, std::uint64_t file_offset,
    std::uint64_t bytes) const {
  auto requested_end = checked_add_u64(file_offset, bytes);
  if (!requested_end.ok() || bytes == 0) {
    return Status::InvalidArgument(
        "DeepSeek stage mapped byte request is invalid");
  }
  for (const auto& mapping : capability_mappings_) {
    auto mapping_end = checked_add_u64(mapping.range.file_offset(),
                                       mapping.range.bytes().size());
    if (!mapping_end.ok()) return mapping_end.status();
    if (mapping.shard_name == shard_name &&
        file_offset >= mapping.range.file_offset() &&
        *requested_end <= *mapping_end) {
      return mapping.range.slice(file_offset - mapping.range.file_offset(),
                                 bytes);
    }
  }
  return Status::InvalidArgument(
      "DeepSeek tensor bytes are outside the stage mapping inventory");
}

}  // namespace pih
