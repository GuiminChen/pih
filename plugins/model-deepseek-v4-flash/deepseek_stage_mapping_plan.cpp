#include "pih/model/deepseek_stage_mapping_plan.h"

#include <algorithm>
#include <limits>
#include <map>

#include "pih/core/checked_math.h"

namespace pih { namespace {

struct ShardView final {
  std::uint64_t file_bytes = 0;
  const SafetensorsHeader* header = nullptr;
};

Result<std::uint64_t> page_end(std::uint64_t value,
                               std::uint64_t page_bytes,
                               std::uint64_t file_bytes) {
  auto adjusted = checked_add_u64(value, page_bytes - 1);
  if (!adjusted.ok()) return adjusted.status();
  const auto aligned = (*adjusted / page_bytes) * page_bytes;
  return std::min(aligned, file_bytes);
}

}  // namespace

Result<DeepSeekStageMappingPlan> DeepSeekStageMappingPlan::Create(
    const DeepSeekPipelinePlan& pipeline,
    std::span<const SafetensorsShardBinding> bindings,
    std::span<const DeepSeekShardHeaderView> headers,
    std::uint64_t page_bytes) {
  if (page_bytes != kProductionPageBytes || headers.empty()) {
    return Status::InvalidArgument(
        "DeepSeek stage mapping page geometry or headers are invalid");
  }
  auto ownership = DeepSeekTensorOwnershipPlan::Create(pipeline, bindings);
  if (!ownership.ok()) return ownership.status();

  return BindTargetManifest(*ownership, headers, page_bytes);
}

Result<DeepSeekStageMappingPlan>
DeepSeekStageMappingPlan::BindTargetManifest(
    const DeepSeekTensorOwnershipPlan& ownership,
    std::span<const DeepSeekShardHeaderView> headers,
    std::uint64_t page_bytes) {
  if (ownership.world_size() == 0 || ownership.world_size() > 4) {
    return Status::InvalidArgument(
        "DeepSeek target mapping ownership world is invalid");
  }
  if (page_bytes != kProductionPageBytes || headers.empty()) {
    return Status::InvalidArgument(
        "DeepSeek target mapping page geometry or headers are invalid");
  }

  std::map<std::string, ShardView, std::less<>> shards;
  std::uint64_t header_tensor_count = 0;
  for (const auto& view : headers) {
    if (view.shard_name.empty() || view.header == nullptr) {
      return Status::InvalidArgument("DeepSeek shard header view is invalid");
    }
    auto prefix_bytes = checked_add_u64(8, view.header->header_bytes());
    if (!prefix_bytes.ok()) return prefix_bytes.status();
    auto expected_file_bytes =
        checked_add_u64(*prefix_bytes, view.header->data_bytes());
    if (!expected_file_bytes.ok()) return expected_file_bytes.status();
    if (view.file_bytes != *expected_file_bytes) {
      return Status::InvalidArgument(
          "DeepSeek shard header file size receipt differs");
    }
    if (!shards.emplace(std::string(view.shard_name),
                        ShardView{view.file_bytes, view.header}).second) {
      return Status::InvalidArgument("DeepSeek shard header is duplicated");
    }
    auto count = checked_add_u64(header_tensor_count,
                                 view.header->tensors().size());
    if (!count.ok()) return count.status();
    header_tensor_count = *count;
  }
  if (header_tensor_count != ownership.records().size()) {
    return Status::InvalidArgument(
        "DeepSeek index and shard header tensor counts differ");
  }

  DeepSeekStageMappingPlan result;
  result.ranks_.resize(ownership.world_size());
  for (std::uint32_t rank = 0; rank < ownership.world_size(); ++rank) {
    result.ranks_[rank].rank = rank;
  }
  for (const auto& owned : ownership.records()) {
    const auto shard = shards.find(owned.shard_name);
    if (shard == shards.end()) {
      return Status::InvalidArgument("DeepSeek index names an unknown shard");
    }
    const auto* tensor = shard->second.header->tensor(owned.tensor_name);
    if (tensor == nullptr || tensor->file_begin >= tensor->file_end ||
        tensor->file_end > shard->second.file_bytes) {
      return Status::InvalidArgument(
          "DeepSeek index and shard header tensor binding differ");
    }
    if (owned.owner_rank == DeepSeekTensorOwnership::kExcludedRank) {
      ++result.excluded_tensor_count_;
      continue;
    }
    auto& rank = result.ranks_[owned.owner_rank];
    ++rank.owned_tensor_count;
    auto logical = checked_add_u64(rank.logical_tensor_bytes,
                                   tensor->file_end - tensor->file_begin);
    if (!logical.ok()) return logical.status();
    rank.logical_tensor_bytes = *logical;
    auto aligned_end = page_end(tensor->file_end, page_bytes,
                                shard->second.file_bytes);
    if (!aligned_end.ok()) return aligned_end.status();
    rank.intervals.push_back({owned.shard_name,
                              (tensor->file_begin / page_bytes) * page_bytes,
                              *aligned_end});
  }

  for (auto& rank : result.ranks_) {
    std::ranges::sort(rank.intervals, [](const auto& left, const auto& right) {
      if (left.shard_name != right.shard_name)
        return left.shard_name < right.shard_name;
      if (left.file_begin != right.file_begin)
        return left.file_begin < right.file_begin;
      return left.file_end < right.file_end;
    });
    std::vector<DeepSeekMappedInterval> merged;
    for (auto& interval : rank.intervals) {
      if (!merged.empty() &&
          merged.back().shard_name == interval.shard_name &&
          interval.file_begin <= merged.back().file_end) {
        merged.back().file_end =
            std::max(merged.back().file_end, interval.file_end);
      } else {
        merged.push_back(std::move(interval));
      }
    }
    rank.intervals = std::move(merged);
    for (const auto& interval : rank.intervals) {
      if (rank.shards.empty() ||
          rank.shards.back().shard_name != interval.shard_name) {
        const auto shard = shards.find(interval.shard_name);
        if (shard == shards.end())
          return Status::Internal("DeepSeek mapped shard was not resolved");
        rank.shards.push_back({interval.shard_name,
                               shard->second.file_bytes});
      }
      auto total = checked_add_u64(rank.mapped_interval_bytes,
                                   interval.file_end - interval.file_begin);
      if (!total.ok()) return total.status();
      rank.mapped_interval_bytes = *total;
    }
  }
  return result;
}

const DeepSeekRankMappingPlan& DeepSeekStageMappingPlan::rank(
    std::uint32_t rank) const {
  return ranks_.at(rank);
}

}  // namespace pih
