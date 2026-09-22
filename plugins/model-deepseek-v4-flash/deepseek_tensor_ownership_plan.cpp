#include "pih/model/deepseek_tensor_ownership_plan.h"

#include <algorithm>
#include <charconv>
#include <limits>

#include "pih/model/deepseek_runtime_records_manifest.h"

namespace pih { namespace {

struct Classification final {
  DeepSeekTensorRole role{};
  std::uint32_t layer = UINT32_MAX;
};

Result<std::uint32_t> indexed_prefix(std::string_view name,
                                     std::string_view prefix,
                                     std::uint32_t maximum) {
  if (!name.starts_with(prefix))
    return Status::InvalidArgument("DeepSeek tensor prefix is invalid");
  const auto begin = prefix.size();
  const auto dot = name.find('.', begin);
  if (dot == name.npos || dot == begin || dot + 1 >= name.size() ||
      (dot - begin > 1 && name[begin] == '0')) {
    return Status::InvalidArgument("DeepSeek tensor layer syntax is invalid");
  }
  std::uint32_t value = 0;
  const auto parsed = std::from_chars(name.data() + begin,
                                      name.data() + dot, value);
  if (parsed.ec != std::errc{} || parsed.ptr != name.data() + dot ||
      value > maximum) {
    return Status::InvalidArgument("DeepSeek tensor layer is out of range");
  }
  return value;
}

Result<Classification> classify(std::string_view name) {
  if (name == "embed.weight")
    return Classification{DeepSeekTensorRole::kEmbedding};
  if (name == "head.weight" || name == "norm.weight" ||
      name == "hc_head_fn" || name == "hc_head_base" ||
      name == "hc_head_scale") {
    return Classification{DeepSeekTensorRole::kFinalHead};
  }
  if (name.starts_with("layers.")) {
    auto layer = indexed_prefix(name, "layers.", 42);
    if (!layer.ok()) return layer.status();
    return Classification{DeepSeekTensorRole::kMainLayer, *layer};
  }
  if (name.starts_with("mtp.")) {
    auto layer = indexed_prefix(name, "mtp.", 2);
    if (!layer.ok()) return layer.status();
    return Classification{DeepSeekTensorRole::kDspark, *layer};
  }
  return Status::InvalidArgument("DeepSeek tensor name has unknown ownership");
}

}  // namespace

Result<DeepSeekTensorOwnershipPlan> DeepSeekTensorOwnershipPlan::Create(
    const DeepSeekPipelinePlan& pipeline,
    std::span<const SafetensorsShardBinding> bindings) {
  if (pipeline.world_size() == 0 || pipeline.world_size() > 4 ||
      bindings.empty()) {
    return Status::InvalidArgument("DeepSeek tensor ownership input is invalid");
  }
  DeepSeekTensorOwnershipPlan result;
  result.per_rank_counts_.resize(pipeline.world_size());
  result.records_.reserve(bindings.size());
  std::vector<SafetensorsShardBinding> canonical(bindings.begin(),
                                                  bindings.end());
  std::ranges::sort(canonical, {}, &SafetensorsShardBinding::tensor_name);
  for (std::size_t index = 0; index < canonical.size(); ++index) {
    const auto& binding = canonical[index];
    if (binding.tensor_name.empty() || binding.shard_name.empty()) {
      return Status::InvalidArgument(
          "DeepSeek tensor ownership binding is empty");
    }
    if (index != 0 &&
        canonical[index - 1].tensor_name == binding.tensor_name) {
      return Status::InvalidArgument(
          "DeepSeek tensor ownership contains a duplicate tensor");
    }
    auto classified = classify(binding.tensor_name);
    if (!classified.ok()) return classified.status();
    std::uint32_t owner = DeepSeekTensorOwnership::kExcludedRank;
    switch (classified->role) {
      case DeepSeekTensorRole::kEmbedding:
        owner = 0;
        break;
      case DeepSeekTensorRole::kFinalHead:
        owner = pipeline.world_size() - 1;
        break;
      case DeepSeekTensorRole::kMainLayer:
        for (std::uint32_t rank = 0; rank < pipeline.world_size(); ++rank) {
          const auto range = pipeline.rank(rank).layers;
          if (classified->layer >= range.first_layer &&
              classified->layer <= range.last_layer) {
            owner = rank;
            break;
          }
        }
        break;
      case DeepSeekTensorRole::kDspark:
        if (pipeline.rank(pipeline.world_size() - 1).owns_dspark) {
          owner = pipeline.world_size() - 1;
        } else {
          ++result.excluded_dspark_count_;
        }
        break;
    }
    if (owner != DeepSeekTensorOwnership::kExcludedRank) {
      if (owner >= pipeline.world_size())
        return Status::Internal("DeepSeek tensor owner was not resolved");
      ++result.per_rank_counts_[owner];
    } else if (classified->role != DeepSeekTensorRole::kDspark) {
      return Status::Internal("DeepSeek required tensor has no owner");
    }
    result.records_.push_back({binding.tensor_name, binding.shard_name,
                               classified->role, classified->layer, owner});
  }
  return result;
}

Result<DeepSeekTensorOwnershipPlan>
DeepSeekTensorOwnershipPlan::BindTargetManifest(
    const DeepSeekPipelinePlan& pipeline,
    const DeepSeekRuntimeRecordsManifest& manifest) {
  if (pipeline.world_size() == 0 || pipeline.world_size() > 4 ||
      manifest.records().empty() ||
      manifest.authority().world_size != pipeline.world_size()) {
    return Status::InvalidArgument(
        "DeepSeek target ownership authority is invalid");
  }
  DeepSeekTensorOwnershipPlan result;
  result.per_rank_counts_.resize(pipeline.world_size());
  result.records_.reserve(manifest.records().size());
  for (const auto& record : manifest.records()) {
    if (record.tensor_name.empty() || record.shard_name.empty() ||
        record.owner_rank >= pipeline.world_size() ||
        (!result.records_.empty() &&
         result.records_.back().tensor_name >= record.tensor_name)) {
      return Status::InvalidArgument(
          "DeepSeek target ownership records are invalid or unordered");
    }
    bool owner_matches = false;
    switch (record.role) {
      case DeepSeekTensorRole::kEmbedding:
        owner_matches = record.owner_rank == 0 &&
                        record.logical_layer == UINT32_MAX;
        break;
      case DeepSeekTensorRole::kFinalHead:
        owner_matches = record.owner_rank == pipeline.world_size() - 1 &&
                        record.logical_layer == UINT32_MAX;
        break;
      case DeepSeekTensorRole::kMainLayer: {
        const auto range = pipeline.rank(record.owner_rank).layers;
        owner_matches = record.logical_layer >= range.first_layer &&
                        record.logical_layer <= range.last_layer;
        break;
      }
      case DeepSeekTensorRole::kDspark:
        owner_matches = manifest.authority().dspark_enabled &&
                        record.owner_rank == pipeline.world_size() - 1 &&
                        pipeline.rank(record.owner_rank).owns_dspark &&
                        record.logical_layer <= 2;
        break;
    }
    if (!owner_matches) {
      return Status::InvalidArgument(
          "DeepSeek target ownership differs from pipeline profile");
    }
    ++result.per_rank_counts_[record.owner_rank];
    result.records_.push_back(
        {record.tensor_name, record.shard_name, record.role,
         record.logical_layer, record.owner_rank});
  }
  return result;
}

std::uint32_t DeepSeekTensorOwnershipPlan::owned_count(
    std::uint32_t rank) const {
  if (rank >= per_rank_counts_.size()) return 0;
  return per_rank_counts_[rank];
}

}  // namespace pih
