#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_tensor_ownership_plan.h"
#include "pih/model/safetensors_header.h"

namespace pih {

struct DeepSeekShardHeaderView final {
  std::string_view shard_name;
  std::uint64_t file_bytes = 0;
  const SafetensorsHeader* header = nullptr;
};

struct DeepSeekMappedInterval final {
  std::string shard_name;
  std::uint64_t file_begin = 0;
  std::uint64_t file_end = 0;
};

struct DeepSeekRankShardPlan final {
  std::string shard_name;
  std::uint64_t file_bytes = 0;
};

struct DeepSeekRankMappingPlan final {
  std::uint32_t rank = 0;
  std::uint32_t owned_tensor_count = 0;
  std::uint64_t logical_tensor_bytes = 0;
  std::uint64_t mapped_interval_bytes = 0;
  std::vector<DeepSeekRankShardPlan> shards;
  std::vector<DeepSeekMappedInterval> intervals;
};

class DeepSeekStageMappingPlan final {
 public:
  static constexpr std::uint64_t kProductionPageBytes = 4096;

  static Result<DeepSeekStageMappingPlan> Create(
      const DeepSeekPipelinePlan& pipeline,
      std::span<const SafetensorsShardBinding> bindings,
      std::span<const DeepSeekShardHeaderView> headers,
      std::uint64_t page_bytes);
  static Result<DeepSeekStageMappingPlan> BindTargetManifest(
      const DeepSeekTensorOwnershipPlan& ownership,
      std::span<const DeepSeekShardHeaderView> headers,
      std::uint64_t page_bytes);

  [[nodiscard]] const DeepSeekRankMappingPlan& rank(
      std::uint32_t rank) const;
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return static_cast<std::uint32_t>(ranks_.size());
  }
  [[nodiscard]] std::uint32_t excluded_tensor_count() const noexcept {
    return excluded_tensor_count_;
  }

 private:
  std::vector<DeepSeekRankMappingPlan> ranks_;
  std::uint32_t excluded_tensor_count_ = 0;
};

}  // namespace pih
