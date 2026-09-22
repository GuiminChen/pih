#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pih/model/deepseek_v4_config.h"
#include "pih/model/safetensors_shard_index.h"

namespace pih {

class DeepSeekRuntimeRecordsManifest;

enum class DeepSeekTensorRole : std::uint8_t {
  kEmbedding = 1,
  kMainLayer = 2,
  kFinalHead = 3,
  kDspark = 4,
};

struct DeepSeekTensorOwnership final {
  static constexpr std::uint32_t kExcludedRank = UINT32_MAX;
  std::string tensor_name;
  std::string shard_name;
  DeepSeekTensorRole role{};
  std::uint32_t logical_layer = UINT32_MAX;
  std::uint32_t owner_rank = kExcludedRank;
};

class DeepSeekTensorOwnershipPlan final {
 public:
  // Offline source-compiler classification. Production bootstrap must not
  // treat tensor-name parsing as ownership authority.
  static Result<DeepSeekTensorOwnershipPlan> Create(
      const DeepSeekPipelinePlan& pipeline,
      std::span<const SafetensorsShardBinding> bindings);
  // Production binding path. Ownership is accepted only from the verified
  // target runtime-record manifest; no tensor-name classification occurs.
  static Result<DeepSeekTensorOwnershipPlan> BindTargetManifest(
      const DeepSeekPipelinePlan& pipeline,
      const DeepSeekRuntimeRecordsManifest& manifest);
  [[nodiscard]] const std::vector<DeepSeekTensorOwnership>& records()
      const noexcept { return records_; }
  [[nodiscard]] std::uint32_t owned_count(std::uint32_t rank) const;
  [[nodiscard]] std::uint32_t excluded_dspark_count() const noexcept {
    return excluded_dspark_count_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return static_cast<std::uint32_t>(per_rank_counts_.size());
  }

 private:
  std::vector<DeepSeekTensorOwnership> records_;
  std::vector<std::uint32_t> per_rank_counts_;
  std::uint32_t excluded_dspark_count_ = 0;
};

}  // namespace pih
