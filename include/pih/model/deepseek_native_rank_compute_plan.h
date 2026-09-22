#pragma once

#include <vector>

#include "pih/model/deepseek_rank_compute_bundle.h"
#include "pih/model/deepseek_v4_config.h"

namespace pih {

class DeepSeekNativeRankComputePlan final {
 public:
  static Result<DeepSeekNativeRankComputePlan> Create(
      DeepSeekPipelinePlanDescriptor descriptor,
      const DeepSeekPipelinePlan& topology,
      std::vector<DeepSeekRankComputePlanWork> rank_work);

  [[nodiscard]] const DeepSeekPipelinePlanDescriptor& descriptor()
      const noexcept { return descriptor_; }
  [[nodiscard]] std::span<const DeepSeekRankComputePlanWork> rank_work()
      const noexcept { return rank_work_; }
  std::vector<DeepSeekRankComputePlanWork> release_rank_work() && noexcept {
    return std::move(rank_work_);
  }

 private:
  DeepSeekNativeRankComputePlan(
      DeepSeekPipelinePlanDescriptor descriptor,
      std::vector<DeepSeekRankComputePlanWork> rank_work) noexcept
      : descriptor_(descriptor), rank_work_(std::move(rank_work)) {}

  DeepSeekPipelinePlanDescriptor descriptor_;
  std::vector<DeepSeekRankComputePlanWork> rank_work_;
};

}  // namespace pih
