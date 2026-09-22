#pragma once

#include <vector>

#include "pih/model/deepseek_rank_compute_work_builder.h"

namespace pih {

class DeepSeekEndpointWorkFactory final {
 public:
  static Result<DeepSeekEndpointWorkFactory> Create(
      DeepSeekStagePlan stage, std::uint32_t maximum_sequences);

  Status append_plan_work(
      DeepSeekPlanPhase phase, std::uint32_t sequence_count,
      std::vector<DeepSeekEndpointStageSequenceWork> endpoint,
      DeepSeekRankComputeWorkBuilder& builder) const;
  [[nodiscard]] const DeepSeekStagePlan& stage() const noexcept {
    return stage_;
  }

 private:
  DeepSeekStagePlan stage_;
  std::uint32_t maximum_sequences_ = 0;
};

}  // namespace pih
