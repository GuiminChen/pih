#pragma once

#include <array>
#include <optional>
#include <span>

#include "pih/model/deepseek_bound_dspark_mtp_operator_backend.h"

namespace pih {

// Holds immutable plan templates and resolves the tentative D-Spark fixed-state
// bank only after the plan transaction has entered PREPARING.  This prevents a
// compiled plan from retaining a stale committed/tentative bank address.
class DeepSeekBoundDsparkMtpStageWorkProvider final
    : public DeepSeekDsparkMtpStageWorkProvider {
 public:
  static Result<DeepSeekBoundDsparkMtpStageWorkProvider> Create(
      bool owns_dspark);

  Status bind(
      const DeepSeekPipelinePlanDescriptor& descriptor,
      std::span<const DeepSeekBoundDsparkMtpStageWork> stages);
  void clear() noexcept { descriptor_.reset(); }

  Result<const DeepSeekBoundDsparkMtpStageWork*> resolve(
      DeepSeekDsparkStageId stage,
      const DeepSeekPipelinePlanDescriptor& plan) override;

 private:
  using StageSet = std::array<DeepSeekBoundDsparkMtpStageWork, 3>;

  std::array<StageSet, 2> templates_{};
  std::array<StageSet, 2> resolved_{};
  std::size_t active_bank_ = 0;
  std::optional<DeepSeekPipelinePlanDescriptor> descriptor_;
};

}  // namespace pih
