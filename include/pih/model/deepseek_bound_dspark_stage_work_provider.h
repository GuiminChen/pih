#pragma once

#include <array>
#include <optional>

#include "pih/model/deepseek_dspark_stage_backend.h"

namespace pih {

class DeepSeekBoundDsparkStageWorkProvider final
    : public DeepSeekDsparkStageWorkProvider {
 public:
  static Result<DeepSeekBoundDsparkStageWorkProvider> Create(
      bool owns_dspark, std::uint32_t maximum_sequences);
  Status bind(const DeepSeekPipelinePlanDescriptor& descriptor,
              DeepSeekDsparkStageWork work);
  void clear() noexcept { descriptor_.reset(); }
  Result<const DeepSeekDsparkStageWork*> resolve(
      const DeepSeekPipelinePlanDescriptor& plan) override;

 private:
  std::uint32_t maximum_sequences_ = 0;
  std::array<DeepSeekDsparkStageWork, 2> banks_{};
  std::size_t active_bank_ = 0;
  std::optional<DeepSeekPipelinePlanDescriptor> descriptor_;
};

}  // namespace pih
