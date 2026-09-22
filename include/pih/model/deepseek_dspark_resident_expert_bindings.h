#pragma once

#include <array>
#include <functional>
#include <string_view>

#include "pih/model/deepseek_expert_bundle_layout.h"
#include "pih/model/deepseek_dspark_stage_identity.h"
#include "pih/model/deepseek_resident_weight_arena.h"
#include "pih/model/deepseek_stage_mapping_plan.h"

namespace pih {

class DeepSeekDsparkResidentExpertBindings final {
 public:
  static constexpr std::uint32_t kExpertCount = 256;
  using Resolver = std::function<Result<TensorView>(std::string_view)>;

  static Result<DeepSeekDsparkResidentExpertBindings> Resolve(
      DeepSeekStagePlan stage, const Resolver& resolver);
  static Result<DeepSeekDsparkResidentExpertBindings> Resolve(
      DeepSeekStagePlan stage, const DeepSeekResidentWeightArena& arena);

  [[nodiscard]] const DeepSeekExpertBundleDeviceView& expert(
      DeepSeekDsparkStageId stage, std::uint32_t id) const {
    return experts_.at(deepseek_dspark_stage_index(stage)).at(id);
  }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }

 private:
  std::array<std::array<DeepSeekExpertBundleDeviceView, kExpertCount>,
             kDeepSeekDsparkStageCount> experts_{};
  std::uint64_t generation_ = 0;
};

}  // namespace pih
