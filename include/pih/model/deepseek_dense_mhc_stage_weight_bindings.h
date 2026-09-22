#pragma once

#include <span>
#include <vector>

#include "pih/model/deepseek_dense_mhc_stage_submission_assembler.h"

namespace pih {

class DeepSeekDenseMhcStageWeightBindings final {
 public:
  using Resolver = DeepSeekAttentionWeightBindings::Resolver;

  static Result<DeepSeekDenseMhcStageWeightBindings> Resolve(
      DeepSeekStageRange owned_layers, const Resolver& resolver);
  static Result<DeepSeekDenseMhcStageWeightBindings> Resolve(
      DeepSeekStageRange owned_layers,
      const DeepSeekResidentWeightArena& arena);

  [[nodiscard]] std::span<const DeepSeekDenseMhcLayerWeightInput> layers()
      const noexcept { return layers_; }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }
  [[nodiscard]] DeepSeekStageRange owned_layers() const noexcept {
    return owned_layers_;
  }

 private:
  DeepSeekStageRange owned_layers_;
  std::vector<DeepSeekDenseMhcLayerWeightInput> layers_;
  std::uint64_t generation_ = 0;
};

}  // namespace pih
