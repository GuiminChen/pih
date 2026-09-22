#pragma once

#include <array>
#include <optional>
#include <span>
#include <vector>

#include "pih/model/deepseek_mhc_stage_backend.h"

namespace pih {

struct DeepSeekBoundMhcLayerWork final {
  std::uint32_t layer = 0;
  std::span<const DeepSeekMhcStageSequenceWork> sequences;
};

class DeepSeekBoundMhcStageWorkProvider final
    : public DeepSeekMhcStageWorkProvider {
 public:
  static Result<DeepSeekBoundMhcStageWorkProvider> Create(
      DeepSeekStageRange owned_layers, std::uint32_t maximum_sequences);
  Status bind(const DeepSeekPipelinePlanDescriptor& descriptor,
              std::span<const DeepSeekBoundMhcLayerWork> attention,
              std::span<const DeepSeekBoundMhcLayerWork> feed_forward);
  void clear() noexcept { descriptor_.reset(); }
  Result<std::span<const DeepSeekMhcStageSequenceWork>> resolve(
      const DeepSeekStageOperatorCommand& command,
      const DeepSeekPipelinePlanDescriptor& plan) override;

 private:
  using Bank = std::array<std::array<
      std::vector<DeepSeekMhcStageSequenceWork>, 43>, 2>;
  DeepSeekStageRange owned_layers_;
  std::uint32_t maximum_sequences_ = 0;
  std::array<Bank, 2> banks_;
  std::size_t active_bank_ = 0;
  std::optional<DeepSeekPipelinePlanDescriptor> descriptor_;
};

}  // namespace pih
