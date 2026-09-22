#pragma once

#include <array>
#include <optional>
#include <span>
#include <vector>

#include "pih/model/deepseek_dense_attention_stage_backend.h"

namespace pih {

struct DeepSeekBoundDenseAttentionLayerWork final {
  std::uint32_t layer = 0;
  std::span<const DeepSeekDenseAttentionStageSequenceWork> sequences;
};

class DeepSeekBoundDenseAttentionWorkProvider final
    : public DeepSeekDenseAttentionStageWorkProvider {
 public:
  static Result<DeepSeekBoundDenseAttentionWorkProvider> Create(
      DeepSeekStageRange owned_layers, std::uint32_t maximum_sequences);
  Status bind(
      const DeepSeekPipelinePlanDescriptor& descriptor,
      std::span<const DeepSeekBoundDenseAttentionLayerWork> layers);
  void clear() noexcept { descriptor_.reset(); }
  Result<std::span<const DeepSeekDenseAttentionStageSequenceWork>> resolve(
      const DeepSeekStageOperatorCommand& command,
      const DeepSeekPipelinePlanDescriptor& plan) override;

 private:
  using Bank = std::array<
      std::vector<DeepSeekDenseAttentionStageSequenceWork>, 43>;
  DeepSeekStageRange owned_layers_;
  std::uint32_t maximum_sequences_ = 0;
  std::array<Bank, 2> banks_;
  std::size_t active_bank_ = 0;
  std::optional<DeepSeekPipelinePlanDescriptor> descriptor_;
};

}  // namespace pih
