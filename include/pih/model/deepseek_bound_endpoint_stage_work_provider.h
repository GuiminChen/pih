#pragma once

#include <array>
#include <optional>
#include <vector>

#include "pih/model/deepseek_endpoint_stage_backend.h"

namespace pih {

class DeepSeekBoundEndpointStageWorkProvider final
    : public DeepSeekEndpointStageWorkProvider {
 public:
  static Result<DeepSeekBoundEndpointStageWorkProvider> Create(
      std::uint32_t maximum_sequences, bool owns_embedding,
      bool owns_lm_head);

  Status bind(const DeepSeekPipelinePlanDescriptor& descriptor,
              std::span<const DeepSeekEndpointStageSequenceWork> work);
  void clear() noexcept { descriptor_.reset(); }

  Result<std::span<const DeepSeekEndpointStageSequenceWork>> resolve(
      const DeepSeekStageOperatorCommand& command,
      const DeepSeekPipelinePlanDescriptor& plan) override;

 private:
  std::uint32_t maximum_sequences_ = 0;
  bool owns_embedding_ = false;
  bool owns_lm_head_ = false;
  std::array<std::vector<DeepSeekEndpointStageSequenceWork>, 2> banks_;
  std::size_t active_bank_ = 0;
  std::optional<DeepSeekPipelinePlanDescriptor> descriptor_;
};

}  // namespace pih
