#pragma once

#include <span>
#include <vector>

#include "pih/model/deepseek_endpoint_sequence_executor.h"
#include "pih/model/deepseek_stage_compute_driver.h"

namespace pih {

struct DeepSeekEndpointStageSequenceWork final {
  DeepSeekEndpointSequenceExecutor* executor = nullptr;
  DeepSeekAttentionSequenceTransaction* transaction = nullptr;
  DeepSeekEmbeddingLaunch embedding;
  DeepSeekHeadSequenceSubmission head;
};

class DeepSeekEndpointStageWorkProvider {
 public:
  virtual ~DeepSeekEndpointStageWorkProvider() = default;
  virtual Result<std::span<const DeepSeekEndpointStageSequenceWork>> resolve(
      const DeepSeekStageOperatorCommand& command,
      const DeepSeekPipelinePlanDescriptor& plan) = 0;
};

class DeepSeekEndpointStageOperatorBackend final
    : public DeepSeekStageOperatorBackend {
 public:
  static Result<DeepSeekEndpointStageOperatorBackend> Create(
      DeepSeekStageOperatorBackend& fallback,
      DeepSeekEndpointStageWorkProvider& provider);
  Status launch(const DeepSeekStageOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor& plan) override;
  Result<DeepSeekStageComputeStatus> poll() override;

 private:
  enum class Mode : std::uint8_t { kIdle, kEndpoint, kFallback, kPoisoned };
  Status poison(Status status);
  DeepSeekStageOperatorBackend* fallback_ = nullptr;
  DeepSeekEndpointStageWorkProvider* provider_ = nullptr;
  Mode mode_ = Mode::kIdle;
};

}  // namespace pih
