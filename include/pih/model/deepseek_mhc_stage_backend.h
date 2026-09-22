#pragma once

#include <span>
#include <vector>

#include "pih/model/deepseek_mhc_sequence_executor.h"
#include "pih/model/deepseek_stage_compute_driver.h"

namespace pih {

struct DeepSeekMhcStageSequenceWork final {
  DeepSeekMhcSequenceExecutor* executor = nullptr;
  DeepSeekAttentionSequenceTransaction* transaction = nullptr;
  DeepSeekMhcSequenceSubmission submission;
};

class DeepSeekMhcStageWorkProvider {
 public:
  virtual ~DeepSeekMhcStageWorkProvider() = default;
  virtual Result<std::span<const DeepSeekMhcStageSequenceWork>> resolve(
      const DeepSeekStageOperatorCommand& command,
      const DeepSeekPipelinePlanDescriptor& plan) = 0;
};

class DeepSeekMhcStageOperatorBackend final
    : public DeepSeekStageOperatorBackend {
 public:
  static Result<DeepSeekMhcStageOperatorBackend> Create(
      DeepSeekStageOperatorBackend& inner,
      DeepSeekMhcStageWorkProvider& provider);

  Status launch(const DeepSeekStageOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor& plan) override;
  Result<DeepSeekStageComputeStatus> poll() override;

 private:
  enum class Mode : std::uint8_t { kIdle, kPassthrough, kMhc, kPoisoned };
  Status poison(Status status);

  DeepSeekStageOperatorBackend* inner_ = nullptr;
  DeepSeekMhcStageWorkProvider* provider_ = nullptr;
  std::vector<DeepSeekMhcStageSequenceWork> work_;
  Mode mode_ = Mode::kIdle;
};

}  // namespace pih
