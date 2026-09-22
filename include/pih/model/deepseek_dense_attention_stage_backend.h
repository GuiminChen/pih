#pragma once

#include <span>
#include <vector>

#include "pih/model/deepseek_attention_output_projection_coordinator.h"
#include "pih/model/deepseek_attention_projection_coordinator.h"
#include "pih/model/deepseek_stage_compute_driver.h"

namespace pih {

struct DeepSeekDenseAttentionStageSequenceWork final {
  DeepSeekAttentionProjectionCoordinator* input_coordinator = nullptr;
  DeepSeekAttentionOutputProjectionCoordinator* output_coordinator = nullptr;
  DeepSeekAttentionSequenceTransaction* transaction = nullptr;
  DeepSeekAttentionProjectionSubmission input;
  DeepSeekAttentionOutputProjectionSubmission output;
  // Addresses consumed and produced by the wrapped sparse-attention backend.
  std::uintptr_t sparse_query_bf16 = 0;
  std::uintptr_t sparse_kv_bf16 = 0;
  std::uintptr_t sparse_output_bf16 = 0;
};

class DeepSeekDenseAttentionStageWorkProvider {
 public:
  virtual ~DeepSeekDenseAttentionStageWorkProvider() = default;
  virtual Result<std::span<const DeepSeekDenseAttentionStageSequenceWork>>
  resolve(const DeepSeekStageOperatorCommand& command,
          const DeepSeekPipelinePlanDescriptor& plan) = 0;
};

class DeepSeekDenseAttentionStageOperatorBackend final
    : public DeepSeekStageOperatorBackend {
 public:
  static Result<DeepSeekDenseAttentionStageOperatorBackend> Create(
      DeepSeekStageOperatorBackend& inner,
      DeepSeekDenseAttentionStageWorkProvider& provider);

  Status launch(const DeepSeekStageOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor& plan) override;
  Result<DeepSeekStageComputeStatus> poll() override;

 private:
  enum class Mode : std::uint8_t {
    kIdle,
    kPassthrough,
    kAttention,
    kPoisoned,
  };
  Status poison(Status status);

  DeepSeekStageOperatorBackend* inner_ = nullptr;
  DeepSeekDenseAttentionStageWorkProvider* provider_ = nullptr;
  std::vector<DeepSeekDenseAttentionStageSequenceWork> work_;
  Mode mode_ = Mode::kIdle;
};

}  // namespace pih
