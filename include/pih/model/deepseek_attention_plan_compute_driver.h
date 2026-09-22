#pragma once

#include <span>
#include <vector>

#include "pih/model/deepseek_attention_sequence_transaction.h"
#include "pih/model/deepseek_pipeline_stage_executor.h"

namespace pih {

class DeepSeekAttentionPlanComputeDriver final
    : public DeepSeekStageComputeDriver {
 public:
  static Result<DeepSeekAttentionPlanComputeDriver> Create(
      DeepSeekStageComputeDriver& inner,
      std::span<DeepSeekAttentionSequenceTransaction* const> transactions,
      std::uintptr_t stream);

  Status launch(const DeepSeekPipelinePlanDescriptor& plan,
                const DeepSeekStagePlan& stage) override;
  Result<DeepSeekStageComputeStatus> poll() override;

 private:
  enum class State : std::uint8_t {
    kReady,
    kCompute,
    kAwaitingState,
    kComplete,
    kPoisoned,
  };
  Status seal_all();
  Result<DeepSeekStageComputeStatus> resolve_all();
  Result<DeepSeekStageComputeStatus> poison(Status status);

  DeepSeekStageComputeDriver* inner_ = nullptr;
  std::vector<DeepSeekAttentionSequenceTransaction*> transactions_;
  std::uintptr_t stream_ = 0;
  bool manages_state_ = false;
  bool compute_failed_ = false;
  State state_ = State::kReady;
};

}  // namespace pih
