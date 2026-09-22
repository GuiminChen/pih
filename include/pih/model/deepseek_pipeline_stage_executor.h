#pragma once

#include "pih/model/deepseek_pipeline_transaction.h"
#include "pih/model/deepseek_v4_config.h"

namespace pih {

enum class DeepSeekStageComputeStatus : std::uint8_t {
  kInProgress,
  kSuccess,
  kError,
};

class DeepSeekStageComputeDriver {
 public:
  virtual ~DeepSeekStageComputeDriver() = default;
  virtual Status launch(const DeepSeekPipelinePlanDescriptor& plan,
                        const DeepSeekStagePlan& stage) = 0;
  virtual Result<DeepSeekStageComputeStatus> poll() = 0;
};


struct DeepSeekStageExecutionDrivers final {
  DeepSeekStageComputeDriver* compute = nullptr;
};

enum class DeepSeekPipelineStageExecutorState : std::uint8_t {
  kReady,
  kCompute,
  kComplete,
  kPoisoned,
};

class DeepSeekPipelineStageExecutor final {
 public:
  static Result<DeepSeekPipelineStageExecutor> Create(
      const DeepSeekPipelineTransaction& transaction,
      DeepSeekStagePlan stage,
      std::uint32_t world_size
      );
  static Result<DeepSeekPipelineStageExecutor> CreateStaged(
      const DeepSeekPipelineTransaction& transaction,
      DeepSeekStagePlan stage,
      std::uint32_t world_size
      );
  Status advance(DeepSeekStageExecutionDrivers& drivers);
  [[nodiscard]] DeepSeekPipelineStageExecutorState state() const noexcept {
    return state_;
  }
  [[nodiscard]] const DeepSeekPipelinePlanDescriptor& descriptor() const noexcept {
    return transaction_->descriptor();
  }
  [[nodiscard]] const DeepSeekStagePlan& stage() const noexcept {
    return stage_;
  }

 private:
  DeepSeekPipelineStageExecutor(
      const DeepSeekPipelineTransaction& transaction, DeepSeekStagePlan stage,
      std::uint32_t world_size
      ) noexcept
      : transaction_(&transaction), stage_(stage), world_size_(world_size)
      {}
  Status launch_compute(DeepSeekStageComputeDriver& driver);
  Status poison(Status status);
  const DeepSeekPipelineTransaction* transaction_ = nullptr;
  DeepSeekStagePlan stage_;
  std::uint32_t world_size_ = 0;
  DeepSeekPipelineStageExecutorState state_ =
      DeepSeekPipelineStageExecutorState::kReady;
};

}  // namespace pih
