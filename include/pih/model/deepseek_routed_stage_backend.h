#pragma once

#include <optional>

#include "pih/model/deepseek_expert_subwave_executor.h"
#include "pih/model/deepseek_resident_subwave_executor.h"
#include "pih/model/deepseek_stage_compute_driver.h"

namespace pih {

class DeepSeekExpertPlanProvider {
 public:
  virtual ~DeepSeekExpertPlanProvider() = default;
  // The returned plan must remain alive until the matching MoE poll succeeds.
  virtual Result<const DeepSeekExpertSubwavePlan*> resolve(
      std::uint32_t layer,
      const DeepSeekPipelinePlanDescriptor& descriptor) = 0;
};

class DeepSeekExpertPlanStore : public DeepSeekExpertPlanProvider {
 public:
  ~DeepSeekExpertPlanStore() override = default;
  virtual Status publish_routes(
      std::uint32_t layer, std::uint32_t token_count,
      std::span<const DeepSeekExpertRoute> routes) = 0;
};

class DeepSeekRoutedStageOperatorBackend final
    : public DeepSeekStageOperatorBackend {
 public:
  static Result<DeepSeekRoutedStageOperatorBackend> Create(
      DeepSeekStageOperatorBackend& dense_backend,
      DeepSeekExpertPlanProvider& plan_provider,
      DeepSeekExpertPager& pager, DeepSeekExpertTransferDriver& transfer,
      DeepSeekExpertKernelDriver& kernel);
  static Result<DeepSeekRoutedStageOperatorBackend> CreateResident(
      DeepSeekStageOperatorBackend& dense_backend,
      DeepSeekExpertPlanProvider& plan_provider,
      DeepSeekExpertKernelDriver& kernel,
      const DeepSeekResidentExpertBindings& resident);

  Status launch(const DeepSeekStageOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor& plan) override;
  Result<DeepSeekStageComputeStatus> poll() override;

 private:
  enum class Mode : std::uint8_t {
    kIdle, kDense, kMoe, kResidentMoe, kPoisoned
  };
  Status poison(Status status);

  DeepSeekStageOperatorBackend* dense_backend_ = nullptr;
  DeepSeekExpertPlanProvider* plan_provider_ = nullptr;
  DeepSeekExpertPager* pager_ = nullptr;
  DeepSeekExpertTransferDriver* transfer_ = nullptr;
  DeepSeekExpertKernelDriver* kernel_ = nullptr;
  const DeepSeekResidentExpertBindings* resident_ = nullptr;
  std::optional<DeepSeekExpertSubwaveExecutor> executor_;
  std::optional<DeepSeekResidentSubwaveExecutor> main_resident_executor_;
  Mode mode_ = Mode::kIdle;
};

}  // namespace pih
