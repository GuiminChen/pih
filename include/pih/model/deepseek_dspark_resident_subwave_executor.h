#pragma once

#include "pih/model/deepseek_dspark_resident_expert_bindings.h"
#include "pih/model/deepseek_expert_subwave_executor.h"

namespace pih {

class DeepSeekDsparkExpertKernelDriver {
 public:
  virtual ~DeepSeekDsparkExpertKernelDriver() = default;
  virtual Status launch_resident(
      DeepSeekDsparkStageId stage, std::uint16_t expert,
      std::uint64_t generation,
      const DeepSeekExpertBundleDeviceView& bundle,
      const DeepSeekExpertRoute* routes, std::uint32_t route_count) = 0;
  virtual Result<DeepSeekExpertAsyncStatus> poll() = 0;
};

class DeepSeekDsparkResidentSubwaveExecutor final {
 public:
  static Result<DeepSeekDsparkResidentSubwaveExecutor> Create(
      DeepSeekDsparkStageId stage,
      const DeepSeekExpertSubwavePlan& plan,
      const DeepSeekDsparkResidentExpertBindings& bindings);

  Status advance(DeepSeekDsparkExpertKernelDriver& kernel);
  [[nodiscard]] DeepSeekExpertSubwaveExecutorState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint32_t completed_experts() const noexcept {
    return completed_experts_;
  }

 private:
  Status poison(Status status);
  DeepSeekDsparkStageId stage_ = DeepSeekDsparkStageId::kMtp0;
  const DeepSeekExpertSubwavePlan* plan_ = nullptr;
  const DeepSeekDsparkResidentExpertBindings* bindings_ = nullptr;
  std::uint16_t next_expert_ = 0;
  std::uint32_t completed_experts_ = 0;
  DeepSeekExpertSubwaveExecutorState state_ =
      DeepSeekExpertSubwaveExecutorState::kReady;
};

}  // namespace pih
