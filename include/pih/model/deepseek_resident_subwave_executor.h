#pragma once

#include "pih/model/deepseek_expert_subwave_executor.h"
#include "pih/model/deepseek_resident_expert_bindings.h"

namespace pih {

class DeepSeekResidentSubwaveExecutor final {
 public:
  static Result<DeepSeekResidentSubwaveExecutor> Create(
      std::uint32_t layer, const DeepSeekExpertSubwavePlan& plan,
      const DeepSeekResidentExpertBindings& bindings);

  Status advance(DeepSeekExpertKernelDriver& kernel);
  [[nodiscard]] DeepSeekExpertSubwaveExecutorState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint32_t completed_experts() const noexcept {
    return completed_experts_;
  }

 private:
  Status poison(Status status);
  std::uint16_t layer_ = 0;
  const DeepSeekExpertSubwavePlan* plan_ = nullptr;
  const DeepSeekResidentExpertBindings* bindings_ = nullptr;
  std::uint16_t next_expert_ = 0;
  std::uint32_t completed_experts_ = 0;
  DeepSeekExpertSubwaveExecutorState state_ =
      DeepSeekExpertSubwaveExecutorState::kReady;
};

}  // namespace pih
