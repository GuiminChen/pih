#pragma once

#include "pih/model/deepseek_dspark_resident_subwave_executor.h"
#include "pih/model/deepseek_expert_compute_driver.h"

namespace pih {

// Adapts strong D-Spark stage identities to the common FP4 expert math
// backend. It never manufactures a DeepSeekExpertIdentity/layer number.
class DeepSeekDsparkExpertComputeDriver final
    : public DeepSeekDsparkExpertKernelDriver {
 public:
  static constexpr std::uint32_t kTokenCount = 5;

  static Result<DeepSeekDsparkExpertComputeDriver> Create(
      DeepSeekExpertComputeArena arena,
      std::uintptr_t source_hidden_bf16,
      std::uintptr_t accumulator_f32,
      std::uintptr_t stream,
      std::uint64_t context_identity,
      DeepSeekExpertComputeBackend& backend);

  Status launch_resident(
      DeepSeekDsparkStageId stage, std::uint16_t expert,
      std::uint64_t generation,
      const DeepSeekExpertBundleDeviceView& bundle,
      const DeepSeekExpertRoute* routes,
      std::uint32_t route_count) override;
  Result<DeepSeekExpertAsyncStatus> poll() override;

 private:
  enum class State : std::uint8_t { kIdle, kInflight, kPoisoned };
  DeepSeekExpertComputeArena arena_;
  std::uintptr_t source_hidden_bf16_ = 0;
  std::uintptr_t accumulator_f32_ = 0;
  std::uintptr_t stream_ = 0;
  std::uint64_t context_identity_ = 0;
  DeepSeekExpertComputeBackend* backend_ = nullptr;
  State state_ = State::kIdle;
};

}  // namespace pih
