#pragma once

#include <cstdint>

#include "pih/model/deepseek_expert_compute_arena.h"
#include "pih/model/deepseek_expert_slot_table.h"
#include "pih/model/deepseek_expert_subwave_executor.h"

namespace pih {

struct DeepSeekExpertComputeSubmission final {
  // The CUDA math consumes an already-verified resident or paged bundle.
  // Main-model identity remains in expert for pager evidence; D-Spark uses
  // the strong stage identity in its dedicated driver and leaves expert
  // empty rather than fabricating a main-model layer number.
  std::uint64_t context_identity = 0;
  DeepSeekExpertBundleDeviceView bundle;
  DeepSeekExpertLeaseDeviceView expert;
  DeepSeekExpertComputeArena arena;
  const DeepSeekExpertRoute* routes = nullptr;
  std::uint32_t route_count = 0;
  std::uint32_t packed_token_count = 0;
  std::uintptr_t source_hidden_bf16 = 0;
  std::uintptr_t accumulator_f32 = 0;
  std::uintptr_t stream = 0;
};

class DeepSeekExpertComputeBackend {
 public:
  virtual ~DeepSeekExpertComputeBackend() = default;
  // submit must consume/copy the host route slice before returning.
  virtual Status submit(const DeepSeekExpertComputeSubmission& submission) = 0;
  virtual Result<DeepSeekExpertAsyncStatus> poll() = 0;
};

class DeepSeekExpertComputeDriver final : public DeepSeekExpertKernelDriver {
 public:
  static Result<DeepSeekExpertComputeDriver> Create(
      DeepSeekExpertSlotTable slots, DeepSeekExpertComputeArena arena,
      std::uint32_t packed_token_count, std::uintptr_t source_hidden_bf16,
      std::uintptr_t accumulator_f32, std::uintptr_t stream,
      DeepSeekExpertComputeBackend& backend);

  Status launch(const DeepSeekExpertLease& lease,
                const DeepSeekExpertRoute* routes,
                std::uint32_t route_count) override;
  Status launch_resident(
      DeepSeekExpertIdentity identity, std::uint64_t generation,
      const DeepSeekExpertBundleDeviceView& bundle,
      const DeepSeekExpertRoute* routes,
      std::uint32_t route_count) override;
  Result<DeepSeekExpertAsyncStatus> poll() override;

 private:
  enum class State : std::uint8_t { kIdle, kInflight, kPoisoned };
  DeepSeekExpertSlotTable slots_;
  DeepSeekExpertComputeArena arena_;
  std::uint32_t packed_token_count_ = 0;
  std::uintptr_t source_hidden_bf16_ = 0;
  std::uintptr_t accumulator_f32_ = 0;
  std::uintptr_t stream_ = 0;
  DeepSeekExpertComputeBackend* backend_ = nullptr;
  State state_ = State::kIdle;
  Status launch_bound(DeepSeekExpertLeaseDeviceView expert,
                      const DeepSeekExpertRoute* routes,
                      std::uint32_t route_count);
};

}  // namespace pih
