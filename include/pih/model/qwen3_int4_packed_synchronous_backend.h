#pragma once

#include <vector>

#include "pih/model/qwen3_bf16_packed_synchronous_backend.h"
#include "pih/model/qwen3_int4_packed_step_builder.h"
#include "pih/model/qwen3_int4_packed_step_transaction.h"

namespace pih {

using QwenInt4PackedBackendArenas=QwenBf16PackedBackendArenas;
using QwenInt4PackedBackendIdentity=QwenBf16PackedBackendIdentity;

struct QwenInt4PackedBackendDrivers final {
  TypedCopyDriver* copy;
  QwenBf16DeviceErrorClearDriver* clear;
  KernelLaunchDriver* kernels;
  QwenInt4LmHeadExecutionDriver* lm_head;
  CompletionEventDriver* events;
  QwenBf16StepHealthProvider* health;
  QwenBf16MonotonicClock* clock;
  QwenBf16PollWaiter* waiter;
};

enum class QwenInt4PackedBackendState : std::uint8_t {
  kReady=1,kRunning=2,kPoisoned=3,
};

class QwenInt4PackedSynchronousBackend final
    : public QwenBf16PackedBatchBackend {
 public:
  static Result<QwenInt4PackedSynchronousBackend> Create(
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> int4_functions,
      std::span<const ResolvedKernelFunction> packed_functions,
      const QwenInt4WeightResourceSet& weights,
      QwenInt4PackedBackendArenas arenas,
      QwenInt4PackedBackendIdentity identity,
      QwenInt4PackedBackendDrivers drivers);
  QwenInt4PackedSynchronousBackend(
      const QwenInt4PackedSynchronousBackend&)=delete;
  QwenInt4PackedSynchronousBackend(
      QwenInt4PackedSynchronousBackend&&) noexcept=default;

  Result<QwenBf16PackedBatchExecutionView> execute_packed(
      const PackedTokenPlan& plan,PackedTokenMetadataView metadata,
      QwenBf16PackedKvMetadataView kv_metadata,
      std::span<const QwenBf16PackedKvBinding> bindings,
      std::span<const QwenKvAppendPlan> append_plans,
      std::span<const Qwen3SamplingDescriptor> sampling) override;
  [[nodiscard]] QwenInt4PackedBackendState state() const noexcept{return state_;}

 private:
  QwenInt4PackedSynchronousBackend(const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> int4_functions,
      std::span<const ResolvedKernelFunction> packed_functions,
      const QwenInt4WeightResourceSet& weights,
      QwenInt4PackedBackendArenas arenas,
      QwenInt4PackedBackendIdentity identity,
      QwenInt4PackedBackendDrivers drivers)
      :commands_(&commands),int4_(int4_functions),packed_(packed_functions),
       weights_(&weights),arenas_(arenas),identity_(identity),drivers_(drivers),
       request_generation_(identity.first_request_generation),
       event_generation_(identity.first_event_generation),
       plan_id_(identity.first_plan_id){}
  const QwenBf16CommandBuffer* commands_;
  std::span<const ResolvedKernelFunction> int4_,packed_;
  const QwenInt4WeightResourceSet* weights_;
  QwenInt4PackedBackendArenas arenas_;
  QwenInt4PackedBackendIdentity identity_;
  QwenInt4PackedBackendDrivers drivers_;
  std::uint64_t request_generation_,event_generation_,plan_id_;
  std::vector<std::uint32_t> sampled_tokens_;
  std::vector<QwenBf16PackedSampleReceipt> sampling_receipts_;
  QwenInt4PackedBackendState state_=QwenInt4PackedBackendState::kReady;
};

}  // namespace pih
