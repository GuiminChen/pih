#pragma once

#include "pih/model/qwen3_bf16_synchronous_backend.h"
#include "pih/model/qwen3_int4_step_builder.h"
#include "pih/model/qwen3_int4_step_transaction.h"

namespace pih {

using QwenInt4SynchronousBackendArenas=QwenBf16SynchronousBackendArenas;
using QwenInt4SynchronousBackendIdentity=QwenBf16SynchronousBackendIdentity;

struct QwenInt4SynchronousBackendDrivers final {
  TypedCopyDriver* copy;
  QwenBf16DeviceErrorClearDriver* clear;
  KernelLaunchDriver* kernels;
  QwenInt4LmHeadExecutionDriver* lm_head;
  CompletionEventDriver* events;
  QwenBf16StepHealthProvider* health;
  QwenBf16MonotonicClock* clock;
  QwenBf16PollWaiter* waiter;
};

enum class QwenInt4SynchronousBackendState : std::uint8_t {
  kReady,kRunning,kPoisoned,
};

class QwenInt4SynchronousBackend final
    : public QwenBf16SequenceBackend,
      public QwenBf16CompletionIdentityProvider {
 public:
  static Result<QwenInt4SynchronousBackend> Create(
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> functions,
      const QwenInt4WeightResourceSet& weights,
      QwenInt4SynchronousBackendArenas arenas,
      QwenInt4SynchronousBackendIdentity identity,
      QwenInt4SynchronousBackendDrivers drivers);
  QwenInt4SynchronousBackend(const QwenInt4SynchronousBackend&)=delete;
  QwenInt4SynchronousBackend& operator=(const QwenInt4SynchronousBackend&)=delete;
  QwenInt4SynchronousBackend(QwenInt4SynchronousBackend&&) noexcept=default;
  QwenInt4SynchronousBackend& operator=(QwenInt4SynchronousBackend&&)=delete;
  Result<std::int64_t> execute_and_read_token(
      std::span<const std::int64_t> tokens,std::uint64_t first_position,
      const QwenKvBlockTable& block_table,
      const QwenKvAppendPlan& append_plan) override;
  Result<QwenKvCompletionEvent> last_completion_event() const override;
  [[nodiscard]] QwenInt4SynchronousBackendState state() const noexcept{return state_;}
  [[nodiscard]] std::uint64_t next_request_generation() const noexcept{return next_request_generation_;}
  [[nodiscard]] std::uint64_t next_event_generation() const noexcept{return next_event_generation_;}
  [[nodiscard]] std::uint64_t next_plan_id() const noexcept{return next_plan_id_;}
 private:
  QwenInt4SynchronousBackend(const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> functions,
      const QwenInt4WeightResourceSet& weights,
      QwenInt4SynchronousBackendArenas arenas,
      QwenInt4SynchronousBackendIdentity identity,
      QwenInt4SynchronousBackendDrivers drivers)
      :commands_(&commands),functions_(functions),weights_(&weights),arenas_(arenas),
       identity_(identity),drivers_(drivers),
       next_request_generation_(identity.first_request_generation),
       next_event_generation_(identity.first_event_generation),
       next_plan_id_(identity.first_plan_id){}
  const QwenBf16CommandBuffer* commands_;
  std::span<const ResolvedKernelFunction> functions_;
  const QwenInt4WeightResourceSet* weights_;
  QwenInt4SynchronousBackendArenas arenas_;
  QwenInt4SynchronousBackendIdentity identity_;
  QwenInt4SynchronousBackendDrivers drivers_;
  std::uint64_t next_request_generation_,next_event_generation_,next_plan_id_;
  QwenKvCompletionEvent last_completion_event_{};
  QwenInt4SynchronousBackendState state_=QwenInt4SynchronousBackendState::kReady;
};

}  // namespace pih
