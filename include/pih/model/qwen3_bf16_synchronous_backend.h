#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/model/qwen3_bf16_sequence_session.h"
#include "pih/model/qwen3_bf16_request_runner.h"
#include "pih/model/qwen3_bf16_step_builder.h"

namespace pih {

class QwenBf16MonotonicClock {
 public:
  virtual ~QwenBf16MonotonicClock() = default;
  virtual Result<std::uint64_t> now_ns() = 0;
};

class QwenBf16PollWaiter {
 public:
  virtual ~QwenBf16PollWaiter() = default;
  virtual Status wait() = 0;
};

struct QwenBf16SynchronousBackendDrivers final {
  TypedCopyDriver* copy;
  QwenBf16DeviceErrorClearDriver* clear;
  KernelLaunchDriver* kernels;
  QwenBf16LinearExecutionDriver* linears;
  CompletionEventDriver* events;
  QwenBf16StepHealthProvider* health;
  QwenBf16MonotonicClock* clock;
  QwenBf16PollWaiter* waiter;
};

struct QwenBf16SynchronousBackendArenas final {
  QwenBf16StepDeviceOwners device;
  CudaCopyEndpoint pinned_staging;
  CudaCopyEndpoint device_staging;
  CudaCopyEndpoint sampled_token;
  CudaCopyEndpoint device_error;
  CudaCopyEndpoint pinned_result;
  std::span<std::byte> pinned_staging_backing;
  std::span<std::byte> pinned_result_backing;
};

struct QwenBf16SynchronousBackendIdentity final {
  std::uint64_t epoch;
  std::uint64_t first_request_generation;
  std::uint64_t first_event_generation;
  std::uint64_t first_plan_id;
  std::uint64_t timeout_ns;
  std::uintptr_t context_identity;
  DriverStreamHandle stream;
  DriverEventHandle event;
  std::int32_t owning_rank;
  std::uint32_t slot_count;
  float rms_epsilon;
  float attention_scale;
};

enum class QwenBf16SynchronousBackendState : std::uint8_t {
  kReady,
  kRunning,
  kPoisoned,
};

class QwenBf16SynchronousBackend final
    : public QwenBf16SequenceBackend,
      public QwenBf16CompletionIdentityProvider {
 public:
  static Result<QwenBf16SynchronousBackend> Create(
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> functions,
      const QwenBf16WeightResourceSet& weights,
      QwenBf16SynchronousBackendArenas arenas,
      QwenBf16SynchronousBackendIdentity identity,
      QwenBf16SynchronousBackendDrivers drivers);

  QwenBf16SynchronousBackend(const QwenBf16SynchronousBackend&) = delete;
  QwenBf16SynchronousBackend& operator=(
      const QwenBf16SynchronousBackend&) = delete;
  QwenBf16SynchronousBackend(QwenBf16SynchronousBackend&&) noexcept = default;
  QwenBf16SynchronousBackend& operator=(
      QwenBf16SynchronousBackend&&) = delete;

  Result<std::int64_t> execute_and_read_token(
      std::span<const std::int64_t> tokens, std::uint64_t first_position,
      const QwenKvBlockTable& block_table,
      const QwenKvAppendPlan& append_plan) override;
  Result<QwenKvCompletionEvent> last_completion_event() const override;

  [[nodiscard]] QwenBf16SynchronousBackendState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint64_t next_request_generation() const noexcept {
    return next_request_generation_;
  }
  [[nodiscard]] std::uint64_t next_event_generation() const noexcept {
    return next_event_generation_;
  }
  [[nodiscard]] std::uint64_t next_plan_id() const noexcept {
    return next_plan_id_;
  }

 private:
  QwenBf16SynchronousBackend(
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> functions,
      const QwenBf16WeightResourceSet& weights,
      QwenBf16SynchronousBackendArenas arenas,
      QwenBf16SynchronousBackendIdentity identity,
      QwenBf16SynchronousBackendDrivers drivers)
      : commands_(&commands), functions_(functions), weights_(&weights),
        arenas_(arenas), identity_(identity), drivers_(drivers),
        next_request_generation_(identity.first_request_generation),
        next_event_generation_(identity.first_event_generation),
        next_plan_id_(identity.first_plan_id) {}

  const QwenBf16CommandBuffer* commands_;
  std::span<const ResolvedKernelFunction> functions_;
  const QwenBf16WeightResourceSet* weights_;
  QwenBf16SynchronousBackendArenas arenas_;
  QwenBf16SynchronousBackendIdentity identity_;
  QwenBf16SynchronousBackendDrivers drivers_;
  std::uint64_t next_request_generation_;
  std::uint64_t next_event_generation_;
  std::uint64_t next_plan_id_;
  QwenKvCompletionEvent last_completion_event_{};
  QwenBf16SynchronousBackendState state_ =
      QwenBf16SynchronousBackendState::kReady;
};

}  // namespace pih
