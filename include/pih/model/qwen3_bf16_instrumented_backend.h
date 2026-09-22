#pragma once

#include <cstdint>
#include <span>

#include "pih/model/qwen3_bf16_step_health.h"
#include "pih/model/qwen3_bf16_tap_fixture_execution.h"

namespace pih {

struct QwenBf16InstrumentedBackendArenas final {
  QwenBf16StepDeviceOwners device;
  CudaCopyEndpoint pinned_staging;
  CudaCopyEndpoint device_staging;
  CudaCopyEndpoint sampled_token;
  CudaCopyEndpoint device_error;
  CudaCopyEndpoint pinned_result;
  std::span<std::byte> pinned_staging_backing;
  std::span<std::byte> pinned_result_backing;
};

struct QwenBf16InstrumentedBackendIdentity final {
  std::uint64_t epoch;
  std::uint64_t first_request_generation;
  std::uint64_t first_event_generation;
  std::uint64_t first_plan_id;
  std::uint64_t timeout_ns;
  std::uintptr_t context_identity;
  DriverStreamHandle execution_stream;
  DriverStreamHandle diagnostic_stream;
  DriverEventHandle diagnostic_event;
  std::int32_t owning_rank;
  std::int32_t numa_node;
  std::uint32_t slot_count;
  float rms_epsilon;
  float attention_scale;
  std::uint64_t source_owner_id_base;
  std::uint64_t device_tap_owner_id;
  std::uint64_t pinned_tap_owner_id;
};

struct QwenBf16InstrumentedBackendDrivers final {
  Allocator* device_allocator;
  RegisteredPinnedAllocator* pinned_allocator;
  PinnedPlacementVerifier* placement;
  TypedCopyDriver* copy;
  QwenBf16DeviceErrorClearDriver* clear;
  KernelLaunchDriver* kernels;
  QwenBf16LinearExecutionDriver* linears;
  CompletionEventDriver* events;
  QwenBf16StepHealthProvider* health;
  QwenBf16MonotonicClock* clock;
  QwenBf16PollWaiter* waiter;
};

enum class QwenBf16InstrumentedBackendState : std::uint8_t {
  kReady = 0,
  kRunning,
  kPoisoned,
};

class QwenBf16InstrumentedBackend final
    : public QwenBf16TapFixtureBackend {
 public:
  static Result<QwenBf16InstrumentedBackend> Create(
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> functions,
      const QwenBf16WeightResourceSet& weights,
      QwenBf16InstrumentedBackendArenas arenas,
      QwenBf16InstrumentedBackendIdentity identity,
      QwenBf16InstrumentedBackendDrivers drivers);

  Result<QwenBf16TapFixtureExecutionReceipt> execute_and_capture(
      const QwenBf16TapFixtureInput& input,
      const QwenNumericalTapPlan& taps,
      const QwenKvBlockTable& block_table,
      const QwenKvAppendPlan& append_plan,
      std::span<const QwenKvSlotState> projected_slot_states,
      std::uint64_t fixture_generation) override;

  [[nodiscard]] QwenBf16InstrumentedBackendState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint64_t tap_device_peak_bytes() const noexcept {
    return tap_device_peak_bytes_;
  }
  [[nodiscard]] std::uint64_t tap_pinned_peak_bytes() const noexcept {
    return tap_pinned_peak_bytes_;
  }
  [[nodiscard]] std::uint64_t next_event_generation() const noexcept {
    return next_event_generation_;
  }
  [[nodiscard]] std::uint64_t next_plan_id() const noexcept {
    return next_plan_id_;
  }

 private:
  QwenBf16InstrumentedBackend(
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> functions,
      const QwenBf16WeightResourceSet& weights,
      QwenBf16InstrumentedBackendArenas arenas,
      QwenBf16InstrumentedBackendIdentity identity,
      QwenBf16InstrumentedBackendDrivers drivers,
      QwenBf16TapSnapshotEvidenceProvider snapshot_evidence)
      : commands_(&commands), functions_(functions), weights_(&weights),
        arenas_(arenas), identity_(identity), drivers_(drivers),
        snapshot_evidence_(std::move(snapshot_evidence)),
        next_request_generation_(identity.first_request_generation),
        next_event_generation_(identity.first_event_generation),
        next_plan_id_(identity.first_plan_id) {}

  const QwenBf16CommandBuffer* commands_;
  std::span<const ResolvedKernelFunction> functions_;
  const QwenBf16WeightResourceSet* weights_;
  QwenBf16InstrumentedBackendArenas arenas_;
  QwenBf16InstrumentedBackendIdentity identity_;
  QwenBf16InstrumentedBackendDrivers drivers_;
  QwenBf16TapSnapshotEvidenceProvider snapshot_evidence_;
  std::uint64_t next_request_generation_;
  std::uint64_t next_event_generation_;
  std::uint64_t next_plan_id_;
  QwenBf16InstrumentedBackendState state_ =
      QwenBf16InstrumentedBackendState::kReady;
  std::uint64_t tap_device_peak_bytes_ = 0;
  std::uint64_t tap_pinned_peak_bytes_ = 0;
};

}  // namespace pih
