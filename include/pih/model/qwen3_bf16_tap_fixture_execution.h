#pragma once

#include <cstdint>

#include "pih/model/qwen3_bf16_synchronous_backend.h"
#include "pih/model/qwen3_bf16_tap_fixture_preparation.h"
#include "pih/model/qwen3_bf16_tap_suite_executor.h"

namespace pih {

struct QwenBf16TapFixtureExecutionDrivers final {
  TypedCopyDriver* copy;
  QwenBf16DeviceErrorClearDriver* clear;
  KernelLaunchDriver* kernels;
  QwenBf16LinearExecutionDriver* linears;
  CompletionEventDriver* events;
  CompletionEvidenceProvider* snapshot_evidence;
  QwenBf16StepHealthProvider* final_health;
  QwenBf16MonotonicClock* clock;
  QwenBf16PollWaiter* waiter;
};

struct QwenBf16TapFixtureExecutionIdentity final {
  DriverStreamHandle execution_stream;
  DriverEventHandle diagnostic_event;
  std::uint64_t timeout_ns;
};

enum class QwenBf16TapFixtureExecutionState : std::uint8_t {
  kPrepared = 0,
  kRunning,
  kSealed,
  kPoisoned,
};

class QwenBf16TapFixtureExecution final : private CompletionEvidenceProvider {
 public:
  static Result<QwenBf16TapFixtureExecution> Create(
      QwenBf16StepUpload upload,
      QwenBf16StepReadback final_readback,
      std::span<std::byte> pinned_result_backing,
      QwenBf16InstrumentedStepCompute& compute,
      QwenBf16TapFixturePreparation preparation,
      QwenBf16TapFixtureExecutionIdentity identity,
      QwenBf16TapFixtureExecutionDrivers drivers);

  QwenBf16TapFixtureExecution(const QwenBf16TapFixtureExecution&) = delete;
  QwenBf16TapFixtureExecution& operator=(
      const QwenBf16TapFixtureExecution&) = delete;
  QwenBf16TapFixtureExecution(QwenBf16TapFixtureExecution&&) noexcept = default;
  QwenBf16TapFixtureExecution& operator=(
      QwenBf16TapFixtureExecution&&) = delete;

  Result<QwenBf16TapFixtureExecutionReceipt> run();
  [[nodiscard]] QwenBf16TapFixtureExecutionState state() const noexcept {
    return state_;
  }

 private:
  QwenBf16TapFixtureExecution(
      QwenBf16StepUpload upload,
      QwenBf16StepReadback final_readback,
      std::span<std::byte> pinned_result_backing,
      QwenBf16InstrumentedStepCompute& compute,
      QwenBf16TapFixturePreparation preparation,
      QwenBf16TapFixtureExecutionIdentity identity,
      QwenBf16TapFixtureExecutionDrivers drivers)
      : upload_(std::move(upload)), final_readback_(std::move(final_readback)),
        pinned_result_backing_(pinned_result_backing), compute_(&compute),
        preparation_(std::move(preparation)), identity_(identity),
        drivers_(drivers) {}
  Status wait_snapshot(std::uint64_t deadline_ns);
  Status wait_host(std::uint64_t deadline_ns);
  Result<CompletionPublicationEvidence> collect() override;

  QwenBf16StepUpload upload_;
  QwenBf16StepReadback final_readback_;
  std::span<std::byte> pinned_result_backing_;
  QwenBf16InstrumentedStepCompute* compute_;
  QwenBf16TapFixturePreparation preparation_;
  QwenBf16TapFixtureExecutionIdentity identity_;
  QwenBf16TapFixtureExecutionDrivers drivers_;
  QwenBf16TapFixtureExecutionState state_ =
      QwenBf16TapFixtureExecutionState::kPrepared;
};

}  // namespace pih
