#include "pih/model/qwen3_bf16_tap_fixture_execution.h"

#include "pih/core/checked_math.h"

#include <cstring>

namespace pih {

Result<QwenBf16TapFixtureExecution> QwenBf16TapFixtureExecution::Create(
    QwenBf16StepUpload upload,
    QwenBf16StepReadback final_readback,
    std::span<std::byte> pinned_result_backing,
    QwenBf16InstrumentedStepCompute& compute,
    QwenBf16TapFixturePreparation preparation,
    QwenBf16TapFixtureExecutionIdentity identity,
    QwenBf16TapFixtureExecutionDrivers drivers) {
  const bool valid_drivers =
      drivers.copy != nullptr && drivers.clear != nullptr &&
      drivers.kernels != nullptr && drivers.linears != nullptr &&
      drivers.events != nullptr && drivers.snapshot_evidence != nullptr &&
      drivers.final_health != nullptr &&
      drivers.clock != nullptr && drivers.waiter != nullptr;
  if (!valid_drivers || identity.execution_stream == 0 ||
      identity.diagnostic_event == 0 || identity.timeout_ns == 0 ||
      upload.state() != QwenBf16StepUploadState::kPrepared ||
      upload.stream() != identity.execution_stream ||
      final_readback.state() != QwenBf16StepReadbackState::kPrepared ||
      final_readback.stream() != preparation.pipeline().diagnostic_stream() ||
      final_readback.event_generation() !=
          preparation.pipeline().host_event_generation() ||
      pinned_result_backing.size() != QwenBf16StepResultLayout::kTotalBytes) {
    return Status::InvalidArgument(
        "Qwen tap fixture execution configuration is invalid");
  }
  const Status initialized =
      QwenBf16StepResultLayout::initialize(pinned_result_backing);
  if (!initialized.ok()) return initialized;
  return QwenBf16TapFixtureExecution(
      std::move(upload), std::move(final_readback), pinned_result_backing,
      compute, std::move(preparation), identity, drivers);
}

Status QwenBf16TapFixtureExecution::wait_snapshot(
    std::uint64_t deadline_ns) {
  for (;;) {
    auto status = preparation_.pipeline().poll_snapshot(
        *drivers_.events, *drivers_.snapshot_evidence);
    if (status.ok()) return status;
    if (status.code() != StatusCode::kUnavailable) return status;
    auto now = drivers_.clock->now_ns();
    if (!now.ok()) return now.status();
    if (*now >= deadline_ns) return preparation_.pipeline().expire(*now);
    status = drivers_.waiter->wait();
    if (!status.ok()) return status;
  }
}

Status QwenBf16TapFixtureExecution::wait_host(
    std::uint64_t deadline_ns) {
  for (;;) {
    auto status = preparation_.pipeline().poll_host(
        *drivers_.events, *this);
    if (status.ok()) return status;
    if (status.code() != StatusCode::kUnavailable) return status;
    auto now = drivers_.clock->now_ns();
    if (!now.ok()) return now.status();
    if (*now >= deadline_ns) return preparation_.pipeline().expire(*now);
    status = drivers_.waiter->wait();
    if (!status.ok()) return status;
  }
}

Result<QwenBf16TapFixtureExecutionReceipt>
QwenBf16TapFixtureExecution::run() {
  if (state_ != QwenBf16TapFixtureExecutionState::kPrepared) {
    return Status::FailedPrecondition("Qwen tap fixture execution is not ready");
  }
  state_ = QwenBf16TapFixtureExecutionState::kRunning;
  const auto fail = [this](Status status)
      -> Result<QwenBf16TapFixtureExecutionReceipt> {
    state_ = QwenBf16TapFixtureExecutionState::kPoisoned;
    return status;
  };
  auto submit_ns = drivers_.clock->now_ns();
  if (!submit_ns.ok()) return fail(submit_ns.status());
  auto snapshot_deadline = checked_add_u64(*submit_ns, identity_.timeout_ns);
  if (!snapshot_deadline.ok()) return fail(snapshot_deadline.status());
  Status status = upload_.submit(*drivers_.copy);
  if (!status.ok()) return fail(status);
  auto snapshots = preparation_.pipeline().inline_snapshot_driver(
      *drivers_.copy);
  if (!snapshots.ok()) return fail(snapshots.status());
  status = compute_->submit_instrumented_and_record(
      *drivers_.clear, *drivers_.kernels, *drivers_.linears, *snapshots,
      preparation_.bindings(), preparation_.pipeline(), *drivers_.events,
      identity_.execution_stream, *submit_ns, *snapshot_deadline);
  if (!status.ok()) return fail(status);
  status = wait_snapshot(*snapshot_deadline);
  if (!status.ok()) return fail(status);
  auto host_submit_ns = drivers_.clock->now_ns();
  if (!host_submit_ns.ok()) return fail(host_submit_ns.status());
  auto host_deadline = checked_add_u64(*host_submit_ns, identity_.timeout_ns);
  if (!host_deadline.ok()) return fail(host_deadline.status());
  status = final_readback_.submit(*drivers_.copy);
  if (!status.ok()) return fail(status);
  status = preparation_.pipeline().submit_host_and_record(
      *drivers_.copy, *drivers_.events, *host_submit_ns, *host_deadline);
  if (!status.ok()) return fail(status);
  status = wait_host(*host_deadline);
  if (!status.ok()) return fail(status);
  auto receipt = preparation_.pipeline().seal(preparation_.arenas());
  if (!receipt.ok()) return fail(receipt.status());
  auto sampled_token = QwenBf16StepResultLayout::parse(
      pinned_result_backing_, true);
  if (!sampled_token.ok()) return fail(sampled_token.status());
  state_ = QwenBf16TapFixtureExecutionState::kSealed;
  return QwenBf16TapFixtureExecutionReceipt{
      *receipt,
      {identity_.diagnostic_event,
       preparation_.pipeline().host_event_generation()},
      *sampled_token};
}

Result<CompletionPublicationEvidence>
QwenBf16TapFixtureExecution::collect() {
  auto health = drivers_.final_health->collect();
  if (!health.ok()) return health.status();
  std::uint32_t device_error = UINT32_MAX;
  std::memcpy(&device_error,
              pinned_result_backing_.data() +
                  QwenBf16StepResultLayout::device_error().offset_bytes,
              sizeof(device_error));
  return CompletionPublicationEvidence{
      health->submit_thread_last_error_clean, device_error,
      health->engine_poisoned};
}

}  // namespace pih
