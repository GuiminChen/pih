#include "pih/model/qwen3_bf16_instrumented_backend.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pih {

Result<QwenBf16InstrumentedBackend>
QwenBf16InstrumentedBackend::Create(
    const QwenBf16CommandBuffer& commands,
    std::span<const ResolvedKernelFunction> functions,
    const QwenBf16WeightResourceSet& weights,
    QwenBf16InstrumentedBackendArenas arenas,
    QwenBf16InstrumentedBackendIdentity identity,
    QwenBf16InstrumentedBackendDrivers drivers) {
  const bool valid_drivers =
      drivers.device_allocator != nullptr &&
      drivers.pinned_allocator != nullptr && drivers.placement != nullptr &&
      drivers.copy != nullptr && drivers.clear != nullptr &&
      drivers.kernels != nullptr && drivers.linears != nullptr &&
      drivers.events != nullptr && drivers.health != nullptr &&
      drivers.clock != nullptr && drivers.waiter != nullptr;
  if (!valid_drivers || identity.epoch == 0 ||
      identity.first_request_generation == 0 ||
      identity.first_event_generation == 0 || identity.first_plan_id == 0 ||
      identity.timeout_ns == 0 || identity.context_identity == 0 ||
      identity.execution_stream == 0 || identity.diagnostic_stream == 0 ||
      identity.diagnostic_event == 0 || identity.owning_rank < 0 ||
      identity.numa_node < 0 || identity.slot_count == 0 ||
      identity.slot_count > QwenKvSlotPool::kMaximumSlots ||
      !std::isfinite(identity.rms_epsilon) || identity.rms_epsilon <= 0.0F ||
      !std::isfinite(identity.attention_scale) ||
      identity.attention_scale <= 0.0F ||
      identity.source_owner_id_base == 0 ||
      identity.device_tap_owner_id == 0 || identity.pinned_tap_owner_id == 0 ||
      weights.owning_rank() != identity.owning_rank ||
      drivers.copy->context_identity() != identity.context_identity ||
      arenas.pinned_staging_backing.empty() ||
      arenas.pinned_result_backing.size() !=
          QwenBf16StepResultLayout::kTotalBytes) {
    return Status::InvalidArgument(
        "Qwen instrumented backend configuration is invalid");
  }
  auto snapshot_evidence =
      QwenBf16TapSnapshotEvidenceProvider::Create(*drivers.health);
  if (!snapshot_evidence.ok()) return snapshot_evidence.status();
  return QwenBf16InstrumentedBackend(
      commands, functions, weights, arenas, identity, drivers,
      std::move(*snapshot_evidence));
}

Result<QwenBf16TapFixtureExecutionReceipt>
QwenBf16InstrumentedBackend::execute_and_capture(
    const QwenBf16TapFixtureInput& input,
    const QwenNumericalTapPlan& taps,
    const QwenKvBlockTable& block_table,
    const QwenKvAppendPlan& append_plan,
    std::span<const QwenKvSlotState> projected_slot_states,
    std::uint64_t fixture_generation) {
  if (state_ != QwenBf16InstrumentedBackendState::kReady ||
      fixture_generation != next_request_generation_ ||
      next_event_generation_ > std::numeric_limits<std::uint64_t>::max() - 1 ||
      next_plan_id_ > std::numeric_limits<std::uint64_t>::max() - 8) {
    state_ = QwenBf16InstrumentedBackendState::kPoisoned;
    return Status::FailedPrecondition(
        "Qwen instrumented backend fixture identity is invalid");
  }
  state_ = QwenBf16InstrumentedBackendState::kRunning;
  const auto fail = [this](Status status)
      -> Result<QwenBf16TapFixtureExecutionReceipt> {
    state_ = QwenBf16InstrumentedBackendState::kPoisoned;
    return status;
  };
  auto owners = arenas_.device;
  owners.device_error.generation = fixture_generation;
  auto built = QwenBf16StepBuilder::Create(
      input.tokens, input.first_position, block_table, append_plan,
      fixture_generation, identity_.owning_rank, identity_.slot_count,
      identity_.rms_epsilon, identity_.attention_scale, *commands_, functions_,
      *weights_, owners, arenas_.pinned_staging_backing);
  if (!built.ok()) return fail(built.status());

  auto device_staging = arenas_.device_staging;
  device_staging.generation = owners.step_staging.generation;
  auto upload = QwenBf16StepUpload::Create(
      built->staging_layout(), arenas_.pinned_staging, device_staging,
      identity_.context_identity, identity_.execution_stream,
      next_event_generation_, next_plan_id_);
  if (!upload.ok()) return fail(upload.status());
  const QwenBf16TapTransferIdentity transfer{
      fixture_generation, fixture_generation, next_plan_id_ + 5,
      next_plan_id_ + 6, next_event_generation_, next_event_generation_ + 1,
      identity_.context_identity, identity_.execution_stream,
      identity_.diagnostic_stream, identity_.device_tap_owner_id,
      identity_.pinned_tap_owner_id,
      static_cast<std::uint32_t>(identity_.owning_rank)};
  auto preparation = QwenBf16TapFixturePreparation::Create(
      taps, *commands_, built->resources(), block_table, projected_slot_states,
      input.first_position, identity_.source_owner_id_base,
      *drivers_.device_allocator, *drivers_.pinned_allocator,
      *drivers_.placement, identity_.numa_node, transfer,
      identity_.diagnostic_event, identity_.epoch);
  if (!preparation.ok()) return fail(preparation.status());
  tap_device_peak_bytes_ =
      std::max(tap_device_peak_bytes_, preparation->arenas().arena_bytes());
  tap_pinned_peak_bytes_ =
      std::max(tap_pinned_peak_bytes_, preparation->arenas().arena_bytes());

  auto sampled = arenas_.sampled_token;
  sampled.generation = owners.sampled_token.generation;
  auto device_error = arenas_.device_error;
  device_error.generation = fixture_generation;
  auto readback = QwenBf16StepReadback::Create(
      sampled, device_error, arenas_.pinned_result, identity_.context_identity,
      identity_.diagnostic_stream, next_event_generation_ + 1,
      next_plan_id_ + 7);
  if (!readback.ok()) return fail(readback.status());
  auto execution = QwenBf16TapFixtureExecution::Create(
      std::move(*upload), std::move(*readback),
      arenas_.pinned_result_backing, built->compute(),
      std::move(*preparation),
      {identity_.execution_stream, identity_.diagnostic_event,
       identity_.timeout_ns},
      {drivers_.copy, drivers_.clear, drivers_.kernels, drivers_.linears,
       drivers_.events, &snapshot_evidence_, drivers_.health,
       drivers_.clock, drivers_.waiter});
  if (!execution.ok()) return fail(execution.status());
  ++next_request_generation_;
  next_event_generation_ += 2;
  next_plan_id_ += 9;
  auto receipt = execution->run();
  if (!receipt.ok()) return fail(receipt.status());
  state_ = QwenBf16InstrumentedBackendState::kReady;
  return receipt;
}

}  // namespace pih
