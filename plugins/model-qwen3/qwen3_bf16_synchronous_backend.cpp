#include "pih/model/qwen3_bf16_synchronous_backend.h"

#include <cmath>
#include <limits>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

bool endpoint_matches(const CudaCopyEndpoint& endpoint,
                      std::uintptr_t base, std::uint64_t bytes,
                      CudaCopyMemoryType type, std::int32_t rank) {
  return endpoint.allocation_base == base && endpoint.allocation_bytes >= bytes &&
         endpoint.offset == 0 && endpoint.owner_id != 0 &&
         endpoint.generation != 0 && endpoint.memory_type == type &&
         endpoint.rank == static_cast<std::uint32_t>(rank) &&
         endpoint.device_or_numa >= 0;
}

}  // namespace

Result<QwenBf16SynchronousBackend> QwenBf16SynchronousBackend::Create(
    const QwenBf16CommandBuffer& commands,
    std::span<const ResolvedKernelFunction> functions,
    const QwenBf16WeightResourceSet& weights,
    QwenBf16SynchronousBackendArenas arenas,
    QwenBf16SynchronousBackendIdentity identity,
    QwenBf16SynchronousBackendDrivers drivers) {
  const bool drivers_valid =
      drivers.copy != nullptr && drivers.clear != nullptr &&
      drivers.kernels != nullptr && drivers.linears != nullptr &&
      drivers.events != nullptr && drivers.health != nullptr &&
      drivers.clock != nullptr && drivers.waiter != nullptr;
  if (!drivers_valid || identity.epoch == 0 ||
      identity.first_request_generation == 0 ||
      identity.first_event_generation == 0 || identity.first_plan_id == 0 ||
      identity.timeout_ns == 0 || identity.context_identity == 0 ||
      identity.stream == 0 || identity.event == 0 || identity.owning_rank < 0 ||
      identity.slot_count == 0 ||
      identity.slot_count > QwenKvSlotPool::kMaximumSlots ||
      !std::isfinite(identity.rms_epsilon) || identity.rms_epsilon <= 0.0F ||
      !std::isfinite(identity.attention_scale) ||
      identity.attention_scale <= 0.0F ||
      functions.size() != QwenBf16KernelBundle::kExecutionPrimitiveCount ||
      weights.owning_rank() != identity.owning_rank ||
      drivers.copy->context_identity() != identity.context_identity ||
      arenas.pinned_result_backing.size() !=
          QwenBf16StepResultLayout::kTotalBytes ||
      arenas.pinned_staging_backing.empty()) {
    return Status::InvalidArgument(
        "Qwen synchronous backend configuration is invalid");
  }
  const auto staging_base = reinterpret_cast<std::uintptr_t>(
      arenas.pinned_staging_backing.data());
  const auto result_base = reinterpret_cast<std::uintptr_t>(
      arenas.pinned_result_backing.data());
  if (!endpoint_matches(arenas.pinned_staging, staging_base,
                        arenas.pinned_staging_backing.size(),
                        CudaCopyMemoryType::kRegisteredPinnedHost,
                        identity.owning_rank) ||
      !endpoint_matches(arenas.pinned_result, result_base,
                        arenas.pinned_result_backing.size(),
                        CudaCopyMemoryType::kRegisteredPinnedHost,
                        identity.owning_rank) ||
      !endpoint_matches(arenas.device_staging, arenas.device.step_staging.base,
                        arenas.device.step_staging.bytes,
                        CudaCopyMemoryType::kDevice, identity.owning_rank) ||
      !endpoint_matches(arenas.sampled_token, arenas.device.sampled_token.base,
                        sizeof(std::int64_t), CudaCopyMemoryType::kDevice,
                        identity.owning_rank) ||
      !endpoint_matches(arenas.device_error, arenas.device.device_error.base,
                        sizeof(std::uint32_t), CudaCopyMemoryType::kDevice,
                        identity.owning_rank)) {
    return Status::InvalidArgument(
        "Qwen synchronous backend copy arenas do not match their owners");
  }
  return QwenBf16SynchronousBackend(commands, functions, weights, arenas,
                                     identity, drivers);
}

Result<std::int64_t> QwenBf16SynchronousBackend::execute_and_read_token(
    std::span<const std::int64_t> tokens, std::uint64_t first_position,
    const QwenKvBlockTable& block_table,
    const QwenKvAppendPlan& append_plan) {
  if (state_ != QwenBf16SynchronousBackendState::kReady) {
    return Status::FailedPrecondition(
        "Qwen synchronous backend is not ready");
  }
  if (next_request_generation_ == std::numeric_limits<std::uint64_t>::max() ||
      next_event_generation_ == std::numeric_limits<std::uint64_t>::max() ||
      next_plan_id_ > std::numeric_limits<std::uint64_t>::max() - 7) {
    state_ = QwenBf16SynchronousBackendState::kPoisoned;
    return Status::ResourceExhausted(
        "Qwen synchronous backend identity space is exhausted");
  }
  state_ = QwenBf16SynchronousBackendState::kRunning;
  const auto fail = [this](Status status) -> Result<std::int64_t> {
    state_ = QwenBf16SynchronousBackendState::kPoisoned;
    return status;
  };
  const std::uint64_t request_generation = next_request_generation_++;
  const std::uint64_t event_generation = next_event_generation_++;
  const std::uint64_t upload_plan_id = next_plan_id_;
  const std::uint64_t readback_plan_id = next_plan_id_ + 5;
  next_plan_id_ += 7;
  auto owners = arenas_.device;
  owners.device_error.generation = request_generation;
  auto built = QwenBf16StepBuilder::Create(
      tokens, first_position, block_table, append_plan, request_generation,
      identity_.owning_rank, identity_.slot_count, identity_.rms_epsilon,
      identity_.attention_scale, *commands_, functions_, *weights_, owners,
      arenas_.pinned_staging_backing);
  if (!built.ok()) return fail(built.status());

  auto device_staging = arenas_.device_staging;
  device_staging.generation = owners.step_staging.generation;
  auto upload = QwenBf16StepUpload::Create(
      built->staging_layout(), arenas_.pinned_staging, device_staging,
      identity_.context_identity, identity_.stream, event_generation,
      upload_plan_id);
  if (!upload.ok()) return fail(upload.status());
  auto sampled_token = arenas_.sampled_token;
  sampled_token.generation = owners.sampled_token.generation;
  auto device_error = arenas_.device_error;
  device_error.generation = request_generation;
  auto readback = QwenBf16StepReadback::Create(
      sampled_token, device_error, arenas_.pinned_result,
      identity_.context_identity, identity_.stream, event_generation,
      readback_plan_id);
  if (!readback.ok()) return fail(readback.status());
  auto event_slot = CompletionEventSlot::Create(
      identity_.event, identity_.context_identity);
  if (!event_slot.ok()) return fail(event_slot.status());
  auto submit_ns = drivers_.clock->now_ns();
  if (!submit_ns.ok()) return fail(submit_ns.status());
  auto deadline_ns = checked_add_u64(*submit_ns, identity_.timeout_ns);
  if (!deadline_ns.ok()) return fail(deadline_ns.status());
  const auto phase = tokens.size() == 1 && first_position != 0
                         ? CudaCompletionPhase::kDecode
                         : CudaCompletionPhase::kPrefill;
  auto frontier = CudaCompletionFrontier::Create(
      {identity_.epoch, static_cast<std::uint32_t>(identity_.owning_rank),
       request_generation, phase, append_plan.target_committed_tokens},
      event_generation, *submit_ns, *deadline_ns);
  if (!frontier.ok()) return fail(frontier.status());
  auto transaction = QwenBf16StepTransaction::Create(
      std::move(*upload), built->compute(), std::move(*readback),
      std::move(*event_slot), std::move(*frontier),
      arenas_.pinned_result_backing, identity_.stream, event_generation,
      *drivers_.health);
  if (!transaction.ok()) return fail(transaction.status());
  Status submitted = transaction->submit(
      *drivers_.copy, *drivers_.clear, *drivers_.kernels, *drivers_.linears,
      *drivers_.events);
  if (!submitted.ok()) return fail(submitted);
  for (;;) {
    auto now = drivers_.clock->now_ns();
    if (!now.ok()) return fail(now.status());
    const Status expiry = transaction->expire(*now);
    if (expiry.code() != StatusCode::kUnavailable) return fail(expiry);
    auto token = transaction->poll(*drivers_.events);
    if (token.ok()) {
      const Status released = transaction->release_completion();
      if (!released.ok()) return fail(released);
      last_completion_event_ = {identity_.event, event_generation};
      state_ = QwenBf16SynchronousBackendState::kReady;
      return token;
    }
    if (token.status().code() != StatusCode::kUnavailable) {
      return fail(token.status());
    }
    const Status waited = drivers_.waiter->wait();
    if (!waited.ok()) return fail(waited);
  }
}

Result<QwenKvCompletionEvent>
QwenBf16SynchronousBackend::last_completion_event() const {
  if (state_ != QwenBf16SynchronousBackendState::kReady ||
      last_completion_event_.handle == 0 ||
      last_completion_event_.generation == 0) {
    return Status::FailedPrecondition(
        "Qwen backend has no successfully completed event");
  }
  return last_completion_event_;
}

}  // namespace pih
