#include "pih/model/qwen3_bf16_packed_synchronous_backend.h"

#include <cmath>
#include <limits>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

bool endpoint_matches(const CudaCopyEndpoint& endpoint, std::uintptr_t base,
                      std::uint64_t bytes, CudaCopyMemoryType type,
                      std::int32_t rank) {
  return endpoint.allocation_base == base && endpoint.allocation_bytes >= bytes &&
         endpoint.offset == 0 && endpoint.owner_id != 0 &&
         endpoint.generation != 0 && endpoint.memory_type == type &&
         endpoint.rank == static_cast<std::uint32_t>(rank) &&
         endpoint.device_or_numa >= 0;
}

}  // namespace

Result<QwenBf16PackedSynchronousBackend>
QwenBf16PackedSynchronousBackend::Create(
    const QwenBf16CommandBuffer& commands,
    std::span<const ResolvedKernelFunction> legacy_functions,
    std::span<const ResolvedKernelFunction> packed_functions,
    const QwenBf16WeightResourceSet& weights,
    QwenBf16PackedBackendArenas arenas,
    QwenBf16PackedBackendIdentity identity,
    QwenBf16PackedBackendDrivers drivers) {
  auto result_layout = QwenBf16PackedResultLayout::Create(identity.sample_capacity);
  if (!result_layout.ok() || !drivers.copy || !drivers.clear ||
      !drivers.kernels || !drivers.linears || !drivers.events ||
      !drivers.health || !drivers.clock || !drivers.waiter ||
      identity.epoch == 0 || identity.first_request_generation == 0 ||
      identity.first_event_generation == 0 || identity.first_plan_id == 0 ||
      identity.timeout_ns == 0 || identity.context_identity == 0 ||
      identity.stream == 0 || identity.event == 0 || identity.owning_rank < 0 ||
      identity.slot_count == 0 || !std::isfinite(identity.rms_epsilon) ||
      identity.rms_epsilon <= 0 || !std::isfinite(identity.attention_scale) ||
      identity.attention_scale <= 0 ||
      legacy_functions.size() != QwenBf16KernelBundle::kExecutionPrimitiveCount ||
      packed_functions.size() != QwenBf16KernelBundle::kPackedPrimitiveCount ||
      weights.owning_rank() != identity.owning_rank ||
      drivers.copy->context_identity() != identity.context_identity ||
      arenas.pinned_staging_backing.empty() ||
      arenas.pinned_result_backing.size() != result_layout->total_bytes())
    return Status::InvalidArgument("packed backend configuration is invalid");
  const auto staging_base = reinterpret_cast<std::uintptr_t>(
      arenas.pinned_staging_backing.data());
  const auto result_base = reinterpret_cast<std::uintptr_t>(
      arenas.pinned_result_backing.data());
  const auto sampled_bytes = result_layout->device_error().offset_bytes;
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
      !endpoint_matches(arenas.sampled_tokens, arenas.device.sampled_token.base,
                        sampled_bytes, CudaCopyMemoryType::kDevice,
                        identity.owning_rank) ||
      !endpoint_matches(arenas.device_error, arenas.device.device_error.base,
                        sizeof(std::uint32_t), CudaCopyMemoryType::kDevice,
                        identity.owning_rank))
    return Status::InvalidArgument("packed backend copy arenas do not match");
  return QwenBf16PackedSynchronousBackend(
      commands, legacy_functions, packed_functions, weights, arenas, identity,
      drivers);
}

Result<QwenBf16PackedBatchExecutionView>
QwenBf16PackedSynchronousBackend::execute_packed(
    const PackedTokenPlan& plan, PackedTokenMetadataView metadata,
    QwenBf16PackedKvMetadataView kv_metadata,
    std::span<const QwenBf16PackedKvBinding> bindings,
    std::span<const QwenKvAppendPlan> append_plans,
    std::span<const Qwen3SamplingDescriptor> sampling) {
  if (state_ != QwenBf16PackedBackendState::kReady)
    return Status::FailedPrecondition("packed backend is not ready");
  if (bindings.size() != plan.sequence_count() ||
      append_plans.size() != plan.sequence_count() ||
      sampling.size() != plan.sequence_count())
    return Status::InvalidArgument("packed backend sequence inputs drifted");
  if (request_generation_ == UINT64_MAX || event_generation_ == UINT64_MAX ||
      plan_id_ > UINT64_MAX - 14) {
    state_ = QwenBf16PackedBackendState::kPoisoned;
    return Status::ResourceExhausted("packed backend identity exhausted");
  }
  state_ = QwenBf16PackedBackendState::kRunning;
  const auto fail = [this](Status s) -> Result<QwenBf16PackedBatchExecutionView> {
    state_ = QwenBf16PackedBackendState::kPoisoned; return s;
  };
  const auto request_generation = request_generation_++;
  const auto event_generation = event_generation_++;
  const auto upload_plan = plan_id_;
  const auto readback_plan = plan_id_ + 12;
  plan_id_ += 14;
  auto owners = arenas_.device;
  owners.device_error.generation = request_generation;
  auto built = QwenBf16PackedStepBuilder::Create(
      plan, metadata, kv_metadata, request_generation, identity_.owning_rank,
      identity_.slot_count, identity_.rms_epsilon, identity_.attention_scale,
      *commands_, legacy_, packed_, *weights_, owners,
      arenas_.pinned_staging_backing, sampling);
  if (!built.ok()) return fail(built.status());
  auto device_staging = arenas_.device_staging;
  device_staging.generation = owners.step_staging.generation;
  auto upload = QwenBf16PackedStepUpload::Create(
      built->staging_layout(), arenas_.pinned_staging, device_staging,
      identity_.context_identity, identity_.stream, event_generation,
      upload_plan);
  if (!upload.ok()) return fail(upload.status());
  auto layout = QwenBf16PackedResultLayout::Create(identity_.sample_capacity);
  if (!layout.ok()) return fail(layout.status());
  auto sampled = arenas_.sampled_tokens;
  sampled.generation = owners.sampled_token.generation;
  auto error = arenas_.device_error;
  error.generation = request_generation;
  const auto samples = static_cast<std::uint32_t>(metadata.sample_row_index.size());
  auto readback = QwenBf16PackedReadback::CreateSampling(
      *layout, samples, sampled, error, arenas_.pinned_result,
      identity_.context_identity, identity_.stream, event_generation,
      readback_plan);
  if (!readback.ok()) return fail(readback.status());
  auto slot = CompletionEventSlot::Create(identity_.event, identity_.context_identity);
  if (!slot.ok()) return fail(slot.status());
  auto now = drivers_.clock->now_ns();
  if (!now.ok()) return fail(now.status());
  auto deadline = checked_add_u64(*now, identity_.timeout_ns);
  if (!deadline.ok()) return fail(deadline.status());
  std::uint64_t committed = 0;
  for (const auto& append : append_plans)
    committed = (std::max)(committed, static_cast<std::uint64_t>(append.target_committed_tokens));
  auto frontier = CudaCompletionFrontier::Create(
      {identity_.epoch, static_cast<std::uint32_t>(identity_.owning_rank),
       request_generation, plan.phase() == PackedTokenPhase::kPrefill
                               ? CudaCompletionPhase::kPrefill
                               : CudaCompletionPhase::kDecode,
       committed},
      event_generation, *now, *deadline);
  if (!frontier.ok()) return fail(frontier.status());
  auto transaction = QwenBf16PackedStepTransaction::Create(
      std::move(*upload), built->compute(), std::move(*readback), *layout,
      samples, std::move(*slot), std::move(*frontier),
      arenas_.pinned_result_backing, identity_.stream, event_generation,
      *drivers_.health);
  if (!transaction.ok()) return fail(transaction.status());
  auto submitted = transaction->submit(*drivers_.copy, *drivers_.clear,
      *drivers_.kernels, *drivers_.linears, *drivers_.events);
  if (!submitted.ok()) return fail(submitted);
  for (;;) {
    auto time = drivers_.clock->now_ns();
    if (!time.ok()) return fail(time.status());
    auto expiry = transaction->expire(*time);
    if (expiry.code() != StatusCode::kUnavailable) return fail(expiry);
    auto receipts = transaction->poll_sampling(*drivers_.events);
    if (receipts.ok()) {
      auto released = transaction->release_completion();
      if (!released.ok()) return fail(released);
      sampling_receipts_ = std::move(*receipts);
      sampled_tokens_.clear();
      sampled_tokens_.reserve(sampling_receipts_.size());
      for (const auto& receipt : sampling_receipts_)
        sampled_tokens_.push_back(receipt.token_id);
      state_ = QwenBf16PackedBackendState::kReady;
      return QwenBf16PackedBatchExecutionView{
          sampled_tokens_, {identity_.event, event_generation},
          sampling_receipts_};
    }
    if (receipts.status().code() != StatusCode::kUnavailable)
      return fail(receipts.status());
    auto waited = drivers_.waiter->wait();
    if (!waited.ok()) return fail(waited);
  }
}

}  // namespace pih
