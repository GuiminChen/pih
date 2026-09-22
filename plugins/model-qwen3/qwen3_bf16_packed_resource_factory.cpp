#include "pih/model/qwen3_bf16_packed_resource_factory.h"

#include <array>
#include <vector>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Result<TensorView> arena_view(const QwenBf16DeviceArenaOwner& owner,
                              std::uint64_t offset, DType dtype,
                              std::span<const std::int64_t> shape,
                              const Device& device) {
  if (owner.base == 0 || owner.bytes == 0 || owner.generation == 0)
    return Status::InvalidArgument("packed arena owner is invalid");
  auto address = checked_add_u64(owner.base, offset);
  if (!address.ok()) return address.status();
  auto result = TensorView::Create(
      reinterpret_cast<void*>(static_cast<std::uintptr_t>(*address)), dtype,
      shape, {}, device, owner.generation);
  if (!result.ok()) return result.status();
  auto end = checked_add_u64(offset, result->byte_span());
  if (!end.ok()) return end.status();
  if (*end > owner.bytes)
    return Status::InvalidArgument("packed tensor exceeds arena owner");
  return result;
}

Result<TensorView> raw_view(const QwenBf16DeviceArenaOwner& owner,
                            QwenBf16ArenaSpan span, const Device& device) {
  if (span.size_bytes == 0 || span.size_bytes > static_cast<std::uint64_t>(INT64_MAX))
    return Status::InvalidArgument("packed raw tensor span is invalid");
  const std::array<std::int64_t, 1> shape{
      static_cast<std::int64_t>(span.size_bytes)};
  return arena_view(owner, span.offset_bytes, DType::kUInt8, shape, device);
}

Status append(std::vector<QwenBf16SlotResource>& resources,
              QwenBf16ActivationSlot slot, Result<TensorView> view) {
  if (!view.ok()) return view.status();
  resources.push_back({slot, std::move(*view)});
  return Status::Ok();
}

}  // namespace

Result<QwenBf16PackedResourceSet> QwenBf16PackedResourceFactory::Create(
    std::uint64_t request_generation, std::int32_t owning_rank,
    std::uint32_t slot_count, std::uint32_t sample_count,
    const QwenBf16ExecutionArenaLayout& execution_layout,
    const QwenBf16PackedStepStagingLayout& staging_layout,
    const QwenBf16StepDeviceOwners& owners) {
  if (request_generation == 0 || owning_rank < 0 || slot_count == 0 ||
      slot_count > QwenKvSlotPool::kMaximumSlots ||
      sample_count > staging_layout.sequence_count() ||
      execution_layout.tokens() != staging_layout.execution_bucket_tokens() ||
      execution_layout.active_logit_rows() != (std::max)(sample_count, 1U) ||
      owners.device_error.generation != request_generation) {
    return Status::InvalidArgument("packed resource factory identity is invalid");
  }
  auto device = Device::Create(DeviceType::kCuda, owning_rank);
  if (!device.ok()) return device.status();
  auto metadata = QwenBf16PackedStagingResources::Create(
      request_generation, owning_rank, staging_layout, owners.step_staging);
  if (!metadata.ok()) return metadata.status();
  const auto tokens = static_cast<std::int64_t>(execution_layout.tokens());
  const auto samples = static_cast<std::int64_t>((std::max)(sample_count, 1U));
  const std::array<std::int64_t, 2> hidden{tokens, 1024};
  const std::array<std::int64_t, 3> query{tokens, 16, 128};
  const std::array<std::int64_t, 3> kv{tokens, 8, 128};
  const std::array<std::int64_t, 2> mlp{tokens, 3072};
  const std::array<std::int64_t, 2> rope{tokens, 64};
  const std::array<std::int64_t, 2> logits{samples, 151936};
  const std::array<std::int64_t, 1> sampled_bytes{samples * 4};
  const std::array<std::int64_t, 1> top_bytes{samples * 20 * 4};
  const std::array<std::int64_t, 1> error_bytes{4};
  auto result_layout = QwenBf16PackedResultLayout::Create(
      execution_layout.active_logit_rows());
  if (!result_layout.ok()) return result_layout.status();
  if (owners.logits.bytes < execution_layout.logit_workspace_bytes() * 2 ||
      owners.sampled_token.bytes <
          result_layout->device_error().offset_bytes) {
    return Status::InvalidArgument(
        "packed sampler arenas cannot cover their bounded layouts");
  }
  const std::array<std::int64_t, 1> workspace_bytes{
      samples * 151936 * 4};
  auto sampler_workspace_ids = arena_view(
      owners.logits, execution_layout.logit_workspace_bytes(), DType::kUInt8,
      workspace_bytes, *device);
  auto selected_logprobs = arena_view(
      owners.sampled_token, result_layout->selected_logprobs().offset_bytes,
      DType::kUInt8, sampled_bytes, *device);
  auto rng_words = arena_view(
      owners.sampled_token, result_layout->rng_words().offset_bytes,
      DType::kUInt8, sampled_bytes, *device);
  auto top_token_ids = arena_view(
      owners.sampled_token,
      result_layout->top_logprob_token_ids().offset_bytes, DType::kUInt8,
      top_bytes, *device);
  auto top_logprobs = arena_view(
      owners.sampled_token, result_layout->top_logprobs().offset_bytes,
      DType::kUInt8, top_bytes, *device);
  auto top_counts = arena_view(
      owners.sampled_token, result_layout->top_logprob_counts().offset_bytes,
      DType::kUInt8, sampled_bytes, *device);
  if (!sampler_workspace_ids.ok()) return sampler_workspace_ids.status();
  if (!selected_logprobs.ok()) return selected_logprobs.status();
  if (!rng_words.ok()) return rng_words.status();
  if (!top_token_ids.ok()) return top_token_ids.status();
  if (!top_logprobs.ok()) return top_logprobs.status();
  if (!top_counts.ok()) return top_counts.status();
  std::vector<QwenBf16SlotResource> resources;
  resources.reserve(QwenBf16ResourceSet::kSlotCount);
  using Slot = QwenBf16ActivationSlot;
  const auto push = [&](Slot slot, Result<TensorView> value) {
    return append(resources, slot, std::move(value));
  };
  for (const Status status : {
           push(Slot::kTokenIds, metadata->view(QwenBf16PackedStagingSlot::kTokenIds)),
           push(Slot::kPositions, metadata->view(QwenBf16PackedStagingSlot::kPositions)),
           push(Slot::kHidden, arena_view(owners.activations,
               execution_layout.hidden().offset_bytes, DType::kBFloat16, hidden, *device)),
           push(Slot::kNormalized, arena_view(owners.activations,
               execution_layout.normalized().offset_bytes, DType::kBFloat16, hidden, *device)),
           push(Slot::kQuery, arena_view(owners.activations,
               execution_layout.query().offset_bytes, DType::kBFloat16, query, *device)),
           push(Slot::kKey, arena_view(owners.activations,
               execution_layout.key().offset_bytes, DType::kBFloat16, kv, *device)),
           push(Slot::kValue, arena_view(owners.activations,
               execution_layout.value().offset_bytes, DType::kBFloat16, kv, *device)),
           push(Slot::kAttention, arena_view(owners.activations,
               execution_layout.attention().offset_bytes, DType::kBFloat16, query, *device)),
           push(Slot::kGate, arena_view(owners.mlp,
               execution_layout.mlp().gate_offset_bytes(), DType::kBFloat16, mlp, *device)),
           push(Slot::kUp, arena_view(owners.mlp,
               execution_layout.mlp().up_offset_bytes(), DType::kBFloat16, mlp, *device)),
           push(Slot::kRopeCosine, arena_view(owners.rope, 0,
               DType::kFloat32, rope, *device)),
           push(Slot::kRopeSine, arena_view(owners.rope,
               execution_layout.rope_workspace_bytes() / 2,
               DType::kFloat32, rope, *device)),
           push(Slot::kKvBacking, raw_view(owners.kv_backing,
               {0, static_cast<std::uint64_t>(slot_count) *
                       QwenKvSlotPool::kSlotPayloadBytes}, *device)),
           push(Slot::kKvSlotStates, raw_view(owners.kv_slot_states,
               {0, static_cast<std::uint64_t>(slot_count) *
                       sizeof(QwenKvSlotState)}, *device)),
           push(Slot::kKvAppendHandles,
                metadata->view(QwenBf16PackedStagingSlot::kKvAppendHandles)),
           push(Slot::kKvVisibleHandles,
                metadata->view(QwenBf16PackedStagingSlot::kKvVisibleHandles)),
           push(Slot::kKvTokenOffsets,
                metadata->view(QwenBf16PackedStagingSlot::kKvTokenOffsets)),
           push(Slot::kDeviceError, arena_view(owners.device_error, 0,
               DType::kUInt8, error_bytes, *device)),
           push(Slot::kLogits, arena_view(owners.logits, 0,
               DType::kFloat32, logits, *device)),
           push(Slot::kSampledToken, arena_view(owners.sampled_token, 0,
               DType::kUInt8, sampled_bytes, *device))}) {
    if (!status.ok()) return status;
  }
  auto activations = QwenBf16ResourceSet::Create(request_generation,
                                                  owning_rank, resources);
  if (!activations.ok()) return activations.status();
  return QwenBf16PackedResourceSet(
      std::move(*activations), std::move(*metadata),
      std::move(*sampler_workspace_ids), std::move(*selected_logprobs),
      std::move(*rng_words), std::move(*top_token_ids),
      std::move(*top_logprobs), std::move(*top_counts),
      staging_layout.execution_bucket_tokens(), sample_count,
      staging_layout.sequence_count());
}

}  // namespace pih
