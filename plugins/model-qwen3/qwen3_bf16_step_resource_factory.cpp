#include "pih/model/qwen3_bf16_step_resource_factory.h"

#include <array>
#include <vector>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Result<TensorView> view(const QwenBf16DeviceArenaOwner& owner,
                        std::uint64_t offset, DType dtype,
                        std::span<const std::int64_t> shape,
                        const Device& device) {
  if (owner.base == 0 || owner.bytes == 0 || owner.generation == 0) {
    return Status::InvalidArgument("Qwen device arena owner is invalid");
  }
  auto address = checked_add_u64(owner.base, offset);
  if (!address.ok()) return address.status();
  auto result = TensorView::Create(
      reinterpret_cast<void*>(static_cast<std::uintptr_t>(*address)), dtype,
      shape, {}, device, owner.generation);
  if (!result.ok()) return result.status();
  auto end = checked_add_u64(offset, result->byte_span());
  if (!end.ok()) return end.status();
  if (*end > owner.bytes) {
    return Status::InvalidArgument("Qwen tensor view exceeds its arena owner");
  }
  return result;
}

Result<TensorView> raw(const QwenBf16DeviceArenaOwner& owner,
                       QwenBf16ArenaSpan span, const Device& device) {
  if (span.size_bytes > static_cast<std::uint64_t>(INT64_MAX)) {
    return Status::ResourceExhausted("Qwen raw resource exceeds tensor rank");
  }
  const std::array<std::int64_t, 1> shape{
      static_cast<std::int64_t>(span.size_bytes)};
  return view(owner, span.offset_bytes, DType::kUInt8, shape, device);
}

Status add(std::vector<QwenBf16SlotResource>& resources,
           QwenBf16ActivationSlot slot, Result<TensorView> tensor) {
  if (!tensor.ok()) return tensor.status();
  resources.push_back({slot, std::move(*tensor)});
  return Status::Ok();
}

}  // namespace

Result<QwenBf16ResourceSet> QwenBf16StepResourceFactory::Create(
    std::uint64_t request_generation, std::int32_t owning_rank,
    std::uint32_t slot_count,
    const QwenBf16ExecutionArenaLayout& execution_layout,
    const QwenBf16StepStagingLayout& staging_layout,
    const QwenBf16StepDeviceOwners& owners) {
  if (request_generation == 0 || owning_rank < 0 || slot_count == 0 ||
      slot_count > QwenKvSlotPool::kMaximumSlots ||
      execution_layout.tokens() != staging_layout.token_count() ||
      owners.device_error.generation != request_generation) {
    return Status::InvalidArgument(
        "Qwen step resource factory identity is invalid");
  }
  auto device = Device::Create(DeviceType::kCuda, owning_rank);
  if (!device.ok()) return device.status();
  const auto tokens = static_cast<std::int64_t>(execution_layout.tokens());
  const std::array<std::int64_t, 1> token_shape{tokens};
  const std::array<std::int64_t, 2> hidden_shape{tokens, 1024};
  const std::array<std::int64_t, 3> query_shape{tokens, 16, 128};
  const std::array<std::int64_t, 3> kv_shape{tokens, 8, 128};
  const std::array<std::int64_t, 2> mlp_shape{tokens, 3072};
  const std::array<std::int64_t, 2> rope_shape{tokens, 64};
  const std::array<std::int64_t, 2> logits_shape{1, 151936};
  const std::array<std::int64_t, 1> sampled_shape{1};
  const std::array<std::int64_t, 1> error_shape{4};

  std::vector<QwenBf16SlotResource> resources;
  resources.reserve(QwenBf16ResourceSet::kSlotCount);
  const auto push = [&](QwenBf16ActivationSlot slot,
                        Result<TensorView> tensor) {
    return add(resources, slot, std::move(tensor));
  };
  using Slot = QwenBf16ActivationSlot;
  for (const Status status : {
           push(Slot::kTokenIds,
                view(owners.step_staging,
                     staging_layout.token_ids().offset_bytes, DType::kInt64,
                     token_shape, *device)),
           push(Slot::kPositions,
                view(owners.step_staging,
                     staging_layout.positions().offset_bytes, DType::kInt64,
                     token_shape, *device)),
           push(Slot::kHidden,
                view(owners.activations,
                     execution_layout.hidden().offset_bytes,
                     DType::kBFloat16, hidden_shape, *device)),
           push(Slot::kNormalized,
                view(owners.activations,
                     execution_layout.normalized().offset_bytes,
                     DType::kBFloat16, hidden_shape, *device)),
           push(Slot::kQuery,
                view(owners.activations, execution_layout.query().offset_bytes,
                     DType::kBFloat16, query_shape, *device)),
           push(Slot::kKey,
                view(owners.activations, execution_layout.key().offset_bytes,
                     DType::kBFloat16, kv_shape, *device)),
           push(Slot::kValue,
                view(owners.activations, execution_layout.value().offset_bytes,
                     DType::kBFloat16, kv_shape, *device)),
           push(Slot::kAttention,
                view(owners.activations,
                     execution_layout.attention().offset_bytes,
                     DType::kBFloat16, query_shape, *device)),
           push(Slot::kGate,
                view(owners.mlp, execution_layout.mlp().gate_offset_bytes(),
                     DType::kBFloat16, mlp_shape, *device)),
           push(Slot::kUp,
                view(owners.mlp, execution_layout.mlp().up_offset_bytes(),
                     DType::kBFloat16, mlp_shape, *device)),
           push(Slot::kRopeCosine,
                view(owners.rope, 0, DType::kFloat32, rope_shape, *device)),
           push(Slot::kRopeSine,
                view(owners.rope,
                     execution_layout.rope_workspace_bytes() / 2,
                     DType::kFloat32, rope_shape, *device)),
           push(Slot::kKvBacking,
                raw(owners.kv_backing,
                    {0, static_cast<std::uint64_t>(slot_count) *
                            QwenKvSlotPool::kSlotPayloadBytes},
                    *device)),
           push(Slot::kKvSlotStates,
                raw(owners.kv_slot_states,
                    {0, static_cast<std::uint64_t>(slot_count) *
                            sizeof(QwenKvSlotState)},
                    *device)),
           push(Slot::kKvAppendHandles,
                raw(owners.step_staging, staging_layout.append_handles(),
                    *device)),
           push(Slot::kKvVisibleHandles,
                raw(owners.step_staging, staging_layout.visible_handles(),
                    *device)),
           push(Slot::kKvTokenOffsets,
                raw(owners.step_staging, staging_layout.token_offsets(),
                    *device)),
           push(Slot::kDeviceError,
                view(owners.device_error, 0, DType::kUInt8, error_shape,
                     *device)),
           push(Slot::kLogits,
                view(owners.logits, 0, DType::kFloat32, logits_shape,
                     *device)),
           push(Slot::kSampledToken,
                view(owners.sampled_token, 0, DType::kInt64, sampled_shape,
                     *device))}) {
    if (!status.ok()) return status;
  }
  return QwenBf16ResourceSet::Create(request_generation, owning_rank,
                                      resources);
}

}  // namespace pih
