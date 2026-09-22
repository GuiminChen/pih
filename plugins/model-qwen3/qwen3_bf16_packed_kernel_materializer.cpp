#include "pih/model/qwen3_bf16_packed_kernel_materializer.h"

#include <array>
#include <cmath>

#include "pih/model/qwen3_int4_bf16_weight_resolver.h"

namespace pih {
namespace {

using Slot = QwenBf16ActivationSlot;
using PackedSlot = QwenBf16PackedStagingSlot;

template <typename WeightSet>
Status validate(const QwenBf16PackedResourceSet& resources,
                const WeightSet& weights,
                const QwenBf16PackedKernelContext& context) {
  if (context.request_generation == 0 ||
      context.request_generation != resources.activations().request_generation() ||
      resources.activations().owning_rank() != weights.owning_rank() ||
      context.real_token_count == 0 ||
      context.real_token_count > resources.execution_bucket_tokens() ||
      context.sequence_count != resources.sequence_count() ||
      context.slot_count == 0 || context.slot_count > QwenKvSlotPool::kMaximumSlots ||
      !std::isfinite(context.rms_epsilon) || context.rms_epsilon <= 0 ||
      !std::isfinite(context.attention_scale) || context.attention_scale <= 0) {
    return Status::InvalidArgument("packed kernel materializer context is invalid");
  }
  return Status::Ok();
}

Result<TensorView> retained_weight(
    const QwenBf16WeightResourceSet& weights,
    const QwenBf16PreparedCommand& command) {
  return weights.view(command.tensor_index);
}

Result<TensorView> retained_weight(
    const QwenInt4WeightResourceSet& weights,
    const QwenBf16PreparedCommand& command) {
  return QwenInt4Bf16WeightResolver::Resolve(command,weights);
}

Result<TensorView> activation(const QwenBf16PackedResourceSet& resources,
                              Slot slot) {
  return resources.activations().view(slot);
}

Result<TensorView> metadata(const QwenBf16PackedResourceSet& resources,
                            PackedSlot slot) {
  return resources.metadata().view(slot);
}

Result<TensorView> leading_hidden(const TensorView& source,
                                  std::uint32_t rows) {
  if (source.dtype() != DType::kBFloat16 || source.rank() != 2 || rows == 0 ||
      rows > source.dim(0) || source.dim(1) != 1024) {
    return Status::InvalidArgument("packed sample hidden destination is invalid");
  }
  const std::array<std::int64_t, 2> shape{
      static_cast<std::int64_t>(rows), 1024};
  return TensorView::Create(source.data(), DType::kBFloat16, shape, {},
                            source.device(), source.generation());
}

}  // namespace

bool QwenBf16PackedKernelMaterializer::uses_packed_primitive(
    const QwenBf16PreparedCommand& command) noexcept {
  const auto op = command.execution_step.operation;
  return op == QwenBf16ExecutionOp::kEmbedding ||
         op == QwenBf16ExecutionOp::kPrepareRopeAngles ||
         op == QwenBf16ExecutionOp::kKvAppend ||
         op == QwenBf16ExecutionOp::kPagedGqa ||
         op == QwenBf16ExecutionOp::kGreedyArgmax;
}

template <typename WeightSet>
Result<QwenBf16PackedDispatchPlan> create_packed(
    const QwenBf16PreparedCommand& command,
    const ResolvedKernelFunction& function,
    const QwenBf16PackedResourceSet& resources,
    const WeightSet& weights,
    const QwenBf16PackedKernelContext& context) {
  auto status = validate(resources, weights, context);
  if (!status.ok()) return status;
  if (command.backend != QwenBf16CommandBackend::kKernel ||
      !QwenBf16PackedKernelMaterializer::uses_packed_primitive(command))
    return Status::InvalidArgument("command does not use a packed primitive");
  const auto rank = resources.activations().owning_rank();
  const auto op = command.execution_step.operation;
  if (op == QwenBf16ExecutionOp::kEmbedding) {
    auto table = retained_weight(weights,command);
    auto ids = metadata(resources, PackedSlot::kTokenIds);
    auto output = activation(resources, Slot::kHidden);
    if (!table.ok()) return table.status(); if (!ids.ok()) return ids.status();
    if (!output.ok()) return output.status();
    return QwenBf16PackedDispatchPlan::CreateEmbedding(
        function, *table, *ids, *output, context.real_token_count, rank);
  }
  if (op == QwenBf16ExecutionOp::kPrepareRopeAngles) {
    auto positions = metadata(resources, PackedSlot::kPositions);
    auto cosine = activation(resources, Slot::kRopeCosine);
    auto sine = activation(resources, Slot::kRopeSine);
    if (!positions.ok()) return positions.status();
    if (!cosine.ok()) return cosine.status(); if (!sine.ok()) return sine.status();
    return QwenBf16PackedDispatchPlan::CreateRopeAngles(
        function, *positions, *cosine, *sine, context.real_token_count, rank);
  }
  if (op == QwenBf16ExecutionOp::kKvAppend) {
    auto key = activation(resources, Slot::kKey);
    auto value = activation(resources, Slot::kValue);
    auto backing = activation(resources, Slot::kKvBacking);
    auto states = activation(resources, Slot::kKvSlotStates);
    auto handles = metadata(resources, PackedSlot::kKvAppendHandles);
    auto offsets = metadata(resources, PackedSlot::kKvTokenOffsets);
    auto requests = metadata(resources, PackedSlot::kRequestIndex);
    auto owners = metadata(resources, PackedSlot::kOwnerSequenceIndices);
    auto error = activation(resources, Slot::kDeviceError);
    if (!key.ok()) return key.status(); if (!value.ok()) return value.status();
    if (!backing.ok()) return backing.status(); if (!states.ok()) return states.status();
    if (!handles.ok()) return handles.status(); if (!offsets.ok()) return offsets.status();
    if (!requests.ok()) return requests.status(); if (!owners.ok()) return owners.status();
    if (!error.ok()) return error.status();
    return QwenBf16PackedDispatchPlan::CreateKvAppend(
        function, *key, *value, *backing, *states, *handles, *offsets,
        *requests, *owners, *error, command.execution_step.layer,
        context.real_token_count, context.sequence_count, context.slot_count,
        rank);
  }
  if (op == QwenBf16ExecutionOp::kPagedGqa) {
    auto query = activation(resources, Slot::kQuery);
    auto output = activation(resources, Slot::kAttention);
    auto backing = activation(resources, Slot::kKvBacking);
    auto states = activation(resources, Slot::kKvSlotStates);
    auto handles = metadata(resources, PackedSlot::kKvVisibleHandles);
    auto handle_offsets = metadata(resources, PackedSlot::kKvVisibleHandleOffsets);
    auto requests = metadata(resources, PackedSlot::kRequestIndex);
    auto query_offsets = metadata(resources, PackedSlot::kQueryStartOffsets);
    auto key_counts = metadata(resources, PackedSlot::kKeyTokenCounts);
    auto owners = metadata(resources, PackedSlot::kOwnerSequenceIndices);
    auto error = activation(resources, Slot::kDeviceError);
    if (!query.ok()) return query.status(); if (!output.ok()) return output.status();
    if (!backing.ok()) return backing.status(); if (!states.ok()) return states.status();
    if (!handles.ok()) return handles.status(); if (!handle_offsets.ok()) return handle_offsets.status();
    if (!requests.ok()) return requests.status(); if (!query_offsets.ok()) return query_offsets.status();
    if (!key_counts.ok()) return key_counts.status(); if (!owners.ok()) return owners.status();
    if (!error.ok()) return error.status();
    return QwenBf16PackedDispatchPlan::CreatePagedGqa(
        function, *query, *output, *backing, *states, *handles,
        *handle_offsets, *requests, *query_offsets, *key_counts, *owners,
        *error, command.execution_step.layer, context.real_token_count,
        context.sequence_count, context.attention_scale, context.slot_count,
        rank);
  }
  auto logits = activation(resources, Slot::kLogits);
  auto sampled = activation(resources, Slot::kSampledToken);
  auto descriptors = metadata(resources, PackedSlot::kSamplingDescriptors);
  auto sample_sequences =
      metadata(resources, PackedSlot::kSampleSequenceIndices);
  auto error = activation(resources, Slot::kDeviceError);
  if (!logits.ok()) return logits.status(); if (!sampled.ok()) return sampled.status();
  if (!descriptors.ok()) return descriptors.status();
  if (!sample_sequences.ok()) return sample_sequences.status();
  if (!error.ok()) return error.status();
  return QwenBf16PackedDispatchPlan::CreateSampler(
      function, *logits, *descriptors, *sample_sequences,
      resources.sampler_workspace_ids(), *sampled,
      resources.selected_logprobs(), resources.rng_words(),
      resources.top_token_ids(), resources.top_logprobs(),
      resources.top_counts(), *error, resources.sample_count(),
      resources.sequence_count(), rank);
}

Result<QwenBf16PackedDispatchPlan>
QwenBf16PackedKernelMaterializer::CreatePacked(
    const QwenBf16PreparedCommand& command,
    const ResolvedKernelFunction& function,
    const QwenBf16PackedResourceSet& resources,
    const QwenBf16WeightResourceSet& weights,
    const QwenBf16PackedKernelContext& context) {
  return create_packed(command,function,resources,weights,context);
}

Result<QwenBf16PackedDispatchPlan>
QwenBf16PackedKernelMaterializer::CreatePacked(
    const QwenBf16PreparedCommand& command,
    const ResolvedKernelFunction& function,
    const QwenBf16PackedResourceSet& resources,
    const QwenInt4WeightResourceSet& weights,
    const QwenBf16PackedKernelContext& context) {
  return create_packed(command,function,resources,weights,context);
}

Result<QwenBf16DispatchPlan> QwenBf16PackedKernelMaterializer::CreateLegacy(
    const QwenBf16PreparedCommand& command,
    const ResolvedKernelFunction& function,
    const QwenBf16PackedResourceSet& resources,
    const QwenBf16WeightResourceSet& weights,
    const QwenBf16PackedKernelContext& context) {
  auto status = validate(resources, weights, context);
  if (!status.ok()) return status;
  if (uses_packed_primitive(command))
    return Status::InvalidArgument("packed command cannot use legacy materializer");
  const QwenBf16KernelContext legacy{context.request_generation, 1, 0,
      context.real_token_count, context.slot_count, context.rms_epsilon,
      context.attention_scale};
  return QwenBf16KernelMaterializer::Create(
      command, function, resources.activations(), weights, legacy);
}

Result<QwenBf16DispatchPlan> QwenBf16PackedKernelMaterializer::CreateLegacy(
    const QwenBf16PreparedCommand& command,
    const ResolvedKernelFunction& function,
    const QwenBf16PackedResourceSet& resources,
    const QwenInt4WeightResourceSet& weights,
    const QwenBf16PackedKernelContext& context) {
  auto status = validate(resources, weights, context);
  if (!status.ok()) return status;
  if (uses_packed_primitive(command))
    return Status::InvalidArgument("packed command cannot use legacy materializer");
  const QwenBf16KernelContext legacy{context.request_generation, 1, 0,
      context.real_token_count, context.slot_count, context.rms_epsilon,
      context.attention_scale};
  return QwenBf16KernelMaterializer::Create(
      command, function, resources.activations(), weights, legacy);
}

Result<QwenBf16PackedDispatchPlan>
QwenBf16PackedKernelMaterializer::CreateSampleHidden(
    const ResolvedKernelFunction& function,
    const QwenBf16PackedResourceSet& resources,
    const QwenBf16PackedKernelContext& context) {
  if (context.request_generation != resources.activations().request_generation())
    return Status::InvalidArgument("packed sample generation is invalid");
  auto normalized = activation(resources, Slot::kNormalized);
  auto hidden = activation(resources, Slot::kHidden);
  auto rows = metadata(resources, PackedSlot::kSampleRowIndex);
  auto error = activation(resources, Slot::kDeviceError);
  if (!normalized.ok()) return normalized.status(); if (!hidden.ok()) return hidden.status();
  if (!rows.ok()) return rows.status(); if (!error.ok()) return error.status();
  auto output = leading_hidden(*hidden, resources.sample_count());
  if (!output.ok()) return output.status();
  return QwenBf16PackedDispatchPlan::CreateSampleHidden(
      function, *normalized, *rows, *output, *error,
      resources.sample_count(), resources.execution_bucket_tokens(),
      resources.activations().owning_rank());
}

}  // namespace pih
