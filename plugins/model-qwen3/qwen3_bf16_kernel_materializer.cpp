#include "pih/model/qwen3_bf16_kernel_materializer.h"

#include <functional>

#include "pih/model/qwen3_int4_bf16_weight_resolver.h"

namespace pih {
namespace {

using Slot = QwenBf16ActivationSlot;

Result<TensorView> required(const QwenBf16ResourceSet& resources, Slot slot) {
  return resources.view(slot);
}

using WeightLookup = std::function<Result<TensorView>(
    const QwenBf16PreparedCommand&)>;

Result<TensorView> required_weight(
    const QwenBf16PreparedCommand& command, const WeightLookup& lookup) {
  if (!command.has_weight()) {
    return Status::InvalidArgument("Qwen kernel command requires a weight");
  }
  return lookup(command);
}

Result<QwenBf16DispatchPlan> materialize(
    const QwenBf16PreparedCommand& command,
    const ResolvedKernelFunction& function,
    const QwenBf16ResourceSet& resources,
    std::int32_t weight_owning_rank, const WeightLookup& weight_lookup,
    const QwenBf16KernelContext& context) {
  if (command.backend != QwenBf16CommandBackend::kKernel ||
      context.request_generation == 0 ||
      context.request_generation != resources.request_generation() ||
      resources.owning_rank() != weight_owning_rank) {
    return Status::InvalidArgument(
        "Qwen kernel materialization identity is invalid");
  }
  const auto rank = resources.owning_rank();
  const auto operation = command.execution_step.operation;

  if (operation == QwenBf16ExecutionOp::kEmbedding) {
    auto table = required_weight(command, weight_lookup);
    auto ids = required(resources, Slot::kTokenIds);
    auto output = required(resources, Slot::kHidden);
    if (!table.ok()) return table.status();
    if (!ids.ok()) return ids.status();
    if (!output.ok()) return output.status();
    return QwenBf16DispatchPlan::CreateEmbedding(function, *table, *ids,
                                                  *output, rank);
  }
  if (operation == QwenBf16ExecutionOp::kPrepareRopeAngles) {
    auto positions = required(resources, Slot::kPositions);
    auto cosine = required(resources, Slot::kRopeCosine);
    auto sine = required(resources, Slot::kRopeSine);
    if (!positions.ok()) return positions.status();
    if (!cosine.ok()) return cosine.status();
    if (!sine.ok()) return sine.status();
    return QwenBf16DispatchPlan::CreateRopeAngles(function, *positions,
                                                   *cosine, *sine, rank);
  }
  if (operation == QwenBf16ExecutionOp::kInputRmsNorm ||
      operation == QwenBf16ExecutionOp::kPostAttentionRmsNorm ||
      operation == QwenBf16ExecutionOp::kFinalRmsNorm) {
    auto input = required(resources, Slot::kHidden);
    auto weight = required_weight(command, weight_lookup);
    auto output = required(resources, Slot::kNormalized);
    if (!input.ok()) return input.status();
    if (!weight.ok()) return weight.status();
    if (!output.ok()) return output.status();
    return QwenBf16DispatchPlan::CreateRmsNorm(
        function, *input, *weight, *output, context.rms_epsilon, rank);
  }
  if (operation == QwenBf16ExecutionOp::kQueryRmsNorm ||
      operation == QwenBf16ExecutionOp::kKeyRmsNorm) {
    const Slot slot = operation == QwenBf16ExecutionOp::kQueryRmsNorm
                          ? Slot::kQuery
                          : Slot::kKey;
    auto tensor = required(resources, slot);
    auto weight = required_weight(command, weight_lookup);
    if (!tensor.ok()) return tensor.status();
    if (!weight.ok()) return weight.status();
    return QwenBf16DispatchPlan::CreateRmsNorm(
        function, *tensor, *weight, *tensor, context.rms_epsilon, rank);
  }
  if (operation == QwenBf16ExecutionOp::kRope) {
    if (command.subcommand > 1) {
      return Status::InvalidArgument("Qwen RoPE subcommand is invalid");
    }
    const Slot slot = command.subcommand == 0 ? Slot::kQuery : Slot::kKey;
    auto tensor = required(resources, slot);
    auto cosine = required(resources, Slot::kRopeCosine);
    auto sine = required(resources, Slot::kRopeSine);
    if (!tensor.ok()) return tensor.status();
    if (!cosine.ok()) return cosine.status();
    if (!sine.ok()) return sine.status();
    return QwenBf16DispatchPlan::CreateRope(function, *tensor, *cosine,
                                             *sine, *tensor, rank);
  }
  if (operation == QwenBf16ExecutionOp::kKvAppend) {
    auto key = required(resources, Slot::kKey);
    auto value = required(resources, Slot::kValue);
    auto backing = required(resources, Slot::kKvBacking);
    auto states = required(resources, Slot::kKvSlotStates);
    auto handles = required(resources, Slot::kKvAppendHandles);
    auto offsets = required(resources, Slot::kKvTokenOffsets);
    auto error = required(resources, Slot::kDeviceError);
    if (!key.ok()) return key.status();
    if (!value.ok()) return value.status();
    if (!backing.ok()) return backing.status();
    if (!states.ok()) return states.status();
    if (!handles.ok()) return handles.status();
    if (!offsets.ok()) return offsets.status();
    if (!error.ok()) return error.status();
    return QwenBf16DispatchPlan::CreateKvAppend(
        function, *key, *value, *backing, *states, *handles, *offsets, *error,
        context.owner_sequence_index, command.execution_step.layer,
        context.slot_count, rank);
  }
  if (operation == QwenBf16ExecutionOp::kPagedGqa) {
    auto query = required(resources, Slot::kQuery);
    auto output = required(resources, Slot::kAttention);
    auto backing = required(resources, Slot::kKvBacking);
    auto states = required(resources, Slot::kKvSlotStates);
    auto handles = required(resources, Slot::kKvVisibleHandles);
    auto error = required(resources, Slot::kDeviceError);
    if (!query.ok()) return query.status();
    if (!output.ok()) return output.status();
    if (!backing.ok()) return backing.status();
    if (!states.ok()) return states.status();
    if (!handles.ok()) return handles.status();
    if (!error.ok()) return error.status();
    return QwenBf16DispatchPlan::CreatePagedGqa(
        function, *query, *output, *backing, *states, *handles, *error,
        context.owner_sequence_index, command.execution_step.layer,
        context.query_start_position, context.key_token_count,
        context.attention_scale, context.slot_count, rank);
  }
  if (operation == QwenBf16ExecutionOp::kAttentionResidual ||
      operation == QwenBf16ExecutionOp::kMlpResidual) {
    auto hidden = required(resources, Slot::kHidden);
    auto normalized = required(resources, Slot::kNormalized);
    if (!hidden.ok()) return hidden.status();
    if (!normalized.ok()) return normalized.status();
    return QwenBf16DispatchPlan::CreateElementwise(
        command.primitive, function, *hidden, *normalized, *hidden, rank);
  }
  if (operation == QwenBf16ExecutionOp::kSiluMul) {
    auto gate = required(resources, Slot::kGate);
    auto up = required(resources, Slot::kUp);
    if (!gate.ok()) return gate.status();
    if (!up.ok()) return up.status();
    return QwenBf16DispatchPlan::CreateElementwise(
        command.primitive, function, *gate, *up, *gate, rank);
  }
  if (operation == QwenBf16ExecutionOp::kGreedyArgmax) {
    auto logits = required(resources, Slot::kLogits);
    auto sampled = required(resources, Slot::kSampledToken);
    auto error = required(resources, Slot::kDeviceError);
    if (!logits.ok()) return logits.status();
    if (!sampled.ok()) return sampled.status();
    if (!error.ok()) return error.status();
    return QwenBf16DispatchPlan::CreateGreedyArgmax(
        function, *logits, *sampled, *error, rank);
  }
  return Status::InvalidArgument(
      "Qwen execution operation is not a materializable kernel");
}

}  // namespace

Result<QwenBf16DispatchPlan> QwenBf16KernelMaterializer::Create(
    const QwenBf16PreparedCommand& command,
    const ResolvedKernelFunction& function,
    const QwenBf16ResourceSet& resources,
    const QwenBf16WeightResourceSet& weights,
    const QwenBf16KernelContext& context) {
  return materialize(command,function,resources,weights.owning_rank(),
      [&](const QwenBf16PreparedCommand& value) {
        return weights.view(value.tensor_index);
      },context);
}

Result<QwenBf16DispatchPlan> QwenBf16KernelMaterializer::Create(
    const QwenBf16PreparedCommand& command,
    const ResolvedKernelFunction& function,
    const QwenBf16ResourceSet& resources,
    const QwenInt4WeightResourceSet& weights,
    const QwenBf16KernelContext& context) {
  return materialize(command,function,resources,weights.owning_rank(),
      [&](const QwenBf16PreparedCommand& value) {
        return QwenInt4Bf16WeightResolver::Resolve(value,weights);
      },context);
}

}  // namespace pih
