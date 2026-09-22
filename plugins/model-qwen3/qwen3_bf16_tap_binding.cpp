#include "pih/model/qwen3_bf16_tap_binding.h"

#include <limits>

namespace pih {
namespace {

struct ProducerSpec final {
  QwenBf16ExecutionOp operation;
  std::uint32_t layer;
  std::uint8_t subcommand;
  QwenBf16ActivationSlot output;
  QwenBf16TapKvComponent kv_component;
};

Result<ProducerSpec> producer_for(const QwenNumericalTapRequest& request) {
  const auto layer = request.layer;
  switch (request.point) {
    case QwenNumericalTapPoint::kLayerHidden:
      return ProducerSpec{QwenBf16ExecutionOp::kMlpResidual, layer, 0,
                          QwenBf16ActivationSlot::kHidden,
                          QwenBf16TapKvComponent::kNotApplicable};
    case QwenNumericalTapPoint::kQueryAfterNorm:
      return ProducerSpec{QwenBf16ExecutionOp::kQueryRmsNorm, layer, 0,
                          QwenBf16ActivationSlot::kQuery,
                          QwenBf16TapKvComponent::kNotApplicable};
    case QwenNumericalTapPoint::kKeyAfterNorm:
      return ProducerSpec{QwenBf16ExecutionOp::kKeyRmsNorm, layer, 0,
                          QwenBf16ActivationSlot::kKey,
                          QwenBf16TapKvComponent::kNotApplicable};
    case QwenNumericalTapPoint::kQueryAfterRope:
      return ProducerSpec{QwenBf16ExecutionOp::kRope, layer, 0,
                          QwenBf16ActivationSlot::kQuery,
                          QwenBf16TapKvComponent::kNotApplicable};
    case QwenNumericalTapPoint::kKeyAfterRope:
      return ProducerSpec{QwenBf16ExecutionOp::kRope, layer, 1,
                          QwenBf16ActivationSlot::kKey,
                          QwenBf16TapKvComponent::kNotApplicable};
    case QwenNumericalTapPoint::kPrefillAttention:
      return ProducerSpec{QwenBf16ExecutionOp::kPagedGqa, layer, 0,
                          QwenBf16ActivationSlot::kAttention,
                          QwenBf16TapKvComponent::kNotApplicable};
    case QwenNumericalTapPoint::kKvKey:
      return ProducerSpec{QwenBf16ExecutionOp::kKvAppend, layer, 0,
                          QwenBf16ActivationSlot::kKvBacking,
                          QwenBf16TapKvComponent::kKey};
    case QwenNumericalTapPoint::kKvValue:
      return ProducerSpec{QwenBf16ExecutionOp::kKvAppend, layer, 0,
                          QwenBf16ActivationSlot::kKvBacking,
                          QwenBf16TapKvComponent::kValue};
    case QwenNumericalTapPoint::kFinalNorm:
      return ProducerSpec{QwenBf16ExecutionOp::kFinalRmsNorm,
                          QwenBf16ExecutionSchedule::kGlobalLayer, 0,
                          QwenBf16ActivationSlot::kNormalized,
                          QwenBf16TapKvComponent::kNotApplicable};
    case QwenNumericalTapPoint::kLogits:
      return ProducerSpec{QwenBf16ExecutionOp::kLmHead,
                          QwenBf16ExecutionSchedule::kGlobalLayer, 0,
                          QwenBf16ActivationSlot::kLogits,
                          QwenBf16TapKvComponent::kNotApplicable};
  }
  return Status::InvalidArgument("Qwen tap has no execution producer");
}

}  // namespace

Result<QwenBf16TapBindingPlan> QwenBf16TapBindingPlan::Create(
    const QwenNumericalTapPlan& taps,
    const QwenBf16CommandBuffer& commands) {
  std::vector<QwenBf16TapBinding> bindings;
  bindings.reserve(taps.captures().size());
  for (std::size_t capture_index = 0;
       capture_index < taps.captures().size(); ++capture_index) {
    const auto& request = taps.captures()[capture_index].request;
    auto spec = producer_for(request);
    if (!spec.ok()) return spec.status();
    std::size_t match = commands.size();
    for (std::size_t command_index = 0; command_index < commands.size();
         ++command_index) {
      const auto& command = commands[command_index];
      if (command.execution_step.operation == spec->operation &&
          command.execution_step.layer == spec->layer &&
          command.subcommand == spec->subcommand) {
        if (match != commands.size()) {
          return Status::FailedPrecondition(
              "Qwen tap producer command is ambiguous");
        }
        match = command_index;
      }
    }
    if (match == commands.size()) {
      return Status::FailedPrecondition(
          "Qwen tap producer command is missing");
    }
    const auto& command = commands[match];
    bindings.push_back({capture_index, request, match, command.execution_step,
                        command.subcommand, spec->output,
                        spec->kv_component});
  }
  if (bindings.size() > std::numeric_limits<std::uint16_t>::max()) {
    return Status::ResourceExhausted(
        "Qwen tap binding index exceeds frozen runtime width");
  }
  std::array<std::uint16_t, QwenBf16CommandBuffer::kCommandCount + 1>
      offsets{};
  for (const auto& binding : bindings) {
    if (binding.producer_command_index >= QwenBf16CommandBuffer::kCommandCount) {
      return Status::FailedPrecondition(
          "Qwen tap producer lies outside the frozen command buffer");
    }
    ++offsets[binding.producer_command_index + 1];
  }
  for (std::size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] = static_cast<std::uint16_t>(offsets[i] + offsets[i - 1]);
  }
  std::vector<std::uint16_t> ordered(bindings.size());
  auto cursors = offsets;
  for (std::size_t binding_index = 0; binding_index < bindings.size();
       ++binding_index) {
    const auto command_index = bindings[binding_index].producer_command_index;
    ordered[cursors[command_index]++] =
        static_cast<std::uint16_t>(binding_index);
  }
  return QwenBf16TapBindingPlan(std::move(bindings), offsets,
                                std::move(ordered));
}

std::span<const std::uint16_t> QwenBf16TapBindingPlan::bindings_after_command(
    std::size_t command_index) const noexcept {
  if (command_index >= QwenBf16CommandBuffer::kCommandCount) return {};
  const auto begin = command_offsets_[command_index];
  const auto end = command_offsets_[command_index + 1];
  return std::span<const std::uint16_t>(ordered_binding_indices_).subspan(
      begin, end - begin);
}

}  // namespace pih
