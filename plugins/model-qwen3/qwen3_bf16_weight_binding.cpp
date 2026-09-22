#include "pih/model/qwen3_bf16_weight_binding.h"

#include <array>

namespace pih {
namespace {

Result<std::uint16_t> tensor_index(const QwenBf16ExecutionStep& step) {
  if (step.layer == QwenBf16ExecutionSchedule::kGlobalLayer) {
    switch (step.operation) {
      case QwenBf16ExecutionOp::kEmbedding:
        return std::uint16_t{1};
      case QwenBf16ExecutionOp::kFinalRmsNorm:
        return std::uint16_t{310};
      case QwenBf16ExecutionOp::kLmHead:
        return std::uint16_t{0};
      default:
        return Status::FailedPrecondition(
            "global Qwen execution operation has invalid weight role");
    }
  }
  if (step.layer >= QwenBf16ExecutionSchedule::kLayerCount) {
    return Status::FailedPrecondition("Qwen weight binding layer is invalid");
  }
  const auto base = static_cast<std::uint16_t>(2 + 11 * step.layer);
  switch (step.operation) {
    case QwenBf16ExecutionOp::kInputRmsNorm: return base;
    case QwenBf16ExecutionOp::kDownLinear: return base + 1;
    case QwenBf16ExecutionOp::kGateLinear: return base + 2;
    case QwenBf16ExecutionOp::kUpLinear: return base + 3;
    case QwenBf16ExecutionOp::kPostAttentionRmsNorm: return base + 4;
    case QwenBf16ExecutionOp::kKeyRmsNorm: return base + 5;
    case QwenBf16ExecutionOp::kKeyLinear: return base + 6;
    case QwenBf16ExecutionOp::kAttentionOutputLinear: return base + 7;
    case QwenBf16ExecutionOp::kQueryRmsNorm: return base + 8;
    case QwenBf16ExecutionOp::kQueryLinear: return base + 9;
    case QwenBf16ExecutionOp::kValueLinear: return base + 10;
    default:
      return Status::FailedPrecondition(
          "layer Qwen execution operation has no weight role");
  }
}

bool requires_weight(QwenBf16ExecutionOp operation) noexcept {
  switch (operation) {
    case QwenBf16ExecutionOp::kEmbedding:
    case QwenBf16ExecutionOp::kInputRmsNorm:
    case QwenBf16ExecutionOp::kQueryLinear:
    case QwenBf16ExecutionOp::kKeyLinear:
    case QwenBf16ExecutionOp::kValueLinear:
    case QwenBf16ExecutionOp::kQueryRmsNorm:
    case QwenBf16ExecutionOp::kKeyRmsNorm:
    case QwenBf16ExecutionOp::kAttentionOutputLinear:
    case QwenBf16ExecutionOp::kPostAttentionRmsNorm:
    case QwenBf16ExecutionOp::kGateLinear:
    case QwenBf16ExecutionOp::kUpLinear:
    case QwenBf16ExecutionOp::kDownLinear:
    case QwenBf16ExecutionOp::kFinalRmsNorm:
    case QwenBf16ExecutionOp::kLmHead:
      return true;
    default:
      return false;
  }
}

}  // namespace

Result<QwenBf16WeightBindingPlan> QwenBf16WeightBindingPlan::Create(
    const QwenBf16ExecutionSchedule& schedule) {
  QwenBf16WeightBindingPlan plan;
  std::array<bool, Qwen3Manifest::kOfficialTensorCount> used{};
  for (std::size_t step_index = 0; step_index < schedule.size(); ++step_index) {
    const auto& step = schedule[step_index];
    if (!requires_weight(step.operation)) continue;
    auto index = tensor_index(step);
    if (!index.ok() || *index >= used.size() || used[*index]) {
      return Status::FailedPrecondition(
          "Qwen execution schedule has an invalid or duplicate weight binding");
    }
    used[*index] = true;
    plan.bindings_[step_index].tensor_index = *index;
    ++plan.bound_weight_count_;
  }
  if (plan.bound_weight_count_ != Qwen3Manifest::kOfficialTensorCount) {
    return Status::FailedPrecondition(
        "Qwen execution schedule does not bind the complete weight manifest");
  }
  return plan;
}

const ExpectedTensor* QwenBf16WeightBindingPlan::tensor(
    std::size_t step) const noexcept {
  if (step >= bindings_.size() || !bindings_[step].has_weight()) return nullptr;
  const auto& tensors = Qwen3Manifest::expected_tensors();
  const auto index = bindings_[step].tensor_index;
  return index < tensors.size() ? &tensors[index] : nullptr;
}

}  // namespace pih
