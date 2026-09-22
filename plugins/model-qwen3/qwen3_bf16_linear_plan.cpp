#include "pih/model/qwen3_bf16_linear_plan.h"

#include <algorithm>
#include <array>
#include <utility>

namespace pih {
namespace {

constexpr std::array<QwenBf16LinearKind, QwenBf16LinearPlanSet::kPlanCount>
    kKinds{QwenBf16LinearKind::kQuery,
           QwenBf16LinearKind::kKey,
           QwenBf16LinearKind::kValue,
           QwenBf16LinearKind::kAttentionOutput,
           QwenBf16LinearKind::kGate,
           QwenBf16LinearKind::kUp,
           QwenBf16LinearKind::kDown,
           QwenBf16LinearKind::kLmHead};

}  // namespace

Result<std::size_t> QwenBf16LinearPlanSet::index(
    QwenBf16LinearKind kind) {
  const auto value = static_cast<std::size_t>(kind);
  if (value >= kKinds.size() || kKinds[value] != kind) {
    return Status::InvalidArgument("unknown Qwen BF16 linear kind");
  }
  return value;
}

Result<std::unique_ptr<QwenBf16LinearPlanSet>>
QwenBf16LinearPlanSet::Create(std::uint64_t tokens,
                              std::uint64_t maximum_workspace_bytes) {
  std::array<std::unique_ptr<GemmPlan>, kPlanCount> plans;
  std::uint64_t required_workspace = 0;
  for (std::size_t ordinal = 0; ordinal < kKinds.size(); ++ordinal) {
    auto rows = qwen_bf16_linear_execution_rows(kKinds[ordinal], tokens, 1);
    if (!rows.ok()) return rows.status();
    auto shape = QwenBf16LinearShape::Create(kKinds[ordinal], *rows);
    if (!shape.ok()) return shape.status();
    auto plan = GemmPlan::Create(shape->tokens(), shape->output_features(),
                                 shape->input_features(),
                                 maximum_workspace_bytes,
                                 shape->output_dtype());
    if (!plan.ok()) return plan.status();
    required_workspace =
        std::max(required_workspace, plan->get()->workspace_bytes());
    plans[ordinal] = std::move(plan).value();
  }
  return std::unique_ptr<QwenBf16LinearPlanSet>(new QwenBf16LinearPlanSet(
      tokens, required_workspace, std::move(plans)));
}

Status QwenBf16LinearPlanSet::execute(
    QwenBf16LinearKind kind, const TensorView& input,
    const TensorView& weight, const TensorView& output, void* workspace,
    std::uint64_t workspace_bytes, cudaStream_t stream) const {
  auto ordinal = index(kind);
  if (!ordinal.ok()) return ordinal.status();
  if (stream == nullptr) {
    return Status::InvalidArgument(
        "Qwen BF16 linear execution requires an explicit stream");
  }
  return plans_[ordinal.value()]->execute(input, weight, output, workspace,
                                           workspace_bytes, stream);
}

Result<std::int32_t> QwenBf16LinearPlanSet::algorithm_id(
    QwenBf16LinearKind kind) const {
  auto ordinal = index(kind);
  if (!ordinal.ok()) return ordinal.status();
  return plans_[ordinal.value()]->algorithm_id();
}

}  // namespace pih
