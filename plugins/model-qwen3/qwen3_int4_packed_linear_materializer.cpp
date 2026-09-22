#include "pih/model/qwen3_int4_packed_linear_materializer.h"

namespace pih {

Result<QwenInt4DispatchPlan> QwenInt4PackedLinearMaterializer::Create(
    const QwenBf16PreparedCommand& command,
    const ResolvedKernelFunction& function,
    const QwenBf16PackedResourceSet& resources,
    const QwenInt4WeightResourceSet& weights,
    std::uint64_t request_generation) {
  if (command.linear_kind == QwenBf16LinearKind::kLmHead) {
    return Status::InvalidArgument(
        "packed Qwen LM head is retained BF16, not W4A16");
  }
  const auto bucket = resources.execution_bucket_tokens();
  if (bucket == 0 || bucket > QwenInt4GemmPlan::kMaximumTokens ||
      resources.sequence_count() == 0 ||
      resources.sequence_count() > bucket ||
      resources.sample_count() > resources.sequence_count()) {
    return Status::InvalidArgument("Qwen INT4 packed batch extent is invalid");
  }
  // Non-final prefill chunks append KV without producing a sample. Their
  // projections still execute; the prepared execution omits LM-head/sampler.
  auto hidden =
      resources.activations().view(QwenBf16ActivationSlot::kHidden);
  if (!hidden.ok()) return hidden.status();
  if (hidden->rank() != 2 || hidden->dim(0) != bucket ||
      hidden->dim(1) != 1024) {
    return Status::InvalidArgument(
        "Qwen INT4 packed activation does not span execution bucket");
  }
  return QwenInt4LinearMaterializer::Create(
      command, function, resources.activations(), weights,
      request_generation);
}

}  // namespace pih
