#include "attention_reduce.h"

namespace pih::deepseek_v41 {
Status ValidateAttentionReduction(const AttentionLocalOutputLaunch& x, std::uint32_t rank, std::uintptr_t communicator) {
  const auto layout = ValidateAttentionLocalOutput(x); if (!layout.ok()) return layout;
  return ValidateFp32Reduction({x.reduction, x.linear.stream, 8 / x.grouped.projection.groups, rank}, communicator);
}
Result<EngramReduction> ReduceAttentionOutput(const AttentionLocalOutputLaunch& x, std::uint32_t rank, std::uintptr_t communicator) {
  const auto layout = ValidateAttentionLocalOutput(x); if (!layout.ok()) return layout;
  return EngramReduction::SubmitFp32({x.reduction, x.linear.stream, 8 / x.grouped.projection.groups, rank}, communicator);
}
}  // namespace pih::deepseek_v41
