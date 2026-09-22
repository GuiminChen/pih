#include "attention_residual.h"

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateAttentionResidual(const AttentionResidualLaunch& x) {
  const auto attention = ValidateAssembledAttentionOutput(x.attention); if (!attention.ok()) return attention;
  const auto residual = ValidateMhcPost(x.residual); if (!residual.ok()) return residual;
  const auto& a = x.attention.assembly;
  const auto& o = x.attention.output;
  const auto& r = x.residual;
  if (r.tokens != a.tokens || r.stream != a.stream || !Same(r.sublayer, o.linear.output) || !Same(r.error_flag, a.error_flag))
    return Status::InvalidArgument("Attention residual stage connection invalid");
  // Check cross-stage aliases in addition to both complete component graphs.
  const std::array attention_reads{a.window, a.compressed, a.selected, o.grouped.attention.query,
      o.grouped.attention.sink, o.grouped.inverse_rope.phases, o.grouped.projection.weight, o.linear.weight, o.linear.weight_scales};
  const std::array attention_writes{a.kv, a.indices, o.grouped.attention.output, o.grouped.inverse_rope.output,
      o.grouped.projection.output, o.linear.quantized, o.linear.activation_scales, o.linear.output, o.reduction, a.error_flag};
  for (const auto read : attention_reads)
    if (Overlap(read, r.output)) return Status::InvalidArgument("Residual output overwrites attention input");
  for (const auto write : attention_writes) {
    if (Overlap(write, r.output)) return Status::InvalidArgument("Residual output aliases attention scratch");
    for (const auto read : std::array{r.residual, r.post, r.comb})
      if (Overlap(write, read)) return Status::InvalidArgument("Attention overwrites residual coefficients or state");
  }
  return Status::Ok();
}
Status LaunchAttentionResidualSingleRank(const AttentionResidualLaunch& x) {
  const auto validation = ValidateAttentionResidual(x); if (!validation.ok()) return validation;
  if (x.attention.output.grouped.projection.groups != 8)
    return Status::InvalidArgument("Single-rank attention residual requires all eight groups");
  const auto attention = LaunchAssembledAttentionOutput(x.attention); if (!attention.ok()) return attention;
  return LaunchMhcPost(x.residual);
}
}  // namespace pih::deepseek_v41
