#include "expert_residual.h"

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateExpertResidual(const ExpertResidualLaunch& x) {
  const auto experts = ValidateExpertSharedMerge(x.experts); if (!experts.ok()) return experts;
  const auto residual = ValidateMhcPost(x.residual); if (!residual.ok()) return residual;
  const auto& s = x.experts.shared; const auto& m = x.experts.merge; const auto& r = x.residual;
  if (!Same(m.output, r.sublayer) || !Same(m.error_flag, r.error_flag) || m.stream != r.stream || m.tokens != r.tokens)
    return Status::InvalidArgument("Expert residual connection mismatch");
  const std::array reads{s.gate.input, s.gate.weight, s.gate.weight_scales,
      s.up.weight, s.up.weight_scales, s.down.weight, s.down.weight_scales, m.routed};
  const std::array writes{s.gate.quantized, s.gate.activation_scales, s.gate.output,
      s.up.quantized, s.up.activation_scales, s.up.output, s.activation.output,
      s.down.quantized, s.down.activation_scales, s.down.output, m.output, m.error_flag};
  for (const auto read : reads) if (Overlap(r.output, read))
    return Status::InvalidArgument("Expert residual output overwrites live expert input");
  for (const auto write : writes) {
    if (Overlap(r.output, write)) return Status::InvalidArgument("Expert residual output overlaps expert scratch");
    for (const auto input : {r.residual, r.post, r.comb}) if (Overlap(write, input))
      return Status::InvalidArgument("Expert computation overwrites residual input or coefficient");
  }
  // The preceding NCCL operation writes routed storage in place as well.
  for (const auto input : {r.residual, r.post, r.comb}) if (Overlap(m.routed, input))
    return Status::InvalidArgument("Routed reduction overwrites residual input or coefficient");
  return Status::Ok();
}
Status LaunchExpertResidual(const ExpertResidualLaunch& x) {
  const auto validation = ValidateExpertResidual(x); if (!validation.ok()) return validation;
  const auto experts = LaunchExpertSharedMerge(x.experts); if (!experts.ok()) return experts;
  return LaunchMhcPost(x.residual);
}
}  // namespace pih::deepseek_v41
