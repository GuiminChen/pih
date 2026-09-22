#include "expert_fp8.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion x, std::uint64_t bytes, unsigned alignment) {
  if (!bytes) return !x.address && !x.bytes;
  return x.address && x.address % alignment == 0 && x.bytes == bytes && bytes <= std::numeric_limits<std::uintptr_t>::max() - x.address;
}
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
template<std::size_t I, std::size_t W>
bool Disjoint(const std::array<EngramDeviceRegion, I>& reads, const std::array<EngramDeviceRegion, W>& writes) {
  for (std::size_t i = 0; i < W; ++i) {
    for (const auto read : reads) if (Overlap(read, writes[i])) return false;
    for (std::size_t j = 0; j < i; ++j) if (Overlap(writes[i], writes[j])) return false;
  }
  return true;
}
}
Status ValidateExpertActivation(const ExpertActivationLaunch& x) {
  const bool routed = x.route_weights.address || x.route_weights.bytes;
  if (!x.stream || !x.rows || x.rows > 4096 || !Valid(x.gate, x.rows * 2304ULL * 2, 2) ||
      !Valid(x.up, x.rows * 2304ULL * 2, 2) || !Valid(x.output, x.rows * 2304ULL * 2, 2) ||
      !Valid(x.route_weights, routed ? x.rows * 4ULL : 0, 4) || !Valid(x.error_flag, 4, 4) ||
      !Disjoint(std::array{x.gate, x.up, x.route_weights}, std::array{x.output, x.error_flag}))
    return Status::InvalidArgument("V4.1 expert activation layout or alias invalid");
  return Status::Ok();
}
namespace {
Status ValidateProjection(const Fp8LinearLaunch& x) { return ValidateFp8Linear(x); }
Status ValidateProjection(const Fp4LinearLaunch& x) { return ValidateFp4Linear(x); }
template<class Expert>
Status ValidateExpertChain(const Expert& x, bool routed) {
  for (const auto* projection : {&x.gate, &x.up, &x.down}) {
    const auto valid = ValidateProjection(*projection); if (!valid.ok()) return valid;
  }
  const auto activation = ValidateExpertActivation(x.activation); if (!activation.ok()) return activation;
  const auto& g = x.gate; const auto& u = x.up; const auto& d = x.down; const auto& a = x.activation;
  if (g.in_features != 5120 || u.in_features != 5120 || g.out_features != 2304 || u.out_features != 2304 ||
      d.in_features != 2304 || d.out_features != 5120 || u.rows != g.rows || d.rows != g.rows || a.rows != g.rows ||
      u.stream != g.stream || d.stream != g.stream || a.stream != g.stream ||
      !Same(g.input, u.input) || !Same(g.output, a.gate) || !Same(u.output, a.up) || !Same(a.output, d.input) ||
      !Same(g.error_flag, u.error_flag) || !Same(g.error_flag, d.error_flag) || !Same(g.error_flag, a.error_flag) ||
      bool(a.route_weights.address || a.route_weights.bytes) != routed ||
      !Disjoint(std::array{g.input, g.weight, g.weight_scales, u.weight, u.weight_scales, d.weight, d.weight_scales, a.route_weights},
          std::array{g.quantized, g.activation_scales, g.output, u.quantized, u.activation_scales, u.output,
                     a.output, d.quantized, d.activation_scales, d.output, g.error_flag}))
    return Status::InvalidArgument("V4.1 expert connection, routing weight or live-buffer alias invalid");
  return Status::Ok();
}
}
Status ValidateSharedExpert(const SharedExpertLaunch& x) { return ValidateExpertChain(x, false); }
Status ValidateRoutedExpert(const RoutedExpertLaunch& x) { return ValidateExpertChain(x, true); }
Status LaunchSharedExpert(const SharedExpertLaunch& x) {
  const auto validation = ValidateSharedExpert(x); if (!validation.ok()) return validation;
  const auto gate = LaunchFp8Linear(x.gate); if (!gate.ok()) return gate;
  const auto up = LaunchFp8Linear(x.up); if (!up.ok()) return up;
  const auto activation = LaunchExpertActivation(x.activation); if (!activation.ok()) return activation;
  return LaunchFp8Linear(x.down);
}
Status LaunchRoutedExpert(const RoutedExpertLaunch& x) {
  const auto validation = ValidateRoutedExpert(x); if (!validation.ok()) return validation;
  const auto gate = LaunchFp4Linear(x.gate); if (!gate.ok()) return gate;
  const auto up = LaunchFp4Linear(x.up); if (!up.ok()) return up;
  const auto activation = LaunchExpertActivation(x.activation); if (!activation.ok()) return activation;
  return LaunchFp4Linear(x.down);
}
Status ValidateExpertMerge(const ExpertMergeLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 ||
      !Valid(x.routed, x.tokens * 5120ULL * 4, 4) || !Valid(x.shared, x.tokens * 5120ULL * 2, 2) ||
      !Valid(x.output, x.tokens * 5120ULL * 2, 2) || !Valid(x.error_flag, 4, 4) ||
      !Disjoint(std::array{x.routed, x.shared}, std::array{x.output, x.error_flag}))
    return Status::InvalidArgument("V4.1 expert merge layout or writable alias invalid");
  return Status::Ok();
}
Status ValidateExpertSharedMerge(const ExpertSharedMergeLaunch& x) {
  const auto shared = ValidateSharedExpert(x.shared); if (!shared.ok()) return shared;
  const auto merge = ValidateExpertMerge(x.merge); if (!merge.ok()) return merge;
  const auto& s = x.shared; const auto& m = x.merge;
  if (s.gate.rows != m.tokens || s.gate.stream != m.stream ||
      !Same(s.gate.error_flag, m.error_flag) || !Same(s.down.output, m.shared))
    return Status::InvalidArgument("Shared expert merge connection mismatch");
  const std::array reads{s.gate.input, s.gate.weight, s.gate.weight_scales, s.up.weight,
      s.up.weight_scales, s.down.weight, s.down.weight_scales, m.routed};
  const std::array writes{s.gate.quantized, s.gate.activation_scales, s.gate.output,
      s.up.quantized, s.up.activation_scales, s.up.output, s.activation.output,
      s.down.quantized, s.down.activation_scales, s.down.output, m.output, m.error_flag};
  // Routed storage may be overwritten by the preceding in-place all-reduce.
  for (std::size_t i = 0; i + 1 < reads.size(); ++i)
    if (Overlap(m.routed, reads[i])) return Status::InvalidArgument("Routed reduction aliases shared expert input or weight");
  if (!Disjoint(reads, writes)) return Status::InvalidArgument("Shared expert merge live-buffer alias");
  return Status::Ok();
}
Status LaunchExpertSharedMerge(const ExpertSharedMergeLaunch& x) {
  const auto validation = ValidateExpertSharedMerge(x); if (!validation.ok()) return validation;
  const auto shared = LaunchSharedExpert(x.shared); if (!shared.ok()) return shared;
  return LaunchExpertMerge(x.merge);
}
}  // namespace pih::deepseek_v41
