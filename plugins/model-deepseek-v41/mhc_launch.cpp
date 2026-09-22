#include "mhc_launch.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion region, std::uint64_t size, unsigned alignment) {
  return region.address && region.address % alignment == 0 && region.bytes == size &&
      size <= std::numeric_limits<std::uintptr_t>::max() - region.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
template<std::size_t I, std::size_t O>
bool Disjoint(const std::array<EngramDeviceRegion, I>& inputs,
    const std::array<EngramDeviceRegion, O>& outputs) {
  for (std::size_t i = 0; i < O; ++i) {
    for (const auto& input : inputs) if (Overlap(input, outputs[i])) return false;
    for (std::size_t j = 0; j < i; ++j) if (Overlap(outputs[j], outputs[i])) return false;
  }
  return true;
}
bool Sequence(std::uint32_t tokens, std::uintptr_t stream) { return stream && tokens && tokens <= 4096; }
}
Status ValidateMhcMix(const MhcMixLaunch& x) {
  if (!Sequence(x.tokens, x.stream) || !Valid(x.residual, x.tokens * 20480ULL * 2, 2) ||
      !Valid(x.fn, 24ULL * 20480 * 4, 4) || !Valid(x.scale, 3 * 4, 4) || !Valid(x.base, 24 * 4, 4) ||
      !Valid(x.pre, x.tokens * 4ULL * 4, 4) || !Valid(x.post, x.tokens * 4ULL * 4, 4) ||
      !Valid(x.comb, x.tokens * 16ULL * 4, 4) || !Valid(x.error_flag, 4, 4) ||
      !Disjoint(std::array{x.residual, x.fn, x.scale, x.base}, std::array{x.pre, x.post, x.comb, x.error_flag}))
    return Status::InvalidArgument("V4.1 mHC mix buffer/sequence contract invalid");
  return Status::Ok();
}
Status ValidateMhcPre(const MhcPreLaunch& x) {
  if (!Sequence(x.tokens, x.stream) || !Valid(x.residual, x.tokens * 20480ULL * 2, 2) ||
      !Valid(x.pre, x.tokens * 4ULL * 4, 4) || !Valid(x.output, x.tokens * 5120ULL * 2, 2) ||
      !Valid(x.error_flag, 4, 4) || !Disjoint(std::array{x.residual, x.pre}, std::array{x.output, x.error_flag}))
    return Status::InvalidArgument("V4.1 mHC pre buffer/sequence contract invalid");
  return Status::Ok();
}
Status ValidateMhcPost(const MhcPostLaunch& x) {
  if (!Sequence(x.tokens, x.stream) || !Valid(x.sublayer, x.tokens * 5120ULL * 2, 2) ||
      !Valid(x.residual, x.tokens * 20480ULL * 2, 2) || !Valid(x.post, x.tokens * 4ULL * 4, 4) ||
      !Valid(x.comb, x.tokens * 16ULL * 4, 4) || !Valid(x.output, x.tokens * 20480ULL * 2, 2) ||
      !Valid(x.error_flag, 4, 4) ||
      !Disjoint(std::array{x.sublayer, x.residual, x.post, x.comb}, std::array{x.output, x.error_flag}))
    return Status::InvalidArgument("V4.1 mHC post buffer/sequence contract invalid");
  return Status::Ok();
}
Status ValidateMhcInitialPre(EngramDeviceRegion pre, std::uint32_t tokens, std::uintptr_t stream) {
  if (!Sequence(tokens, stream) || !Valid(pre, tokens * 4ULL * 4, 4))
    return Status::InvalidArgument("V4.1 mHC initial pre-mix contract invalid");
  return Status::Ok();
}
Status ValidateMhcInput(const MhcInputLaunch& x) {
  const auto pre = ValidateMhcPre(x.collapse);
  if (!pre.ok()) return pre;
  const auto norm = ValidateRmsNorm(x.norm);
  if (!norm.ok()) return norm;
  const auto same = [](EngramDeviceRegion a, EngramDeviceRegion b) {
    return a.address == b.address && a.bytes == b.bytes;
  };
  if (x.norm.width != 5120 || x.norm.rows != x.collapse.tokens || x.norm.stream != x.collapse.stream ||
      !same(x.norm.input, x.collapse.output) || !same(x.norm.error_flag, x.collapse.error_flag) ||
      !Disjoint(std::array{x.collapse.residual, x.collapse.pre, x.norm.weight},
                std::array{x.collapse.output, x.norm.output, x.norm.error_flag}))
    return Status::InvalidArgument("V4.1 mHC input chain connection or live-buffer alias invalid");
  return Status::Ok();
}
Status LaunchMhcInput(const MhcInputLaunch& x) {
  const auto validation = ValidateMhcInput(x);
  if (!validation.ok()) return validation;
  const auto collapse = LaunchMhcPre(x.collapse);
  if (!collapse.ok()) return collapse;
  return LaunchRmsNorm(x.norm);
}
Status ValidateMhcSublayerInput(const MhcSublayerInputLaunch& x) {
  const auto mix = ValidateMhcMix(x.mix); if (!mix.ok()) return mix;
  const auto input = ValidateMhcInput(x.input); if (!input.ok()) return input;
  const auto& m = x.mix; const auto& c = x.input.collapse; const auto& n = x.input.norm;
  const auto same = [](EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; };
  if (m.tokens != c.tokens || m.stream != c.stream || !same(m.residual, c.residual) || !same(m.error_flag, c.error_flag) ||
      !Disjoint(std::array{m.residual, m.fn, m.scale, m.base, c.pre, n.weight},
          std::array{m.pre, m.post, m.comb, c.output, n.output, m.error_flag}))
    return Status::InvalidArgument("mHC sublayer input connection, carried-pre or live-buffer alias invalid");
  return Status::Ok();
}
Status LaunchMhcSublayerInput(const MhcSublayerInputLaunch& x) {
  const auto validation = ValidateMhcSublayerInput(x); if (!validation.ok()) return validation;
  const auto mix = LaunchMhcMix(x.mix); if (!mix.ok()) return mix;
  return LaunchMhcInput(x.input);
}
}  // namespace pih::deepseek_v41
