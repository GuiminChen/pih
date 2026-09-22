#include "attention_input.h"
#include <span>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
// Called only after every underlying launch has checked pointer arithmetic.
bool LiveBuffers(std::span<const EngramDeviceRegion> inputs, std::span<const EngramDeviceRegion> writes,
    std::size_t inplace_first, std::size_t inplace_second) {
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto input : inputs) if (Overlap(input, writes[i])) return false;
    for (std::size_t j = 0; j < i; ++j)
      if (Overlap(writes[i], writes[j]) &&
          !(i == inplace_second && j == inplace_first && Same(writes[i], writes[j]))) return false;
  }
  return true;
}
}
Status ValidateAttentionQuery(const AttentionQueryLaunch& x) {
  const auto low = ValidateFp8Linear(x.low_rank); if (!low.ok()) return low;
  const auto norm = ValidateRmsNorm(x.norm); if (!norm.ok()) return norm;
  const auto expand = ValidateFp8Linear(x.expand); if (!expand.ok()) return expand;
  const auto rope = ValidateRopeApply(x.rope); if (!rope.ok()) return rope;
  const auto& a = x.low_rank;
  if (a.in_features != 5120 || a.out_features != 1280 || x.norm.width != 1280 ||
      x.expand.in_features != 1280 || x.expand.out_features != x.rope.heads * 512 ||
      (x.rope.heads != 8 && x.rope.heads != 16 && x.rope.heads != 32 && x.rope.heads != 64) ||
      x.rope.width != 512 || x.rope.inverse || x.norm.rows != a.rows || x.expand.rows != a.rows ||
      x.rope.tokens != a.rows || x.norm.stream != a.stream || x.expand.stream != a.stream || x.rope.stream != a.stream ||
      !Same(a.output, x.norm.input) || !Same(x.norm.output, x.expand.input) || !Same(x.expand.output, x.rope.input) ||
      !Same(a.error_flag, x.norm.error_flag) || !Same(a.error_flag, x.expand.error_flag) || !Same(a.error_flag, x.rope.error_flag))
    return Status::InvalidArgument("V4.1 query projection stage connection invalid");
  const std::array inputs{a.input, a.weight, a.weight_scales, x.norm.weight,
      x.expand.weight, x.expand.weight_scales, x.rope.phases};
  const std::array writes{a.quantized, a.activation_scales, a.output, x.norm.output,
      x.expand.quantized, x.expand.activation_scales, x.expand.output, x.rope.output, a.error_flag};
  if (!LiveBuffers(inputs, writes, 6, 7)) return Status::InvalidArgument("V4.1 query chain live-buffer alias");
  return Status::Ok();
}
Status LaunchAttentionQuery(const AttentionQueryLaunch& x) {
  const auto validation = ValidateAttentionQuery(x); if (!validation.ok()) return validation;
  const auto low = LaunchFp8Linear(x.low_rank); if (!low.ok()) return low;
  const auto norm = LaunchRmsNorm(x.norm); if (!norm.ok()) return norm;
  const auto expand = LaunchFp8Linear(x.expand); if (!expand.ok()) return expand;
  return LaunchRopeApply(x.rope);
}
Status ValidateAttentionWindow(const AttentionWindowLaunch& x) {
  const auto projection = ValidateFp8Linear(x.projection); if (!projection.ok()) return projection;
  const auto prepare = ValidateWindowKvPrepare(x.prepare); if (!prepare.ok()) return prepare;
  const auto& a = x.projection;
  if (a.in_features != 5120 || a.out_features != 512 || a.rows != x.prepare.cache.tokens ||
      a.stream != x.prepare.cache.stream || !Same(a.output, x.prepare.norm.input) ||
      !Same(a.error_flag, x.prepare.cache.error_flag))
    return Status::InvalidArgument("V4.1 window projection stage connection invalid");
  const std::array inputs{a.input, a.weight, a.weight_scales, x.prepare.norm.weight, x.prepare.rope.phases};
  const std::array writes{a.quantized, a.activation_scales, a.output, x.prepare.norm.output,
      x.prepare.rope.output, x.prepare.cache.ring, a.error_flag};
  if (!LiveBuffers(inputs, writes, 3, 4)) return Status::InvalidArgument("V4.1 window chain live-buffer alias");
  return Status::Ok();
}
Status LaunchAttentionWindow(const AttentionWindowLaunch& x) {
  const auto validation = ValidateAttentionWindow(x); if (!validation.ok()) return validation;
  const auto projection = LaunchFp8Linear(x.projection); if (!projection.ok()) return projection;
  return LaunchWindowKvPrepare(x.prepare);
}
}  // namespace pih::deepseek_v41
