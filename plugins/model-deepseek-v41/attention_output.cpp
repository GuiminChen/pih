#include "attention_output.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion x, std::uint64_t bytes, unsigned alignment) {
  return x.address && x.address % alignment == 0 && x.bytes == bytes &&
      bytes <= std::numeric_limits<std::uintptr_t>::max() - x.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
}
Status ValidateGroupedOutput(const GroupedOutputLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 ||
      (x.groups != 1 && x.groups != 2 && x.groups != 4 && x.groups != 8) ||
      !Valid(x.input, x.tokens * static_cast<std::uint64_t>(x.groups) * 4096 * 2, 2) ||
      !Valid(x.weight, x.groups * 1024ULL * 4096 * 2, 2) ||
      !Valid(x.output, x.tokens * static_cast<std::uint64_t>(x.groups) * 1024 * 2, 2) ||
      !Valid(x.error_flag, 4, 4) || Overlap(x.output, x.input) || Overlap(x.output, x.weight) ||
      Overlap(x.error_flag, x.input) || Overlap(x.error_flag, x.weight) || Overlap(x.error_flag, x.output))
    return Status::InvalidArgument("V4.1 grouped output projection contract invalid");
  return Status::Ok();
}
Status ValidateAttentionOutput(const AttentionOutputLaunch& x) {
  const auto attention = ValidateSparseAttention(x.attention); if (!attention.ok()) return attention;
  const auto rope = ValidateRopeApply(x.inverse_rope); if (!rope.ok()) return rope;
  const auto projection = ValidateGroupedOutput(x.projection); if (!projection.ok()) return projection;
  if (!x.inverse_rope.inverse || x.inverse_rope.width != 512 ||
      x.attention.heads != x.projection.groups * 8 || x.inverse_rope.heads != x.attention.heads ||
      x.projection.tokens != x.attention.tokens || x.inverse_rope.tokens != x.attention.tokens ||
      x.projection.stream != x.attention.stream || x.inverse_rope.stream != x.attention.stream ||
      !Same(x.attention.output, x.inverse_rope.input) || !Same(x.inverse_rope.output, x.projection.input) ||
      !Same(x.attention.error_flag, x.projection.error_flag) || !Same(x.attention.error_flag, x.inverse_rope.error_flag))
    return Status::InvalidArgument("V4.1 attention output stage connection invalid");
  const std::array inputs{x.attention.query, x.attention.kv, x.attention.sink, x.attention.indices,
      x.inverse_rope.phases, x.projection.weight};
  const std::array writes{x.attention.output, x.inverse_rope.output, x.projection.output, x.attention.error_flag};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto input : inputs)
      if (Overlap(writes[i], input)) return Status::InvalidArgument("Attention output overwrites a live input");
    for (std::size_t j = 0; j < i; ++j)
      if (Overlap(writes[i], writes[j]) && !(i == 1 && j == 0 && Same(writes[i], writes[j])))
        return Status::InvalidArgument("Attention output writable regions overlap");
  }
  return Status::Ok();
}
Status LaunchAttentionOutput(const AttentionOutputLaunch& x) {
  const auto validation = ValidateAttentionOutput(x); if (!validation.ok()) return validation;
  const auto attention = LaunchSparseAttention(x.attention); if (!attention.ok()) return attention;
  const auto rope = LaunchRopeApply(x.inverse_rope); if (!rope.ok()) return rope;
  return LaunchGroupedOutput(x.projection);
}
Status ValidateAttentionLocalOutput(const AttentionLocalOutputLaunch& x) {
  const auto grouped = ValidateAttentionOutput(x.grouped); if (!grouped.ok()) return grouped;
  const auto linear = ValidateFp8Linear(x.linear); if (!linear.ok()) return linear;
  const auto& a = x.grouped.attention;
  if (!Valid(x.reduction, static_cast<std::uint64_t>(a.tokens) * 5120 * 4, 4))
    return Status::InvalidArgument("Attention FP32 reduction scratch invalid");
  if (x.linear.rows != a.tokens || x.linear.in_features != x.grouped.projection.groups * 1024 ||
      x.linear.out_features != 5120 || x.linear.stream != a.stream ||
      !Same(x.linear.input, x.grouped.projection.output) || !Same(x.linear.error_flag, a.error_flag))
    return Status::InvalidArgument("V4.1 wo_b stage connection invalid");
  const std::array inputs{a.query, a.kv, a.sink, a.indices, x.grouped.inverse_rope.phases,
      x.grouped.projection.weight, x.linear.weight, x.linear.weight_scales};
  const std::array writes{a.output, x.grouped.inverse_rope.output, x.grouped.projection.output,
      x.linear.quantized, x.linear.activation_scales, x.linear.output, x.reduction, a.error_flag};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto input : inputs)
      if (Overlap(input, writes[i])) return Status::InvalidArgument("Attention wo_b chain overwrites a live input");
    for (std::size_t j = 0; j < i; ++j)
      if (Overlap(writes[i], writes[j]) && !(i == 1 && j == 0 && Same(writes[i], writes[j])))
        return Status::InvalidArgument("Attention wo_b chain writable regions overlap");
  }
  return Status::Ok();
}
Status LaunchAttentionLocalOutput(const AttentionLocalOutputLaunch& x) {
  const auto validation = ValidateAttentionLocalOutput(x); if (!validation.ok()) return validation;
  const auto grouped = LaunchAttentionOutput(x.grouped); if (!grouped.ok()) return grouped;
  const auto linear = LaunchFp8Linear(x.linear); if (!linear.ok()) return linear;
  return PromoteAttentionOutput(x);
}
}  // namespace pih::deepseek_v41
