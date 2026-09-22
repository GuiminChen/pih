#include "attention_assemble.h"
#include <algorithm>
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion x, std::uint64_t bytes, unsigned alignment) {
  if (!bytes) return !x.address && !x.bytes;
  return x.address && x.address % alignment == 0 && x.bytes == bytes && bytes <= std::numeric_limits<std::uintptr_t>::max() - x.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
template<std::size_t I, std::size_t W>
bool Disjoint(const std::array<EngramDeviceRegion, I>& reads, const std::array<EngramDeviceRegion, W>& writes) {
  for (std::size_t i = 0; i < W; ++i) {
    for (const auto read : reads) if (Overlap(read, writes[i])) return false;
    for (std::size_t j = 0; j < i; ++j) if (Overlap(writes[i], writes[j])) return false;
  }
  return true;
}
}
Result<AttentionAssemblyShape> GetAttentionAssemblyShape(std::uint32_t start, std::uint32_t tokens, std::uint32_t ratio) {
  if (!tokens || tokens > 4096 || (start && tokens != 1) || start >= 1048576 || tokens > 1048576 - start || ratio > 2)
    return Status::InvalidArgument("V4.1 attention assembly step invalid");
  const unsigned compressed = ratio ? (start + tokens) / ratio : 0;
  return AttentionAssemblyShape{start ? 128U : tokens, compressed, start ? 128U : std::min(tokens, 128U), std::min(compressed, 512U)};
}
Status ValidateAttentionAssembly(const AttentionAssemblyLaunch& x) {
  const auto shape = GetAttentionAssemblyShape(x.start, x.tokens, x.ratio); if (!shape.ok()) return shape.status();
  const auto& s = *shape;
  if (!x.stream || !Valid(x.window, s.window_rows * 512ULL * 2, 2) || !Valid(x.compressed, s.compressed_rows * 512ULL * 2, 2) ||
      !Valid(x.selected, x.tokens * std::uint64_t(s.compressed_columns) * 4, 4) ||
      !Valid(x.kv, (s.window_rows + s.compressed_rows) * 512ULL * 2, 2) ||
      !Valid(x.indices, x.tokens * std::uint64_t(s.window_columns + s.compressed_columns) * 4, 4) || !Valid(x.error_flag, 4, 4) ||
      !Disjoint(std::array{x.window, x.compressed, x.selected}, std::array{x.kv, x.indices, x.error_flag}))
    return Status::InvalidArgument("V4.1 attention assembly layout or alias invalid");
  return Status::Ok();
}
Status ValidateAssembledAttention(const AssembledAttentionLaunch& x) {
  const auto assembly = ValidateAttentionAssembly(x.assembly); if (!assembly.ok()) return assembly;
  const auto attention = ValidateSparseAttention(x.attention); if (!attention.ok()) return attention;
  const auto& a = x.assembly; const auto& b = x.attention;
  const auto s = *GetAttentionAssemblyShape(a.start, a.tokens, a.ratio);
  if (b.tokens != a.tokens || b.stream != a.stream || b.kv_rows != s.window_rows + s.compressed_rows ||
      b.picks != s.window_columns + s.compressed_columns || !Same(b.kv, a.kv) || !Same(b.indices, a.indices) ||
      !Same(b.error_flag, a.error_flag) ||
      !Disjoint(std::array{a.window, a.compressed, a.selected, b.query, b.sink}, std::array{a.kv, a.indices, b.output, a.error_flag}))
    return Status::InvalidArgument("V4.1 assembled attention connection or alias invalid");
  return Status::Ok();
}
Status LaunchAssembledAttention(const AssembledAttentionLaunch& x) {
  const auto validation = ValidateAssembledAttention(x); if (!validation.ok()) return validation;
  const auto assembly = LaunchAttentionAssembly(x.assembly); if (!assembly.ok()) return assembly;
  return LaunchSparseAttention(x.attention);
}
Status ValidateAssembledAttentionOutput(const AssembledAttentionOutputLaunch& x) {
  const auto output = ValidateAttentionLocalOutput(x.output); if (!output.ok()) return output;
  const auto& a = x.assembly;
  const auto& b = x.output.grouped.attention;
  const auto assembled = ValidateAssembledAttention({a, b}); if (!assembled.ok()) return assembled;
  const auto& r = x.output.grouped.inverse_rope;
  const auto& p = x.output.grouped.projection;
  const auto& l = x.output.linear;
  const std::array reads{a.window, a.compressed, a.selected, b.query, b.sink, r.phases,
      p.weight, l.weight, l.weight_scales};
  const std::array writes{a.kv, a.indices, b.output, r.output, p.output,
      l.quantized, l.activation_scales, l.output, x.output.reduction, a.error_flag};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads)
      if (Overlap(read, writes[i])) return Status::InvalidArgument("Assembled output overwrites live input");
    for (std::size_t j = 0; j < i; ++j)
      if (Overlap(writes[i], writes[j]) && !(i == 3 && j == 2 && Same(writes[i], writes[j])))
        return Status::InvalidArgument("Assembled output writable alias");
  }
  return Status::Ok();
}
Status LaunchAssembledAttentionOutput(const AssembledAttentionOutputLaunch& x) {
  const auto validation = ValidateAssembledAttentionOutput(x); if (!validation.ok()) return validation;
  const auto assembly = LaunchAttentionAssembly(x.assembly); if (!assembly.ok()) return assembly;
  return LaunchAttentionLocalOutput(x.output);
}
}  // namespace pih::deepseek_v41
