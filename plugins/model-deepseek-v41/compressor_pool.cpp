#include "compressor_pool.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion x, std::uint64_t bytes, unsigned alignment) {
  if (!bytes) return !x.address && !x.bytes;
  return x.address && x.address % alignment == 0 && x.bytes == bytes &&
      bytes <= std::numeric_limits<std::uintptr_t>::max() - x.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
template<std::size_t I, std::size_t W>
bool Disjoint(const std::array<EngramDeviceRegion, I>& inputs, const std::array<EngramDeviceRegion, W>& writes) {
  for (std::size_t i = 0; i < W; ++i) {
    for (const auto input : inputs) if (Overlap(input, writes[i])) return false;
    for (std::size_t j = 0; j < i; ++j) if (Overlap(writes[i], writes[j])) return false;
  }
  return true;
}
}
Result<std::uint32_t> CompressorOutputRows(std::uint32_t start, std::uint32_t tokens) {
  if (!tokens || tokens > 4096 || (start && tokens != 1) || start >= 1048576 || tokens > 1048576 - start)
    return Status::InvalidArgument("V4.1 ratio-two compressor step invalid");
  if (!start) return tokens / 2;
  return (start + 1) % 2 == 0 ? 1U : 0U;
}
Status ValidateCompressorPool(const CompressorPoolLaunch& x) {
  const auto rows = CompressorOutputRows(x.start, x.tokens); if (!rows.ok()) return rows.status();
  if (!x.stream || !Valid(x.values, x.tokens * 512ULL * 4, 4) || !Valid(x.scores, x.tokens * 512ULL * 4, 4) ||
      !Valid(x.state_values, 2 * 512 * 4, 4) || !Valid(x.state_scores, 2 * 512 * 4, 4) ||
      !Valid(x.output, *rows * 512ULL * 2, 2) || !Valid(x.error_flag, 4, 4) ||
      !Disjoint(std::array{x.values, x.scores}, std::array{x.state_values, x.state_scores, x.output, x.error_flag}))
    return Status::InvalidArgument("V4.1 compressor pool buffer contract invalid");
  return Status::Ok();
}
Status ValidateCompressorNormalize(const CompressorNormalizeLaunch& x) {
  const auto pool = ValidateCompressorPool(x.pool); if (!pool.ok()) return pool;
  const auto rows = CompressorOutputRows(x.pool.start, x.pool.tokens);
  if (!*rows) return x.norm ? Status::InvalidArgument("Incomplete compressor step must not normalize") : Status::Ok();
  if (!x.norm) return Status::InvalidArgument("Completed compressor step requires normalization");
  const auto norm = ValidateRmsNorm(*x.norm); if (!norm.ok()) return norm;
  if (x.norm->rows != *rows || x.norm->width != 512 || x.norm->stream != x.pool.stream ||
      !Same(x.norm->input, x.pool.output) || !Same(x.norm->error_flag, x.pool.error_flag) ||
      !Disjoint(std::array{x.pool.values, x.pool.scores, x.norm->weight},
          std::array{x.pool.state_values, x.pool.state_scores, x.pool.output, x.norm->output, x.pool.error_flag}))
    return Status::InvalidArgument("V4.1 compressor normalization connection or alias invalid");
  return Status::Ok();
}
Status LaunchCompressorNormalize(const CompressorNormalizeLaunch& x) {
  const auto validation = ValidateCompressorNormalize(x); if (!validation.ok()) return validation;
  const auto pool = LaunchCompressorPool(x.pool); if (!pool.ok()) return pool;
  return x.norm ? LaunchRmsNorm(*x.norm) : Status::Ok();
}
Status ValidateCompressorProjection(const CompressorProjectionLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 || (x.ratio != 1 && x.ratio != 2))
    return Status::InvalidArgument("V4.1 compressor projection geometry invalid");
  const unsigned element = x.ratio == 1 ? 2 : 4;
  const auto weight_bytes = 512ULL * 5120 * element;
  const auto output_bytes = x.tokens * 512ULL * element;
  if (!Valid(x.input, x.tokens * 5120ULL * 2, 2) ||
      !Valid(x.value_weight, weight_bytes, element) ||
      !Valid(x.gate_weight, x.ratio == 2 ? weight_bytes : 0, element) ||
      !Valid(x.values, output_bytes, element) ||
      !Valid(x.scores, x.ratio == 2 ? output_bytes : 0, element) || !Valid(x.error_flag, 4, 4) ||
      !Disjoint(std::array{x.input, x.value_weight, x.gate_weight}, std::array{x.values, x.scores, x.error_flag}))
    return Status::InvalidArgument("V4.1 compressor projection buffer contract invalid");
  return Status::Ok();
}
Status ValidateCompressor(const CompressorLaunch& x) {
  const auto projection = ValidateCompressorProjection(x.projection); if (!projection.ok()) return projection;
  const auto& p = x.projection;
  if (p.ratio == 1) {
    if (!x.direct_norm || x.pooled) return Status::InvalidArgument("Ratio-one compressor requires direct normalization only");
    const auto& n = *x.direct_norm;
    const auto norm = ValidateRmsNorm(n); if (!norm.ok()) return norm;
    if (n.rows != p.tokens || n.width != 512 || n.stream != p.stream ||
        !Same(n.input, p.values) || !Same(n.error_flag, p.error_flag) ||
        !Disjoint(std::array{p.input, p.value_weight, n.weight}, std::array{p.values, n.output, p.error_flag}))
      return Status::InvalidArgument("Ratio-one compressor connection or alias invalid");
  } else {
    if (x.direct_norm || !x.pooled) return Status::InvalidArgument("Ratio-two compressor requires pooling only");
    const auto pooled = ValidateCompressorNormalize(*x.pooled); if (!pooled.ok()) return pooled;
    const auto& pool = x.pooled->pool;
    const auto& norm = x.pooled->norm;
    if (pool.tokens != p.tokens || pool.stream != p.stream || !Same(pool.values, p.values) ||
        !Same(pool.scores, p.scores) || !Same(pool.error_flag, p.error_flag) ||
        !Disjoint(std::array{p.input, p.value_weight, p.gate_weight, norm ? norm->weight : EngramDeviceRegion{}},
          std::array{p.values, p.scores, pool.state_values, pool.state_scores, pool.output,
                     norm ? norm->output : EngramDeviceRegion{}, p.error_flag}))
      return Status::InvalidArgument("Ratio-two compressor connection or alias invalid");
  }
  return Status::Ok();
}
Status LaunchCompressor(const CompressorLaunch& x) {
  const auto validation = ValidateCompressor(x); if (!validation.ok()) return validation;
  const auto projection = LaunchCompressorProjection(x.projection); if (!projection.ok()) return projection;
  return x.direct_norm ? LaunchRmsNorm(*x.direct_norm) : LaunchCompressorNormalize(*x.pooled);
}
}  // namespace pih::deepseek_v41
