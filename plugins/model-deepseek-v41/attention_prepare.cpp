#include "attention_prepare.h"
#include <vector>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateAttentionPrepare(const FlashConfig& config, const AttentionPrepareLaunch& x) {
  if (config.config_sha256() == Sha256Digest{} || x.layer >= FlashConfig::kMainLayers ||
      (x.world_size != 1 && x.world_size != 2 && x.world_size != 4 && x.world_size != 8))
    return Status::InvalidArgument("Attention preparation requires admitted backbone geometry");
  const auto input = ValidateMhcSublayerInput(x.input); if (!input.ok()) return input;
  const auto query = ValidateAttentionQuery(x.query); if (!query.ok()) return query;
  const auto window = ValidateAttentionWindow(x.window); if (!window.ok()) return window;
  const auto& m = x.input.mix; const auto& c = x.input.input.collapse; const auto& n = x.input.input.norm;
  const auto& q = x.query; const auto& w = x.window;
  if (q.rope.heads != 64U / x.world_size || q.low_rank.rows != m.tokens || w.projection.rows != m.tokens ||
      q.low_rank.stream != m.stream || w.projection.stream != m.stream ||
      !Same(n.output, q.low_rank.input) || !Same(n.output, w.projection.input) ||
      !Same(m.error_flag, q.low_rank.error_flag) || !Same(m.error_flag, w.projection.error_flag) ||
      !Same(q.rope.phases, w.prepare.rope.phases))
    return Status::InvalidArgument("Attention preparation hidden, step, rank or phase connection mismatch");
  const std::array reads{m.residual, m.fn, m.scale, m.base, c.pre, n.weight,
      q.low_rank.weight, q.low_rank.weight_scales, q.norm.weight, q.expand.weight, q.expand.weight_scales,
      q.rope.phases, w.projection.weight, w.projection.weight_scales, w.prepare.norm.weight};
  std::vector<EngramDeviceRegion> writes{m.pre, m.post, m.comb, c.output, n.output,
      q.low_rank.quantized, q.low_rank.activation_scales, q.low_rank.output, q.norm.output,
      q.expand.quantized, q.expand.activation_scales, q.expand.output,
      w.projection.quantized, w.projection.activation_scales, w.projection.output,
      w.prepare.norm.output, w.prepare.cache.ring, m.error_flag};
  // Only these two existing exact in-place RoPE edges are allowed. Other
  // outputs, including qr and the mHC carry coefficients, remain live.
  if (!Same(q.expand.output, q.rope.output)) writes.push_back(q.rope.output);
  if (!Same(w.prepare.norm.output, w.prepare.rope.output)) writes.push_back(w.prepare.rope.output);
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads) if (Overlap(writes[i], read))
      return Status::InvalidArgument("Attention preparation overwrites live input or weight");
    for (std::size_t j = 0; j < i; ++j) if (Overlap(writes[i], writes[j]))
      return Status::InvalidArgument("Attention preparation cross-stage writable alias");
  }
  return Status::Ok();
}
Status LaunchAttentionPrepare(const FlashConfig& config, const AttentionPrepareLaunch& x) {
  const auto validation = ValidateAttentionPrepare(config, x); if (!validation.ok()) return validation;
  const auto input = LaunchMhcSublayerInput(x.input); if (!input.ok()) return input;
  const auto query = LaunchAttentionQuery(x.query); if (!query.ok()) return query;
  return LaunchAttentionWindow(x.window);
}
}  // namespace pih::deepseek_v41
