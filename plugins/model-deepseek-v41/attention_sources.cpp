#include "attention_sources.h"
#include <vector>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateAttentionSources(const FlashConfig& config, const AttentionSourcesLaunch& x) {
  const auto attention = ValidateAttentionPrepare(config, x.attention); if (!attention.ok()) return attention;
  const auto& a = x.attention; const auto& role = config.attention_sharing()[a.layer];
  if (bool(x.compressed) != (role.owns_kv && role.compression_ratio != 0))
    return Status::InvalidArgument("Attention compressed producer presence differs from layer ownership");
  if (!x.compressed) return Status::Ok();
  const auto compressed = ValidateCompressedPrepare(config, *x.compressed); if (!compressed.ok()) return compressed;
  const auto& cp = *x.compressed; const auto& p = cp.compressor.projection;
  const auto& m = a.input.mix; const auto& c = a.input.input.collapse; const auto& n = a.input.input.norm;
  const auto& q = a.query; const auto& w = a.window;
  if (cp.layer != a.layer || cp.start != w.prepare.cache.start || p.tokens != m.tokens || p.stream != m.stream ||
      !Same(p.input, n.output) || !Same(p.error_flag, m.error_flag))
    return Status::InvalidArgument("Attention compressor source, layer or step connection mismatch");
  const std::array attention_reads{m.residual, m.fn, m.scale, m.base, c.pre, n.weight,
      q.low_rank.weight, q.low_rank.weight_scales, q.norm.weight, q.expand.weight, q.expand.weight_scales,
      q.rope.phases, w.projection.weight, w.projection.weight_scales, w.prepare.norm.weight};
  // Internal RoPE exact aliases are already admitted by the component validator.
  const std::array attention_writes{m.pre, m.post, m.comb, c.output, n.output,
      q.low_rank.quantized, q.low_rank.activation_scales, q.low_rank.output, q.norm.output,
      q.expand.quantized, q.expand.activation_scales, q.expand.output, q.rope.output,
      w.projection.quantized, w.projection.activation_scales, w.projection.output,
      w.prepare.norm.output, w.prepare.rope.output, w.prepare.cache.ring};
  std::vector<EngramDeviceRegion> compressed_reads{p.value_weight, p.gate_weight};
  std::vector<EngramDeviceRegion> compressed_writes{p.values, p.scores};
  if (cp.compressor.direct_norm) {
    compressed_reads.push_back(cp.compressor.direct_norm->weight);
    compressed_writes.push_back(cp.compressor.direct_norm->output);
  } else {
    const auto& pool = cp.compressor.pooled->pool;
    compressed_writes.push_back(pool.state_values); compressed_writes.push_back(pool.state_scores);
    compressed_writes.push_back(pool.output);
    if (cp.compressor.pooled->norm) {
      compressed_reads.push_back(cp.compressor.pooled->norm->weight);
      compressed_writes.push_back(cp.compressor.pooled->norm->output);
    }
  }
  if (cp.cache) {
    compressed_reads.push_back(cp.cache->rope.phases);
    compressed_writes.push_back(cp.cache->rope.output); compressed_writes.push_back(cp.cache->cache.cache);
  }
  for (const auto write : attention_writes) {
    for (const auto read : compressed_reads) if (Overlap(write, read))
      return Status::InvalidArgument("Attention source overwrites compressor weight or phase");
    for (const auto other : compressed_writes) if (Overlap(write, other))
      return Status::InvalidArgument("Attention and compressor writable storage overlap");
  }
  for (const auto write : compressed_writes) for (const auto read : attention_reads) if (Overlap(write, read))
    return Status::InvalidArgument("Compression overwrites retained attention input or weight");
  // The common error flag is the only shared writer, and must be disjoint from
  // both graphs' live inputs and all other writable regions.
  for (const auto read : compressed_reads) if (Overlap(m.error_flag, read))
    return Status::InvalidArgument("Attention error flag overwrites compressor input");
  for (const auto write : compressed_writes) if (Overlap(m.error_flag, write))
    return Status::InvalidArgument("Compressor output overwrites common error flag");
  return Status::Ok();
}
Status LaunchAttentionSources(const FlashConfig& config, const AttentionSourcesLaunch& x) {
  const auto validation = ValidateAttentionSources(config, x); if (!validation.ok()) return validation;
  const auto attention = LaunchAttentionPrepare(config, x.attention); if (!attention.ok()) return attention;
  return x.compressed ? LaunchCompressedPrepare(config, *x.compressed) : Status::Ok();
}
}  // namespace pih::deepseek_v41
