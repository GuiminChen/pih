#include "compressed_prepare.h"
#include <vector>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateCompressedPrepare(const FlashConfig& config, const CompressedPrepareLaunch& x) {
  if (config.config_sha256() == Sha256Digest{} || x.layer >= FlashConfig::kMainLayers)
    return Status::InvalidArgument("Compressed preparation requires admitted backbone layer");
  const auto& role = config.attention_sharing()[x.layer];
  if (!role.owns_kv || !role.compression_ratio || role.kv_source != x.layer)
    return Status::InvalidArgument("Layer does not own compressed KV production");
  const auto compressor = ValidateCompressor(x.compressor); if (!compressor.ok()) return compressor;
  const auto& p = x.compressor.projection;
  const auto step = CompressorOutputRows(x.start, p.tokens); if (!step.ok()) return step.status();
  if (p.ratio != role.compression_ratio || (x.compressor.pooled && x.compressor.pooled->pool.start != x.start))
    return Status::InvalidArgument("Compressed preparation ratio or step mismatch");
  const unsigned rows = (x.start + p.tokens) / p.ratio - x.start / p.ratio;
  if (bool(x.cache) != (rows != 0)) return Status::InvalidArgument("Compressed cache stage must match emitted rows");
  if (!rows) return Status::Ok();
  const auto cache = ValidateCompressedKvPrepare(*x.cache); if (!cache.ok()) return cache;
  const auto& n = x.compressor.direct_norm ? *x.compressor.direct_norm : *x.compressor.pooled->norm;
  const auto& r = x.cache->rope; const auto& c = x.cache->cache;
  if (r.tokens != rows || c.rows != rows || c.first_slot != x.start / p.ratio ||
      r.stream != p.stream || !Same(r.input, n.output) || !Same(r.error_flag, p.error_flag))
    return Status::InvalidArgument("Compressed latent, cache slot or stream connection mismatch");
  const std::array reads{p.input, p.value_weight, p.gate_weight, n.weight, r.phases};
  std::vector<EngramDeviceRegion> writes{p.values, p.scores, n.output, r.output, c.cache, p.error_flag};
  if (x.compressor.pooled) {
    const auto& pool = x.compressor.pooled->pool;
    writes.push_back(pool.state_values); writes.push_back(pool.state_scores); writes.push_back(pool.output);
  }
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads) if (Overlap(writes[i], read))
      return Status::InvalidArgument("Compressed preparation overwrites live input or weight");
    for (std::size_t j = 0; j < i; ++j) if (Overlap(writes[i], writes[j]))
      return Status::InvalidArgument("Compressed preparation writable alias");
  }
  return Status::Ok();
}
Status LaunchCompressedPrepare(const FlashConfig& config, const CompressedPrepareLaunch& x) {
  const auto validation = ValidateCompressedPrepare(config, x); if (!validation.ok()) return validation;
  const auto compressor = LaunchCompressor(x.compressor); if (!compressor.ok()) return compressor;
  return x.cache ? LaunchCompressedKvPrepare(*x.cache) : Status::Ok();
}
}  // namespace pih::deepseek_v41
