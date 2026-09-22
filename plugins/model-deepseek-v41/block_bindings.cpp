#include "block_bindings.h"

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
}
Status ValidatePreparedBlockSources(const FlashConfig& config,
    const IndexedSourcesLaunch& sources, const PreparedBlockLaunch& block) {
  const auto source_valid = ValidateIndexedSources(config, sources); if (!source_valid.ok()) return source_valid;
  const auto attention_valid = ValidateAttentionResidual(block.attention); if (!attention_valid.ok()) return attention_valid;
  const auto ffn_valid = ValidateFfnRoute(config, block.ffn); if (!ffn_valid.ok()) return ffn_valid;
  const auto& source = sources.sources.attention; const auto& m = source.input.mix;
  const auto& window = source.window.prepare.cache;
  const auto& a = block.attention.attention.assembly; const auto& o = block.attention.attention.output;
  const auto& r = block.attention.residual; const auto& f = block.ffn;
  if (a.start != window.start || a.tokens != window.tokens || a.stream != window.stream ||
      a.ratio != config.attention_sharing()[source.layer].compression_ratio ||
      f.router.layer != source.layer || f.dispatch.world_size != source.world_size ||
      o.grouped.projection.groups != 8U / source.world_size || !Same(a.error_flag, m.error_flag) ||
      !Same(o.grouped.attention.query, source.query.rope.output) ||
      !Same(o.grouped.inverse_rope.phases, source.query.rope.phases) ||
      !Same(a.window, window.start ? window.ring : window.kv) ||
      !Same(r.residual, m.residual) || !Same(r.post, m.post) || !Same(r.comb, m.comb) ||
      !Same(block.attention_pre, m.pre) || !Same(f.input.input.collapse.pre, m.pre) ||
      !Same(f.input.mix.residual, r.output) || f.dispatch.tokens != a.tokens ||
      f.dispatch.stream != a.stream || !Same(f.dispatch.error_flag, a.error_flag))
    return Status::InvalidArgument("Prepared block source, layer, step or coefficient binding mismatch");
  if (sources.indexer && !Same(a.selected, sources.indexer->selection.output))
    return Status::InvalidArgument("Prepared block does not consume this step's produced indices");
  if (sources.sources.compressed && sources.sources.compressed->cache) {
    const auto& cache = sources.sources.compressed->cache->cache;
    if (a.compressed.address != cache.cache.address || a.compressed.bytes > cache.cache.bytes)
      return Status::InvalidArgument("Prepared block compressed prefix differs from produced cache");
  }
  return Status::Ok();
}
Result<PreparedBlockLaunch> WirePreparedBlockSources(const IndexedSourcesLaunch& sources, PreparedBlockLaunch block) {
  const auto& source = sources.sources.attention; const auto& m = source.input.mix;
  const auto& window = source.window.prepare.cache;
  auto& a = block.attention.attention.assembly; auto& o = block.attention.attention.output;
  auto& r = block.attention.residual; auto& f = block.ffn;
  // Preserve the caller's declared geometry and reject mismatches below; only
  // producer-consumer regions are populated, never inferred weights or caches.
  o.grouped.attention.query = source.query.rope.output;
  o.grouped.inverse_rope.phases = source.query.rope.phases;
  a.window = window.start ? window.ring : window.kv;
  r.residual = m.residual; r.post = m.post; r.comb = m.comb;
  block.attention_pre = m.pre; f.input.input.collapse.pre = m.pre;
  f.input.mix.residual = r.output; f.input.input.collapse.residual = r.output;
  if (sources.indexer) a.selected = sources.indexer->selection.output;
  if (sources.sources.compressed && sources.sources.compressed->cache) {
    const auto shape = GetAttentionAssemblyShape(a.start, a.tokens, a.ratio); if (!shape.ok()) return shape.status();
    a.compressed = {sources.sources.compressed->cache->cache.cache.address, shape->compressed_rows * 512ULL * 2};
  }
  return block;
}
Result<PreparedBlockLaunch> BindPreparedBlockSources(const FlashConfig& config,
    const IndexedSourcesLaunch& sources, PreparedBlockLaunch block) {
  const auto source_valid = ValidateIndexedSources(config, sources); if (!source_valid.ok()) return source_valid;
  auto wired = WirePreparedBlockSources(sources, std::move(block)); if (!wired.ok()) return wired.status();
  const auto validation = ValidatePreparedBlockSources(config, sources, *wired); if (!validation.ok()) return validation;
  return std::move(*wired);
}
Status ValidateSourcePhases(const FlashConfig& config, const StepPhasesLaunch& phases, const IndexedSourcesLaunch& x) {
  const auto phase = ValidateStepPhases(config, phases); if (!phase.ok()) return phase;
  const auto source = ValidateIndexedSources(config, x); if (!source.ok()) return source;
  const auto& a = x.sources.attention; const auto& q = phases.query.table;
  if (q.layer != a.layer || q.tokens != a.input.mix.tokens || q.stream != a.input.mix.stream ||
      phases.start != a.window.prepare.cache.start || !Same(q.error_flag, a.input.mix.error_flag) ||
      !Same(q.output, a.query.rope.phases) || !Same(q.output, a.window.prepare.rope.phases) ||
      (x.indexer && !Same(q.output, x.indexer->query.rope.phases)))
    return Status::InvalidArgument("Generated query phases differ from source step or consumers");
  const bool cache = x.sources.compressed && x.sources.compressed->cache;
  if (bool(phases.compressed) != cache) return Status::InvalidArgument("Generated compressed phases differ from emitted cache stage");
  if (cache && (!Same(phases.compressed->table.output, x.sources.compressed->cache->rope.phases) ||
      (x.indexer && x.indexer->key && !Same(phases.compressed->table.output, x.indexer->key->rope.phases))))
    return Status::InvalidArgument("Generated compressed phases differ from cache/index-key consumers");
  return Status::Ok();
}
Result<IndexedSourcesLaunch> WireSourcePhases(const FlashConfig& config, const StepPhasesLaunch& phases,
    IndexedSourcesLaunch x) {
  const auto phase = ValidateStepPhases(config, phases); if (!phase.ok()) return phase;
  x.sources.attention.query.rope.phases = phases.query.table.output;
  x.sources.attention.window.prepare.rope.phases = phases.query.table.output;
  if (x.indexer) x.indexer->query.rope.phases = phases.query.table.output;
  if (phases.compressed) {
    if (x.sources.compressed && x.sources.compressed->cache)
      x.sources.compressed->cache->rope.phases = phases.compressed->table.output;
    if (x.indexer && x.indexer->key) x.indexer->key->rope.phases = phases.compressed->table.output;
  }
  return x;
}
Result<IndexedSourcesLaunch> BindSourcePhases(const FlashConfig& config, const StepPhasesLaunch& phases,
    IndexedSourcesLaunch x) {
  auto wired = WireSourcePhases(config, phases, std::move(x)); if (!wired.ok()) return wired.status();
  const auto validation = ValidateSourcePhases(config, phases, *wired); if (!validation.ok()) return validation;
  return std::move(*wired);
}
}  // namespace pih::deepseek_v41
