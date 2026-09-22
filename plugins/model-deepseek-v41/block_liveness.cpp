#include "block_bindings.h"
#include <limits>
#include <vector>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateBlockCrossStageBuffers(const FlashConfig& config, const IndexedSourcesLaunch& x,
    const PreparedBlockLaunch& block, std::span<const ExpertWeights> weights, EngramDeviceRegion workspace,
    const StepPhasesLaunch* phases, const EngramLaunch* engram, std::span<const EngramDeviceRegion> retained) {
  const auto binding = ValidatePreparedBlockSources(config, x, block); if (!binding.ok()) return binding;
  if (phases) { const auto phase = ValidateSourcePhases(config, *phases, x); if (!phase.ok()) return phase; }
  const auto& a = x.sources.attention; const auto& m = a.input.mix; const auto& n = a.input.input.norm;
  const auto& q = a.query; const auto& w = a.window;
  if (config.attention_sharing()[a.layer].has_engram != (engram != nullptr))
    return Status::InvalidArgument("Block Engram presence disagrees with configured layer");
  if (engram) {
    const auto valid = ValidateEngramChain(*engram); if (!valid.ok()) return valid;
    if (engram->lookup.layer != a.layer || engram->lookup.world_size != a.world_size ||
        engram->lookup.rank != block.ffn.dispatch.rank || engram->gate.tokens != m.tokens ||
        engram->gate.stream != m.stream || !Same(engram->gate.error_flag, m.error_flag) ||
        !Same(engram->gate.output, m.residual))
      return Status::InvalidArgument("Block Engram layer, rank or residual connection mismatch");
  }
  const auto accumulator = ExpertWorkspaceAccumulator(workspace, m.tokens); if (!accumulator.ok()) return accumulator.status();
  if (block.tail.experts.merge.routed.address || block.tail.experts.merge.routed.bytes)
    return Status::InvalidArgument("Block tail accumulator must remain unbound before workspace admission");
  auto tail = block.tail; tail.experts.merge.routed = *accumulator;
  const auto tail_valid = ValidateExpertResidual(tail); if (!tail_valid.ok()) return tail_valid;
  const auto& f = block.ffn; const auto& fm = f.input.mix; const auto& fn = f.input.input.norm;
  const auto& s = tail.experts.shared; const auto& r = tail.residual;
  if (!Same(s.gate.input, fn.output) || !Same(r.residual, fm.residual) || !Same(r.post, fm.post) ||
      !Same(r.comb, fm.comb) || !Same(r.error_flag, m.error_flag) || r.tokens != m.tokens || r.stream != m.stream ||
      weights.size() != 384U / a.world_size)
    return Status::InvalidArgument("Block FFN tail connection or expert count mismatch");
  std::vector<EngramDeviceRegion> source_reads{m.residual, m.fn, m.scale, m.base, a.input.input.collapse.pre, n.weight,
      q.low_rank.weight, q.low_rank.weight_scales, q.norm.weight, q.expand.weight, q.expand.weight_scales,
      w.projection.weight, w.projection.weight_scales, w.prepare.norm.weight};
  if (!phases) source_reads.push_back(q.rope.phases);
  std::vector<EngramDeviceRegion> source_writes{m.pre, m.post, m.comb, a.input.input.collapse.output, n.output,
      q.low_rank.quantized, q.low_rank.activation_scales, q.low_rank.output, q.norm.output,
      q.expand.quantized, q.expand.activation_scales, q.expand.output, q.rope.output,
      w.projection.quantized, w.projection.activation_scales, w.projection.output, w.prepare.norm.output,
      w.prepare.rope.output, w.prepare.cache.ring, m.error_flag};
  if (x.sources.compressed) {
    const auto& cp = *x.sources.compressed; const auto& p = cp.compressor.projection;
    source_reads.insert(source_reads.end(), {p.value_weight, p.gate_weight});
    source_writes.insert(source_writes.end(), {p.values, p.scores});
    if (cp.compressor.direct_norm) {
      source_reads.push_back(cp.compressor.direct_norm->weight); source_writes.push_back(cp.compressor.direct_norm->output);
    } else {
      const auto& pool = cp.compressor.pooled->pool;
      source_writes.insert(source_writes.end(), {pool.state_values, pool.state_scores, pool.output});
      if (cp.compressor.pooled->norm) {
        source_reads.push_back(cp.compressor.pooled->norm->weight); source_writes.push_back(cp.compressor.pooled->norm->output);
      }
    }
    if (cp.cache) {
      if (!phases) source_reads.push_back(cp.cache->rope.phases);
      source_writes.insert(source_writes.end(), {cp.cache->rope.output, cp.cache->cache.cache});
    }
  }
  if (x.indexer) {
    const auto& i = *x.indexer;
    source_reads.insert(source_reads.end(), {i.scoring.weights.weight, i.selection.candidates,
        i.query.projection.weight, i.query.projection.weight_scales});
    if (!phases) source_reads.push_back(i.query.rope.phases);
    source_writes.insert(source_writes.end(), {i.scoring.weights.output, i.scoring.score.output, i.selection.output,
        i.query.projection.quantized, i.query.projection.activation_scales, i.query.projection.output, i.query.quantize.values});
    if (i.candidates) source_writes.push_back(i.candidates->output);
    if (i.key) {
      const auto& k = *i.key;
      source_reads.insert(source_reads.end(), {k.projection.weight, k.norm.weight});
      if (!phases) source_reads.push_back(k.rope.phases);
      source_writes.insert(source_writes.end(), {k.projection.output, k.norm.output, k.cache.quantize.values, k.cache.cache});
    } else source_reads.push_back(i.scoring.score.key);
  }
  const auto& assembly = block.attention.attention.assembly; const auto& out = block.attention.attention.output;
  // Producer-consumer reads (Q/window/new prefix/indices/mHC) have already been
  // bound and are omitted here. Only external tail inputs must preexist sources.
  std::vector<EngramDeviceRegion> tail_reads{out.grouped.attention.sink, out.grouped.projection.weight,
      out.linear.weight, out.linear.weight_scales, fm.fn, fm.scale, fm.base, fn.weight,
      f.router.weight, f.router.bias, f.router.image_bias, f.router.image_mask};
  if (!x.indexer) tail_reads.push_back(assembly.selected);
  if (!x.sources.compressed || !x.sources.compressed->cache) tail_reads.push_back(assembly.compressed);
  for (const auto& expert : weights) for (const auto p : {expert.gate, expert.up, expert.down}) {
    for (const auto region : {p.weight, p.scales})
      if (!region.address || !region.bytes || region.bytes > std::numeric_limits<std::uintptr_t>::max() - region.address)
        return Status::InvalidArgument("Block expert weight region invalid");
    if (p.weight.bytes != 5120ULL * 2304 / 2 || p.scales.bytes != 5120ULL * 2304 / 32)
      return Status::InvalidArgument("Block expert weight extent invalid");
    tail_reads.insert(tail_reads.end(), {p.weight, p.scales});
  }
  std::vector<EngramDeviceRegion> tail_writes{assembly.kv, assembly.indices, out.grouped.attention.output,
      out.grouped.inverse_rope.output, out.grouped.projection.output, out.linear.quantized, out.linear.activation_scales,
      out.linear.output, out.reduction, block.attention.residual.output, fm.pre, fm.post, fm.comb,
      f.input.input.collapse.output, fn.output, f.router.logits, f.router.indices, f.router.route_weights,
      f.dispatch.counts, f.dispatch.slots, workspace, tail.experts.merge.output, r.output, m.error_flag};
  for (const auto* p : {&s.gate, &s.up, &s.down}) {
    tail_reads.insert(tail_reads.end(), {p->weight, p->weight_scales});
    tail_writes.insert(tail_writes.end(), {p->quantized, p->activation_scales, p->output});
  }
  tail_writes.push_back(s.activation.output);
  std::vector<EngramDeviceRegion> all_writes = source_writes;
  all_writes.insert(all_writes.end(), tail_writes.begin(), tail_writes.end());
  if (engram) all_writes.insert(all_writes.end(), {engram->lookup.output, engram->projection.quantized,
      engram->projection.activation_scales, engram->projection.output, engram->gate.output});
  if (phases) {
    all_writes.insert(all_writes.end(), {phases->query.table.positions, phases->query.table.output});
    if (phases->compressed) all_writes.insert(all_writes.end(),
        {phases->compressed->table.positions, phases->compressed->table.output});
  }
  for (const auto region : retained) {
    if (!region.address || !region.bytes || region.bytes > std::numeric_limits<std::uintptr_t>::max() - region.address)
      return Status::InvalidArgument("Retained sequence region is invalid");
    for (const auto write : all_writes) if (Overlap(write, region))
      return Status::InvalidArgument("Block overwrites another layer's retained sequence cache or publication");
  }
  if (engram) {
    const auto& e = *engram;
    const std::vector<EngramDeviceRegion> reads{e.lookup.table, e.lookup.scales, e.lookup.ids,
        e.projection.weight, e.projection.weight_scales, e.gate.input, e.gate.q_weight, e.gate.k_weight, e.gate.mask};
    const std::vector<EngramDeviceRegion> writes{e.lookup.output, e.projection.quantized,
        e.projection.activation_scales, e.projection.output, e.gate.output, e.lookup.error_flag};
    for (const auto write : writes) {
      // Only the first source read is the explicit gate-output consumer. Do not
      // exempt a weight merely because it aliases that consumer's address.
      for (std::size_t i = 1; i < source_reads.size(); ++i) if (Overlap(write, source_reads[i]))
        return Status::InvalidArgument("Engram overwrites a later source input or weight");
      for (const auto read : tail_reads) if (Overlap(write, read))
        return Status::InvalidArgument("Engram overwrites a later block input or weight");
      for (const auto* regions : {&source_writes, &tail_writes}) for (const auto other : *regions)
        if (Overlap(write, other) && !(Same(write, m.error_flag) && Same(other, m.error_flag)))
          return Status::InvalidArgument("Block overwrites retained Engram storage");
    }
    for (const auto* regions : {&source_writes, &tail_writes}) for (const auto write : *regions)
      for (const auto read : reads) if (Overlap(write, read))
        return Status::InvalidArgument("Block overwrites Engram input or weight");
    if (phases) {
      std::vector<EngramDeviceRegion> phase_writes{phases->query.table.positions, phases->query.table.output};
      if (phases->compressed) phase_writes.insert(phase_writes.end(),
          {phases->compressed->table.positions, phases->compressed->table.output});
      for (const auto write : phase_writes) for (const auto* regions : {&reads, &writes})
        for (const auto region : *regions) if (Overlap(write, region))
          return Status::InvalidArgument("Generated phases overwrite Engram storage");
    }
  }
  if (phases) {
    std::vector<EngramDeviceRegion> phase_writes{phases->query.table.positions, phases->query.table.output};
    if (phases->compressed) phase_writes.insert(phase_writes.end(),
        {phases->compressed->table.positions, phases->compressed->table.output});
    for (const auto write : phase_writes) {
      for (const auto* regions : {&source_reads, &source_writes, &tail_reads, &tail_writes})
        for (const auto region : *regions) if (Overlap(write, region))
          return Status::InvalidArgument("Generated RoPE positions/phases overlap block input, weight or writable storage");
    }
  }
  for (const auto write : source_writes) {
    for (const auto read : tail_reads) if (Overlap(write, read))
      return Status::InvalidArgument("Source/index stage overwrites later block input or weight");
    for (const auto other : tail_writes) if (Overlap(write, other) && !(Same(write, m.error_flag) && Same(other, m.error_flag)))
      return Status::InvalidArgument("Block execution overwrites retained source/index output");
  }
  for (const auto write : tail_writes) for (const auto read : source_reads) if (Overlap(write, read))
    return Status::InvalidArgument("Block execution overwrites source/index input or weight");
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
