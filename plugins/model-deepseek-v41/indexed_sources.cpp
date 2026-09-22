#include "indexed_sources.h"
#include <vector>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateIndexedSources(const FlashConfig& config, const IndexedSourcesLaunch& x) {
  const auto source = ValidateAttentionSources(config, x.sources); if (!source.ok()) return source;
  const auto& a = x.sources.attention; const auto& role = config.attention_sharing()[a.layer];
  const auto& m = a.input.mix; const auto& n = a.input.input.norm;
  const auto& q = a.query; const auto& w = a.window;
  const unsigned start = w.prepare.cache.start, tokens = m.tokens;
  const unsigned positions = role.compression_ratio ? (start + tokens) / role.compression_ratio : 0;
  if (bool(x.indexer) != (role.owns_index && positions != 0))
    return Status::InvalidArgument("Index producer presence differs from layer role or populated prefix");
  if (!x.indexer) return Status::Ok();
  const auto& i = *x.indexer;
  const auto indexer = ValidateIndexerLayer(config, a.layer, i); if (!indexer.ok()) return indexer;
  if (i.selection.start != start || i.selection.tokens != tokens || i.selection.positions != positions ||
      i.selection.stream != m.stream || !Same(i.selection.error_flag, m.error_flag) ||
      i.scoring.score.heads != 32U / a.world_size || !Same(i.query.projection.input, q.norm.output) ||
      !Same(i.scoring.weights.input, n.output) || !Same(i.query.rope.phases, q.rope.phases))
    return Status::InvalidArgument("Source/indexer step, query, hidden or phase connection mismatch");
  if (i.key) {
    if (!x.sources.compressed || !x.sources.compressed->cache)
      return Status::InvalidArgument("Index key production requires freshly normalized compressed latent");
    const auto& cp = *x.sources.compressed;
    const auto& norm = cp.compressor.direct_norm ? *cp.compressor.direct_norm : *cp.compressor.pooled->norm;
    if (!Same(i.key->projection.input, norm.output) || !Same(i.key->rope.phases, cp.cache->rope.phases))
      return Status::InvalidArgument("Compressed latent/index-key connection mismatch");
  }
  std::vector<EngramDeviceRegion> source_reads{m.residual, m.fn, m.scale, m.base, a.input.input.collapse.pre, n.weight,
      q.low_rank.weight, q.low_rank.weight_scales, q.norm.weight, q.expand.weight, q.expand.weight_scales,
      q.rope.phases, w.projection.weight, w.projection.weight_scales, w.prepare.norm.weight};
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
      source_reads.push_back(cp.cache->rope.phases);
      source_writes.insert(source_writes.end(), {cp.cache->rope.output, cp.cache->cache.cache});
    }
  }
  // Produced qr/hidden/latent are already connected above, so only external
  // indexer inputs participate in the source-write vs indexer-read check.
  std::vector<EngramDeviceRegion> index_reads{i.scoring.weights.weight, i.selection.candidates,
      i.query.projection.weight, i.query.projection.weight_scales, i.query.rope.phases};
  std::vector<EngramDeviceRegion> index_writes{i.scoring.weights.output, i.scoring.score.output, i.selection.output,
      i.query.projection.quantized, i.query.projection.activation_scales, i.query.projection.output, i.query.quantize.values};
  if (i.candidates) index_writes.push_back(i.candidates->output);
  if (i.key) {
    const auto& k = *i.key;
    index_reads.insert(index_reads.end(), {k.projection.weight, k.norm.weight, k.rope.phases});
    index_writes.insert(index_writes.end(), {k.projection.output, k.norm.output, k.cache.quantize.values, k.cache.cache});
  } else index_reads.push_back(i.scoring.score.key);
  for (const auto write : source_writes) {
    for (const auto read : index_reads) if (Overlap(write, read))
      return Status::InvalidArgument("Attention sources overwrite indexer external input");
    for (const auto other : index_writes) if (Overlap(write, other))
      return Status::InvalidArgument("Indexer overwrites retained attention source output");
  }
  for (const auto write : index_writes) for (const auto read : source_reads) if (Overlap(write, read))
    return Status::InvalidArgument("Indexer overwrites attention source input or weight");
  return Status::Ok();
}
IndexedSourcesOperation::IndexedSourcesOperation(IndexedSourcesOperation&& other) noexcept
    : indexer_(std::move(other.indexer_)), completion_(std::move(other.completion_)), deadline_(other.deadline_),
      failed_(other.failed_), complete_(other.complete_) { other.failed_ = true; }
Result<IndexedSourcesOperation> IndexedSourcesOperation::Start(const FlashConfig& config, const IndexedSourcesLaunch& x,
    std::uint32_t rank, std::uintptr_t communicator, const EngramCompletionResources& resources, Clock::time_point deadline) {
  const auto validation = ValidateIndexedSources(config, x); if (!validation.ok()) return validation;
  const auto& a = x.sources.attention;
  if (a.world_size == 1 || rank >= a.world_size) return Status::InvalidArgument("Indexed sources operation requires valid multi-rank geometry");
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Indexed sources deadline expired");
  const auto ready = ValidateEngramCompletionResources(resources); if (!ready.ok()) return ready;
  if (x.indexer) {
    const auto reduction = ValidateIndexerReduction(x.indexer->scoring.score, rank, communicator); if (!reduction.ok()) return reduction;
  }
  const auto sources = LaunchAttentionSources(config, x.sources); if (!sources.ok()) return sources;
  IndexedSourcesOperation operation; operation.deadline_ = deadline;
  if (x.indexer) {
    auto indexer = IndexerTensorParallel::Start(config, a.layer, *x.indexer, rank, communicator, resources, deadline);
    if (!indexer.ok()) return indexer.status();
    operation.indexer_.emplace(std::move(*indexer));
  } else {
    auto completion = EngramCompletion::RecordFlag(a.input.mix.error_flag, a.input.mix.stream, resources);
    if (!completion.ok()) return completion.status();
    operation.completion_.emplace(std::move(*completion));
  }
  operation.failed_ = false; return operation;
}
Result<bool> IndexedSourcesOperation::Poll() {
  if (failed_) return Status::FailedPrecondition("Indexed sources failed or moved from");
  if (complete_) return true;
  failed_ = true;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Indexed sources deadline expired during observation");
  if (indexer_) {
    const auto result = indexer_->Advance(); if (!result.ok()) return result.status();
    complete_ = *result == IndexerPipelineState::kComplete;
  } else {
    const auto result = completion_->Poll(); if (!result.ok()) return result.status();
    complete_ = *result;
  }
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Indexed sources deadline expired at observation");
  failed_ = false; return complete_;
}
}  // namespace pih::deepseek_v41
