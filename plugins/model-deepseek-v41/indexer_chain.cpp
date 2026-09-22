#include "indexer_chain.h"
#include <utility>
#include <vector>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateIndexerPipeline(const IndexerPipelineLaunch& x) {
  const auto query = ValidateIndexerQuery(x.query); if (!query.ok()) return query;
  const auto scoring = ValidateIndexerWeightedScore(x.scoring); if (!scoring.ok()) return scoring;
  const auto selection = ValidateIndexerSelect(x.selection); if (!selection.ok()) return selection;
  const auto& s = x.scoring.score;
  const auto& t = x.selection;
  if (s.tokens != t.tokens || s.positions != t.positions || s.stream != t.stream ||
      !Same(s.output, t.scores) || !Same(s.error_flag, t.error_flag))
    return Status::InvalidArgument("Indexer score/selection connection invalid");
  const auto& q = x.query;
  if (q.quantize.tokens != s.tokens || q.quantize.heads != s.heads || q.quantize.stream != s.stream ||
      !Same(q.quantize.values, s.query) || !Same(q.quantize.error_flag, s.error_flag))
    return Status::InvalidArgument("Indexer query/scoring connection invalid");
  if (x.key) {
    const auto key = ValidateIndexerKey(*x.key); if (!key.ok()) return key;
    const auto& k = *x.key;
    const auto first = t.start / t.ratio, emitted = t.positions - first;
    if (!emitted || k.projection.rows != emitted || k.cache.first_slot != first ||
        k.cache.capacity < t.positions || k.projection.stream != s.stream ||
        !Same(k.projection.error_flag, s.error_flag) || s.key.address != k.cache.cache.address ||
        s.key.bytes > k.cache.cache.bytes)
      return Status::InvalidArgument("Indexer key-cache write/read step connection invalid");
  }
  if (x.candidates) {
    const auto candidate = ValidateIndexerCandidates(*x.candidates); if (!candidate.ok()) return candidate;
    const auto& c = *x.candidates;
    if (t.ratio != 1 || t.candidates.address || t.candidates.bytes || c.start != t.start || c.tokens != t.tokens ||
        c.positions != t.positions || c.stream != t.stream || !Same(c.scores, t.scores) || !Same(c.error_flag, t.error_flag))
      return Status::InvalidArgument("Indexer candidate-source connection invalid");
  }
  const auto& w = x.scoring.weights;
  std::vector<EngramDeviceRegion> reads{w.input, w.weight, t.candidates,
      q.projection.input, q.projection.weight, q.projection.weight_scales, q.rope.phases};
  std::vector<EngramDeviceRegion> writes{w.output, s.output, t.output, t.error_flag,
      x.candidates ? x.candidates->output : EngramDeviceRegion{}};
  writes.insert(writes.end(), {q.projection.quantized, q.projection.activation_scales, q.projection.output});
  if (!Same(q.projection.output, q.quantize.values)) writes.push_back(q.quantize.values);
  if (x.key) {
    const auto& k = *x.key;
    reads.insert(reads.end(), {k.projection.input, k.projection.weight, k.norm.weight, k.rope.phases});
    writes.insert(writes.end(), {k.projection.output, k.norm.output, k.cache.cache});
    if (!Same(k.norm.output, k.cache.quantize.values)) writes.push_back(k.cache.quantize.values);
  } else {
    reads.push_back(s.key);
  }
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads) if (Overlap(read, writes[i])) return Status::InvalidArgument("Indexer pipeline overwrites live input");
    for (std::size_t j = 0; j < i; ++j) if (Overlap(writes[i], writes[j])) return Status::InvalidArgument("Indexer pipeline writable alias");
  }
  return Status::Ok();
}

Status ValidateIndexerLayer(const FlashConfig& config, std::uint32_t layer,
    const IndexerPipelineLaunch& launch) {
  if (config.config_sha256() == Sha256Digest{} || layer >= FlashConfig::kMainLayers)
    return Status::InvalidArgument("Indexer requires admitted backbone configuration/layer");
  const auto& role = config.attention_sharing()[layer];
  if (!role.compression_ratio || !role.owns_index || role.index_source != layer)
    return Status::InvalidArgument("Layer does not own an index-selection computation");
  const auto validation = ValidateIndexerPipeline(launch); if (!validation.ok()) return validation;
  const auto& step = launch.selection;
  const unsigned emitted = step.positions - step.start / step.ratio;
  const bool needs_key = role.owns_kv && emitted != 0;
  const bool has_mask = step.candidates.address || step.candidates.bytes;
  if (step.ratio != role.compression_ratio || step.offset != (step.start ? 128U : step.tokens) ||
      bool(launch.key) != needs_key || bool(launch.candidates) != role.produces_candidates ||
      has_mask != role.index_uses_candidates)
    return Status::InvalidArgument("Indexer launch disagrees with frozen layer ownership/step");
  return Status::Ok();
}
IndexerSingleRank::IndexerSingleRank(IndexerSingleRank&& other) noexcept
    : completion_(std::move(other.completion_)), deadline_(other.deadline_),
      failed_(other.failed_), complete_(other.complete_) {
  other.failed_ = true;
}
Result<IndexerSingleRank> IndexerSingleRank::Start(const FlashConfig& config, std::uint32_t layer,
    const IndexerPipelineLaunch& launch,
    const EngramCompletionResources& resources, Clock::time_point deadline) {
  const auto validation = ValidateIndexerLayer(config, layer, launch); if (!validation.ok()) return validation;
  if (launch.scoring.score.heads != 32) return Status::InvalidArgument("Single-rank indexer requires all 32 heads");
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Indexer deadline expired before submission");
  const auto resource_status = ValidateEngramCompletionResources(resources); if (!resource_status.ok()) return resource_status;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Indexer deadline expired during preflight");
  if (launch.key) {
    const auto key = LaunchIndexerKey(*launch.key); if (!key.ok()) return key;
  }
  const auto query = LaunchIndexerQuery(launch.query); if (!query.ok()) return query;
  const auto score = LaunchIndexerWeightedScore(launch.scoring); if (!score.ok()) return score;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Indexer deadline expired before selection");
  if (launch.candidates) {
    const auto candidates = LaunchIndexerCandidates(*launch.candidates); if (!candidates.ok()) return candidates;
  }
  const auto selected = LaunchIndexerSelect(launch.selection); if (!selected.ok()) return selected;
  auto completion = EngramCompletion::RecordFlag(launch.selection.error_flag, launch.selection.stream, resources);
  if (!completion.ok()) return completion.status();
  IndexerSingleRank operation;
  operation.completion_.emplace(std::move(*completion)); operation.deadline_ = deadline; operation.failed_ = false;
  return operation;
}
Result<bool> IndexerSingleRank::Poll() {
  if (failed_ || !completion_) return Status::FailedPrecondition("Single-rank indexer failed or moved from");
  if (complete_) return true;
  failed_ = true;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Single-rank indexer deadline expired");
  const auto result = completion_->Poll(); if (!result.ok()) return result.status();
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Single-rank indexer deadline expired at observation");
  failed_ = false; complete_ = *result; return complete_;
}
}  // namespace pih::deepseek_v41
