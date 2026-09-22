#include "indexer_plan.h"
#include <algorithm>
#include <limits>

namespace pih::deepseek_v41 {
namespace {
enum Slot : unsigned { Quant, Scale, Query, QueryRope, Key, KeyNorm, KeyRope, Weights, Scores, Selected, Mask };
bool Absent(EngramDeviceRegion r) { return !r.address && !r.bytes; }
bool Valid(EngramDeviceRegion r) {
  return r.address && r.bytes && r.bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Result<IndexerPlan> IndexerPlan::Create(const FlashConfig& config, std::uint32_t tokens,
    std::uint32_t maximum_positions, std::uint32_t world, std::uint32_t rank, std::uint64_t budget) {
  if (config.config_sha256() == Sha256Digest{} || !tokens || tokens > 4096 ||
      maximum_positions < tokens || maximum_positions > 1048576 ||
      (world != 1 && world != 2 && world != 4 && world != 8) || rank >= world)
    return Status::InvalidArgument("Indexer plan configuration, capacity or rank invalid");
  IndexerPlan plan; plan.config_sha256_ = config.config_sha256();
  plan.tokens_ = tokens; plan.positions_ = maximum_positions; plan.world_ = world; plan.rank_ = rank;
  // Prefill starts at zero; every later step is one token. Avoid reserving
  // token_capacity * maximum_context for a combination that cannot execute.
  const auto cells = std::max(std::uint64_t(tokens) * tokens, std::uint64_t(maximum_positions));
  const std::array<std::uint64_t, 11> sizes{tokens * 1280ULL, tokens * 40ULL,
      tokens * (32ULL / world) * 256, tokens * (32ULL / world) * 256,
      tokens * 256ULL, tokens * 256ULL, tokens * 256ULL, tokens * (32ULL / world) * 2,
      cells * 2, tokens * std::min(512U, maximum_positions) * 4ULL, cells};
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    const auto aligned = (sizes[i] + 255) & ~std::uint64_t{255};
    if (aligned > budget - plan.bytes_) return Status::ResourceExhausted("Indexer arena exceeds budget");
    plan.segments_[i] = {plan.bytes_, sizes[i]}; plan.bytes_ += aligned;
  }
  return plan;
}
Result<IndexerPipelineLaunch> IndexerPlan::Bind(const BackboneWeightUpload& weights, const FlashConfig& config,
    EngramDeviceRegion arena, std::uint32_t layer, std::uint32_t start, std::uint32_t tokens,
    EngramDeviceRegion hidden, EngramDeviceRegion query_rank, EngramDeviceRegion query_phases,
    EngramDeviceRegion latent, EngramDeviceRegion compressed_phases, const LayerCacheRegions& cache,
    EngramDeviceRegion key_prefix, EngramDeviceRegion candidates, EngramDeviceRegion error,
    std::uintptr_t stream) const {
  if (config.config_sha256() != config_sha256_ || weights.catalog().config_sha256() != config_sha256_ ||
      weights.catalog().world_size() != world_ || weights.catalog().rank() != rank_ || layer >= 40 ||
      !tokens || tokens > tokens_ || (start && tokens != 1) || start >= positions_ || tokens > positions_ - start ||
      !stream || !Valid(arena) || arena.address % 256 || arena.bytes != bytes_)
    return Status::InvalidArgument("Indexer plan binding mismatch");
  const auto& role = config.attention_sharing()[layer];
  if (!role.owns_index || !role.compression_ratio)
    return Status::InvalidArgument("Indexer plan requires index owner");
  const auto positions = (start + tokens) / role.compression_ratio;
  const auto emitted = positions - start / role.compression_ratio;
  const bool key = role.owns_kv && emitted;
  if (!positions || (key ? (!Valid(latent) || !Valid(compressed_phases)) :
      (!Absent(latent) || !Absent(compressed_phases))) ||
      (role.index_uses_candidates ? !Valid(candidates) : !Absent(candidates)))
    return Status::InvalidArgument("Indexer plan has no scored positions or mismatched key/candidate inputs");
  for (const auto r : {hidden, query_rank, query_phases, latent, compressed_phases, cache.window,
      cache.compressed, cache.index_keys, cache.pool_values, cache.pool_scores, key_prefix, candidates, error})
    if ((!Absent(r) && !Valid(r)) || Overlap(arena, r))
      return Status::InvalidArgument("Indexer arena overlaps external input, cache or error");
  auto status = weights.ValidateScratch(std::array{arena, key ? cache.index_keys : EngramDeviceRegion{}, error});
  if (!status.ok()) return status;
  const auto r = [&](unsigned slot, std::uint64_t bytes) -> EngramDeviceRegion {
    return {arena.address + segments_[slot].offset, bytes};
  };
  const auto heads = 32U / world_;
  IndexerPipelineLaunch x;
  x.query.projection = {query_rank, {}, {}, r(Quant, tokens * 1280ULL), r(Scale, tokens * 40ULL),
      r(Query, tokens * std::uint64_t(heads) * 256), error, stream, tokens, 1280, heads * 128};
  x.query.rope = {x.query.projection.output, query_phases, r(QueryRope, tokens * std::uint64_t(heads) * 256),
      error, stream, tokens, heads, 128, false};
  x.query.quantize = {x.query.rope.output, error, stream, tokens, heads};
  if (key) {
    IndexerKeyLaunch k;
    k.projection = {latent, {}, r(Key, emitted * 256ULL), error, stream, emitted};
    k.norm = {k.projection.output, {}, r(KeyNorm, emitted * 256ULL), error, EngramStorage::kBF16, stream, emitted, 128};
    k.rope = {k.norm.output, compressed_phases, r(KeyRope, emitted * 256ULL), error, stream, emitted, 1, 128, false};
    k.cache = {{k.rope.output, error, stream, emitted, 1}, cache.index_keys, start / role.compression_ratio, cache.compressed_capacity};
    x.key = k;
  }
  x.scoring.weights = {hidden, {}, r(Weights, tokens * std::uint64_t(heads) * 2), error, stream, tokens, heads};
  x.scoring.score = {x.query.quantize.values, key_prefix, x.scoring.weights.output,
      r(Scores, tokens * std::uint64_t(positions) * 2), error, stream, tokens, heads, positions};
  x.selection = {x.scoring.score.output, r(Selected, tokens * std::uint64_t(std::min(512U, positions)) * 4),
      error, candidates, stream, start, tokens, role.compression_ratio, positions, start ? 128U : tokens};
  if (role.produces_candidates)
    x.candidates = IndexerCandidatesLaunch{x.scoring.score.output, r(Mask, tokens * std::uint64_t(positions)),
        error, stream, start, tokens, positions};
  return BindUploadedIndexer(weights, config, layer, x);
}
}  // namespace pih::deepseek_v41
