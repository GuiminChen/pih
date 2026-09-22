#include "block_sequence.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
Status Retain(EngramDeviceRegion& saved, EngramDeviceRegion value) {
  if (saved.bytes && !Same(saved, value))
    return Status::InvalidArgument("Sequence persistent cache allocation changed");
  saved = value; return Status::Ok();
}
Result<EngramDeviceRegion> Prefix(EngramDeviceRegion region, std::uint64_t bytes) {
  if (!bytes) return EngramDeviceRegion{};
  if (!region.address || bytes > region.bytes || region.bytes > std::numeric_limits<std::uintptr_t>::max() - region.address)
    return Status::FailedPrecondition("Sequence cache has no admitted populated prefix");
  return EngramDeviceRegion{region.address, bytes};
}
}
Status BlockSequence::Prepare(const FlashConfig& config, const StepPhasesLaunch& phases,
    IndexedSourcesLaunch& sources, PreparedBlockLaunch& block, std::uintptr_t communicator) {
  if (failed_ || active_) return Status::FailedPrecondition("Sequence failed or already has an active block");
  auto& a = sources.sources.attention; auto& m = a.input.mix;
  if (config.config_sha256() == Sha256Digest{} || config.config_sha256() != config_.config_sha256() ||
      a.layer != layer_ || phases.start != start_ || a.window.prepare.cache.start != start_ ||
      !m.tokens || m.tokens > 4096 || (start_ && m.tokens != 1) ||
      start_ >= FlashConfig::kMaximumPositions || m.tokens > FlashConfig::kMaximumPositions - start_ ||
      (layer_ && m.tokens != tokens_))
    return Status::InvalidArgument("Sequence config, layer order or contiguous step mismatch");
  if (stream_ && (m.stream != stream_ || communicator != communicator_ || !Same(m.error_flag, error_) ||
      a.world_size != world_ || block.ffn.dispatch.rank != rank_))
    return Status::InvalidArgument("Sequence stream, communicator, rank or error storage changed");
  if (!layer_ && (!input_ready_ || m.tokens != input_tokens_))
    return Status::FailedPrecondition("Layer zero requires completed token embedding for this step");
  a.input.mix.residual = residual_; a.input.input.collapse.residual = residual_;
  a.input.input.collapse.pre = pre_;
  pending_ = published_[layer_];
  auto retained = Retain(pending_.ring, a.window.prepare.cache.ring); if (!retained.ok()) return retained;
  const auto& role = config_.attention_sharing()[layer_];
  const auto positions = role.compression_ratio ? (start_ + m.tokens) / role.compression_ratio : 0;
  if (sources.sources.compressed) {
    const auto& cp = *sources.sources.compressed;
    if (cp.compressor.pooled) {
      retained = Retain(pending_.values, cp.compressor.pooled->pool.state_values); if (!retained.ok()) return retained;
      retained = Retain(pending_.scores, cp.compressor.pooled->pool.state_scores); if (!retained.ok()) return retained;
    }
    if (cp.cache) {
      retained = Retain(pending_.compressed, cp.cache->cache.cache); if (!retained.ok()) return retained;
    }
  }
  if (sources.indexer && sources.indexer->key) {
    retained = Retain(pending_.key, sources.indexer->key->cache.cache); if (!retained.ok()) return retained;
  }
  if (positions) {
    if (role.kv_source > layer_ || role.index_source > layer_)
      return Status::InvalidArgument("Sequence sharing source has not executed");
    const auto& kv = role.owns_kv ? pending_ : published_[role.kv_source];
    auto compressed = Prefix(kv.compressed, positions * 512ULL * 2); if (!compressed.ok()) return compressed.status();
    block.attention.attention.assembly.compressed = *compressed;
    if (sources.indexer) {
      auto key = Prefix(kv.key, positions * 128ULL * 2); if (!key.ok()) return key.status();
      sources.indexer->scoring.score.key = *key;
      if (role.index_uses_candidates) {
        const auto mask = published_[20].candidates;
        if (mask.bytes != std::uint64_t(m.tokens) * positions)
          return Status::FailedPrecondition("Sequence candidate source was not published for this step");
        sources.indexer->selection.candidates = mask;
      }
    } else {
      const auto selected = published_[role.index_source].selected;
      const auto columns = positions < 512 ? positions : 512;
      if (selected.bytes != std::uint64_t(m.tokens) * columns * 4)
        return Status::FailedPrecondition("Sequence index source was not published for this step");
      block.attention.attention.assembly.selected = selected;
    }
  } else {
    block.attention.attention.assembly.compressed = {};
    block.attention.attention.assembly.selected = {};
  }
  return Status::Ok();
}
Status BlockSequence::Reserve(const IndexedSourcesLaunch& sources, const PreparedBlockLaunch& block, std::uintptr_t communicator) {
  if (failed_ || active_) return Status::FailedPrecondition("Sequence cannot reserve another block");
  const auto& a = sources.sources.attention;
  // Prepare and all graph admission run synchronously before this reservation.
  pending_tokens_ = a.input.mix.tokens;
  pending_.selected = sources.indexer ? sources.indexer->selection.output : EngramDeviceRegion{};
  pending_.candidates = sources.indexer && sources.indexer->candidates ? sources.indexer->candidates->output : EngramDeviceRegion{};
  pending_residual_ = block.tail.residual.output; pending_pre_ = block.ffn.input.mix.pre;
  stream_ = a.input.mix.stream; error_ = a.input.mix.error_flag;
  communicator_ = communicator;
  world_ = a.world_size; rank_ = block.ffn.dispatch.rank;
  if (!layer_) input_ready_ = false;
  active_ = true; return Status::Ok();
}
std::vector<EngramDeviceRegion> BlockSequence::Retained(bool include_current, bool include_inputs) const {
  std::vector<EngramDeviceRegion> regions;
  if (weight_arena_.bytes) regions.push_back(weight_arena_);
  if (embedding_weight_.bytes) regions.push_back(embedding_weight_);
  if (include_inputs) {
    if (embedding_ids_.bytes) regions.push_back(embedding_ids_);
    for (const auto ids : engram_ids_) if (ids.bytes) regions.push_back(ids);
  }
  for (std::uint32_t i = 0; i < published_.size(); ++i) {
    if (!include_current && i == layer_) continue;  // Current-owner writes are admitted by its graph.
    const auto& p = published_[i];
    for (const auto r : {p.ring, p.values, p.scores, p.compressed, p.key, p.selected, p.candidates})
      if (r.bytes) regions.push_back(r);
  }
  return regions;
}
void BlockSequence::Complete() noexcept {
  if (!active_ || failed_) { Fail(); return; }
  published_[layer_] = pending_; residual_ = pending_residual_; pre_ = pending_pre_;
  tokens_ = pending_tokens_; active_ = false;
  if (++layer_ == FlashConfig::kMainLayers) {
    layer_ = 0; start_ += tokens_; completed_tokens_ = tokens_; tokens_ = 0;
    // Top-k indices/candidate masks are per-step, unlike persistent KV caches.
    for (auto& p : published_) { p.selected = {}; p.candidates = {}; }
  }
}
Result<SequenceStepOutput> BlockSequence::StepOutput() const {
  if (failed_ || active_ || input_ready_ || layer_ || !completed_tokens_)
    return Status::FailedPrecondition("Sequence step output requires all 40 layers to complete");
  return SequenceStepOutput{residual_, pre_, start_ - completed_tokens_, completed_tokens_};
}
}  // namespace pih::deepseek_v41
