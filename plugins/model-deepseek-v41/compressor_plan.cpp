#include "compressor_plan.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
enum Slot : unsigned { Values, Scores, Pooled, Normalized, Rotated };
constexpr std::array<std::uint64_t, 5> kStride{2048, 2048, 1024, 1024, 1024};
bool Valid(EngramDeviceRegion r) {
  return r.address && r.bytes && r.bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
}
bool Absent(EngramDeviceRegion r) { return !r.address && !r.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Result<CompressorPlan> CompressorPlan::Create(const FlashConfig& config,
    std::uint32_t tokens, std::uint64_t budget) {
  if (config.config_sha256() == Sha256Digest{} || !tokens || tokens > 4096)
    return Status::InvalidArgument("Compressor plan requires admitted config and token capacity");
  CompressorPlan plan; plan.config_sha256_ = config.config_sha256(); plan.tokens_ = tokens;
  for (std::size_t i = 0; i < plan.segments_.size(); ++i) {
    const auto bytes = tokens * kStride[i];
    const auto aligned = (bytes + 255) & ~std::uint64_t{255};
    if (aligned > budget - plan.bytes_)
      return Status::ResourceExhausted("Compressor transient arena exceeds budget");
    plan.segments_[i] = {plan.bytes_, bytes}; plan.bytes_ += aligned;
  }
  return plan;
}
Result<CompressedPrepareLaunch> CompressorPlan::Bind(const BackboneWeightUpload& weights,
    const FlashConfig& config, EngramDeviceRegion arena, std::uint32_t layer,
    std::uint32_t start, std::uint32_t tokens, EngramDeviceRegion hidden,
    EngramDeviceRegion phases, const LayerCacheRegions& cache,
    EngramDeviceRegion error, std::uintptr_t stream) const {
  if (config.config_sha256() != config_sha256_ || weights.catalog().config_sha256() != config_sha256_ ||
      layer >= 40 || !tokens || tokens > tokens_ || !stream ||
      !Valid(arena) || arena.address % 256 || arena.bytes != bytes_)
    return Status::InvalidArgument("Compressor plan binding mismatch");
  const auto& role = config.attention_sharing()[layer];
  if (!role.owns_kv || (role.compression_ratio != 1 && role.compression_ratio != 2))
    return Status::InvalidArgument("Compressor plan requires a compressed KV owner");
  const auto step = CompressorOutputRows(start, tokens); if (!step.ok()) return step.status();
  const auto ratio = role.compression_ratio;
  const auto rows = ratio == 1 ? tokens : *step;
  if ((start + tokens) / ratio > cache.compressed_capacity ||
      (rows ? !Valid(phases) : !Absent(phases)) ||
      (ratio == 1 && (!Absent(cache.pool_values) || !Absent(cache.pool_scores))))
    return Status::InvalidArgument("Compressor cache capacity, phases or pooling state mismatch");
  for (const auto external : {hidden, phases, cache.window, cache.compressed, cache.index_keys,
      cache.pool_values, cache.pool_scores, error}) {
    if ((!Absent(external) && !Valid(external)) || Overlap(arena, external))
      return Status::InvalidArgument("Compressor arena overlaps external input, phase, cache or error");
  }
  auto status = weights.ValidateScratch(std::array{arena, cache.compressed, cache.pool_values, cache.pool_scores, error});
  if (!status.ok()) return status;
  const auto r = [&](unsigned slot, std::uint64_t bytes) -> EngramDeviceRegion {
    return bytes ? EngramDeviceRegion{arena.address + segments_[slot].offset, bytes} : EngramDeviceRegion{};
  };
  CompressedPrepareLaunch x; x.layer = layer; x.start = start;
  auto& c = x.compressor;
  c.projection = {hidden, {}, {}, r(Values, tokens * 512ULL * (ratio == 1 ? 2 : 4)),
      r(Scores, ratio == 2 ? tokens * 2048ULL : 0), error, stream, tokens, ratio};
  if (ratio == 1) {
    c.direct_norm = RmsNormLaunch{c.projection.values, {}, r(Normalized, rows * 1024ULL),
        error, EngramStorage::kBF16, stream, rows, 512};
  } else {
    CompressorNormalizeLaunch pooled;
    pooled.pool = {c.projection.values, c.projection.scores, cache.pool_values, cache.pool_scores,
        r(Pooled, rows * 1024ULL), error, stream, start, tokens};
    if (rows) pooled.norm = RmsNormLaunch{pooled.pool.output, {}, r(Normalized, rows * 1024ULL),
        error, EngramStorage::kBF16, stream, rows, 512};
    c.pooled = pooled;
  }
  if (rows) {
    CompressedKvPrepareLaunch output;
    output.rope = {r(Normalized, rows * 1024ULL), phases, r(Rotated, rows * 1024ULL),
        error, stream, rows, 1, 512, false};
    output.cache = {output.rope.output, cache.compressed, error, stream, rows, start / ratio, cache.compressed_capacity};
    x.cache = output;
  }
  return BindUploadedCompressed(weights, config, x);
}
}  // namespace pih::deepseek_v41
