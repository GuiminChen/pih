#include "step_phases.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateStepPhases(const FlashConfig& config, const StepPhasesLaunch& x) {
  const auto query = ValidateRopeSequence(x.query); if (!query.ok()) return query;
  const auto& q = x.query;
  if (config.config_sha256() == Sha256Digest{} || q.table.layer >= FlashConfig::kMainLayers ||
      x.start != q.first_position || q.stride != 1 || (x.start && q.table.tokens != 1))
    return Status::InvalidArgument("Step phases require admitted backbone prefill/decode positions");
  const auto& role = config.attention_sharing()[q.table.layer];
  const unsigned rows = role.compression_ratio ? (x.start + q.table.tokens) / role.compression_ratio - x.start / role.compression_ratio : 0;
  const bool produces = role.owns_kv && role.compression_ratio && rows;
  if (bool(x.compressed) != produces) return Status::InvalidArgument("Compressed phases differ from owner-layer emitted rows");
  if (!x.compressed) return Status::Ok();
  const auto compressed = ValidateRopeSequence(*x.compressed); if (!compressed.ok()) return compressed;
  const auto& c = *x.compressed;
  if (c.table.layer != q.table.layer || c.table.tokens != rows || c.table.stream != q.table.stream ||
      !Same(c.table.error_flag, q.table.error_flag) || c.stride != role.compression_ratio ||
      c.first_position != (x.start / role.compression_ratio) * role.compression_ratio)
    return Status::InvalidArgument("Compressed phases must use each completed group's first original position");
  for (const auto left : {q.table.positions, q.table.output})
    for (const auto right : {c.table.positions, c.table.output})
      if (Overlap(left, right)) return Status::InvalidArgument("Query and compressed phase writable storage overlap");
  return Status::Ok();
}
Status LaunchStepPhases(const FlashConfig& config, const StepPhasesLaunch& x) {
  const auto validation = ValidateStepPhases(config, x); if (!validation.ok()) return validation;
  const auto query = LaunchRopeSequence(x.query); if (!query.ok()) return query;
  return x.compressed ? LaunchRopeSequence(*x.compressed) : Status::Ok();
}
Result<StepPhasesPlan> StepPhasesPlan::Create(const FlashConfig& config,
    std::uint32_t tokens, std::uint64_t budget) {
  if (config.config_sha256() == Sha256Digest{} || !tokens || tokens > 4096)
    return Status::InvalidArgument("Phase plan requires admitted config and bounded token capacity");
  StepPhasesPlan plan; plan.config_sha256_ = config.config_sha256(); plan.tokens_ = tokens;
  for (unsigned i = 0; i < 4; ++i) {
    const auto aligned = (tokens * (i % 2 ? 256ULL : 4ULL) + 255) & ~std::uint64_t{255};
    if (aligned > budget - plan.bytes_) return Status::ResourceExhausted("Phase arena exceeds budget");
    plan.offsets_[i] = plan.bytes_; plan.bytes_ += aligned;
  }
  return plan;
}
Result<StepPhasesLaunch> StepPhasesPlan::Bind(const FlashConfig& config, EngramDeviceRegion arena,
    std::uint32_t layer, std::uint32_t start, std::uint32_t tokens,
    EngramDeviceRegion error, std::uintptr_t stream) const {
  if (config.config_sha256() != config_sha256_ || layer >= FlashConfig::kMainLayers ||
      !tokens || tokens > tokens_ || (start && tokens != 1) || start >= FlashConfig::kMaximumPositions ||
      tokens > FlashConfig::kMaximumPositions - start || !stream || !arena.address || arena.address % 256 ||
      arena.bytes != bytes_ || arena.bytes > std::numeric_limits<std::uintptr_t>::max() - arena.address ||
      !error.address || error.address % 4 || error.bytes != 4 ||
      error.bytes > std::numeric_limits<std::uintptr_t>::max() - error.address || Overlap(arena, error))
    return Status::InvalidArgument("Phase plan binding mismatch or overlapping error flag");
  const auto region = [&](unsigned i, unsigned rows) -> EngramDeviceRegion {
    return {arena.address + offsets_[i], rows * (i % 2 ? 256ULL : 4ULL)};
  };
  StepPhasesLaunch x; x.start = start;
  x.query = {{region(0, tokens), region(1, tokens), error, stream, tokens, layer}, start, 1};
  const auto& role = config.attention_sharing()[layer];
  if (role.owns_kv && role.compression_ratio) {
    const auto ratio = role.compression_ratio;
    const auto rows = (start + tokens) / ratio - start / ratio;
    if (rows) x.compressed = RopeSequenceLaunch{
        {region(2, rows), region(3, rows), error, stream, rows, layer}, start / ratio * ratio, ratio};
  }
  const auto status = ValidateStepPhases(config, x); if (!status.ok()) return status;
  return x;
}
}  // namespace pih::deepseek_v41
