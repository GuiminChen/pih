#include "engram_plan.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
constexpr std::array<std::uint64_t, 5> kStride{12288, 6144, 192, 51200, 40960};
bool Valid(EngramDeviceRegion r) {
  return r.address && r.bytes && r.bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Result<EngramPlan> EngramPlan::Create(const FlashConfig& config, std::uint32_t tokens,
    std::uint32_t world, std::uint32_t rank, std::uint64_t budget) {
  if (config.config_sha256() == Sha256Digest{} || !tokens || tokens > 4096 ||
      (world != 1 && world != 2 && world != 4 && world != 8) || rank >= world)
    return Status::InvalidArgument("Engram plan config, token capacity or rank invalid");
  EngramPlan plan; plan.config_sha256_ = config.config_sha256();
  plan.tokens_ = tokens; plan.world_ = world; plan.rank_ = rank;
  for (std::size_t i = 0; i < kStride.size(); ++i) {
    const auto bytes = tokens * kStride[i], aligned = (bytes + 255) & ~std::uint64_t{255};
    if (aligned > budget - plan.bytes_) return Status::ResourceExhausted("Engram arena exceeds memory budget");
    plan.segments_[i] = {plan.bytes_, bytes}; plan.bytes_ += aligned;
  }
  return plan;
}
Result<EngramLaunch> EngramPlan::Bind(const BackboneWeightUpload& weights, const FlashConfig& config,
    EngramDeviceRegion arena, std::uint32_t layer, std::uint32_t tokens,
    EngramDeviceRegion hashes, EngramDeviceRegion residual, EngramDeviceRegion error, std::uintptr_t stream) const {
  if (config.config_sha256() != config_sha256_ || weights.catalog().config_sha256() != config_sha256_ ||
      weights.catalog().world_size() != world_ || weights.catalog().rank() != rank_ ||
      (layer != 1 && layer != 14) || !tokens || tokens > tokens_ || !stream ||
      !Valid(arena) || arena.address % 256 || arena.bytes != bytes_)
    return Status::InvalidArgument("Engram plan binding config, step, layer or arena invalid");
  for (const auto input : {hashes, residual, error})
    if (!Valid(input) || Overlap(arena, input))
      return Status::InvalidArgument("Engram arena overlaps external hash/residual/error storage");
  auto status = weights.ValidateScratch(std::array{arena, error}); if (!status.ok()) return status;
  const auto r = [&](unsigned slot) -> EngramDeviceRegion {
    return {arena.address + segments_[slot].offset, tokens * kStride[slot]};
  };
  EngramLaunch x;
  x.lookup = {{}, {}, hashes, r(0), error, stream, tokens, layer, world_, rank_};
  x.projection = {r(0), {}, {}, r(1), r(2), r(3), error, stream, tokens};
  x.gate = {residual, r(3), {}, {}, {}, r(4), error, EngramStorage::kBF16, stream, tokens};
  return BindUploadedEngram(weights, x);
}
}  // namespace pih::deepseek_v41
