#include "attention_prepare_plan.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
enum Slot : unsigned { Pre, Post, Comb, Collapse, Norm, LowQuant, LowScale, LowOutput, QueryNorm,
  ExpandQuant, ExpandScale, ExpandOutput, QueryRope, WindowQuant, WindowScale, WindowOutput, WindowNorm, WindowRope };
constexpr std::array<std::uint64_t, 18> kStride{16, 16, 64, 10240, 10240, 5120, 160, 2560, 2560,
  1280, 40, 0, 0, 5120, 160, 1024, 1024, 1024};
bool Valid(EngramDeviceRegion r) {
  return r.address && r.bytes && r.bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Result<AttentionPreparePlan> AttentionPreparePlan::Create(const FlashConfig& config, std::uint32_t tokens,
    std::uint32_t world, std::uint32_t rank, std::uint64_t budget) {
  if (config.config_sha256() == Sha256Digest{} || !tokens || tokens > 4096 ||
      (world != 1 && world != 2 && world != 4 && world != 8) || rank >= world)
    return Status::InvalidArgument("Attention preparation plan config, capacity or rank invalid");
  AttentionPreparePlan plan; plan.config_sha256_ = config.config_sha256();
  plan.tokens_ = tokens; plan.world_ = world; plan.rank_ = rank;
  for (std::size_t i = 0; i < plan.segments_.size(); ++i) {
    const auto bytes = tokens * (kStride[i] ? kStride[i] : 64ULL / world * 512 * 2);
    const auto aligned = (bytes + 255) & ~std::uint64_t{255};
    if (aligned > budget - plan.bytes_) return Status::ResourceExhausted("Attention preparation arena exceeds budget");
    plan.segments_[i] = {plan.bytes_, bytes}; plan.bytes_ += aligned;
  }
  return plan;
}
Result<AttentionPrepareLaunch> AttentionPreparePlan::Bind(const BackboneWeightUpload& weights, const FlashConfig& config,
    EngramDeviceRegion arena, std::uint32_t layer, std::uint32_t start, std::uint32_t tokens,
    EngramDeviceRegion residual, EngramDeviceRegion pre, EngramDeviceRegion phases,
    EngramDeviceRegion ring, EngramDeviceRegion error, std::uintptr_t stream) const {
  if (config.config_sha256() != config_sha256_ || weights.catalog().config_sha256() != config_sha256_ ||
      weights.catalog().world_size() != world_ || weights.catalog().rank() != rank_ ||
      layer >= 40 || !tokens || tokens > tokens_ || !stream || !Valid(arena) || arena.address % 256 || arena.bytes != bytes_)
    return Status::InvalidArgument("Attention preparation plan binding mismatch");
  for (const auto external : {residual, pre, phases, ring, error})
    if (!Valid(external) || Overlap(arena, external))
      return Status::InvalidArgument("Attention preparation arena overlaps external input/cache/error");
  auto status = weights.ValidateScratch(std::array{arena, ring, error}); if (!status.ok()) return status;
  const auto r = [&](unsigned i) -> EngramDeviceRegion {
    return {arena.address + segments_[i].offset, tokens * (kStride[i] ? kStride[i] : 64ULL / world_ * 512 * 2)};
  };
  AttentionPrepareLaunch x;
  x.layer = layer; x.world_size = world_;
  x.input.mix = {residual, {}, {}, {}, r(Pre), r(Post), r(Comb), error, stream, tokens};
  x.input.input.collapse = {residual, pre, r(Collapse), error, stream, tokens};
  x.input.input.norm = {r(Collapse), {}, r(Norm), error, EngramStorage::kBF16, stream, tokens, 5120};
  x.query.low_rank = {r(Norm), {}, {}, r(LowQuant), r(LowScale), r(LowOutput), error, stream, tokens, 5120, 1280};
  x.query.norm = {r(LowOutput), {}, r(QueryNorm), error, EngramStorage::kBF16, stream, tokens, 1280};
  x.query.expand = {r(QueryNorm), {}, {}, r(ExpandQuant), r(ExpandScale), r(ExpandOutput), error,
      stream, tokens, 1280, 64U / world_ * 512};
  x.query.rope = {r(ExpandOutput), phases, r(QueryRope), error, stream, tokens, 64U / world_, 512, false};
  x.window.projection = {r(Norm), {}, {}, r(WindowQuant), r(WindowScale), r(WindowOutput), error, stream, tokens, 5120, 512};
  x.window.prepare.norm = {r(WindowOutput), {}, r(WindowNorm), error, EngramStorage::kBF16, stream, tokens, 512};
  x.window.prepare.rope = {r(WindowNorm), phases, r(WindowRope), error, stream, tokens, 1, 512, false};
  x.window.prepare.cache = {r(WindowRope), ring, error, stream, start, tokens};
  return BindUploadedAttentionPrepare(weights, config, x);
}
}  // namespace pih::deepseek_v41
