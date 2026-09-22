#include "ffn_plan.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
enum Slot : unsigned { Pre, Post, Comb, Collapse, Norm, Logits, Indices, Routes, Counts, Slots,
  GateQuant, GateScale, GateOut, UpQuant, UpScale, UpOut, Activation, DownQuant, DownScale, DownOut, Merge, Output };
constexpr std::array<std::uint64_t, 22> kStrides{
  16, 16, 64, 10240, 10240, 1536, 24, 24, 0, 0,
  5120, 160, 4608, 5120, 160, 4608, 4608, 2304, 72, 10240, 10240, 40960};
bool Valid(EngramDeviceRegion r) {
  return r.address && r.bytes && r.bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Result<FfnPlan> FfnPlan::Create(const FlashConfig& config, std::uint32_t tokens,
    std::uint32_t world, std::uint32_t rank, std::uint64_t budget) {
  if (config.config_sha256() == Sha256Digest{} || !tokens || tokens > 4096 ||
      (world != 1 && world != 2 && world != 4 && world != 8) || rank >= world)
    return Status::InvalidArgument("FFN plan configuration, token capacity or rank invalid");
  FfnPlan plan; plan.config_sha256_ = config.config_sha256(); plan.tokens_ = tokens;
  plan.world_ = world; plan.rank_ = rank;
  for (std::size_t i = 0; i < plan.segments_.size(); ++i) {
    const auto bytes = i == Counts ? 384ULL / world * 4 :
        i == Slots ? 384ULL / world * tokens * 4 : tokens * kStrides[i];
    const auto aligned = (bytes + 255) & ~std::uint64_t{255};
    if (aligned > budget - plan.bytes_) return Status::ResourceExhausted("FFN arena exceeds memory budget");
    plan.segments_[i] = {plan.bytes_, bytes}; plan.bytes_ += aligned;
  }
  return plan;
}
Result<FfnViews> FfnPlan::Bind(const BackboneWeightUpload& weights, const FlashConfig& config,
    EngramDeviceRegion arena, std::uint32_t layer, std::uint32_t tokens,
    EngramDeviceRegion residual, EngramDeviceRegion pre, EngramDeviceRegion accumulator,
    EngramDeviceRegion error, std::uintptr_t stream) const {
  if (config.config_sha256() != config_sha256_ || weights.catalog().config_sha256() != config_sha256_ ||
      weights.catalog().world_size() != world_ || weights.catalog().rank() != rank_ ||
      layer >= 40 || !tokens || tokens > tokens_ || !stream || !Valid(arena) || arena.address % 256 || arena.bytes != bytes_)
    return Status::InvalidArgument("FFN plan binding config, rank, arena or step invalid");
  for (const auto external : {residual, pre, accumulator, error})
    if (!Valid(external) || Overlap(arena, external))
      return Status::InvalidArgument("FFN arena overlaps external state, accumulator or error storage");
  auto status = weights.ValidateScratch(std::array{arena, accumulator, error}); if (!status.ok()) return status;
  const auto r = [&](unsigned slot) -> EngramDeviceRegion {
    const auto bytes = slot == Counts ? 384ULL / world_ * 4 :
        slot == Slots ? 384ULL / world_ * tokens * 4 : tokens * kStrides[slot];
    return {arena.address + segments_[slot].offset, bytes};
  };
  FfnViews x;
  x.route.input.mix = {residual, {}, {}, {}, r(Pre), r(Post), r(Comb), error, stream, tokens};
  x.route.input.input.collapse = {residual, pre, r(Collapse), error, stream, tokens};
  x.route.input.input.norm = {r(Collapse), {}, r(Norm), error, EngramStorage::kBF16, stream, tokens, 5120};
  x.route.router = {r(Norm), {}, {}, {}, {}, r(Logits), r(Indices), r(Routes), error, stream, tokens, layer};
  x.route.dispatch = {r(Indices), r(Counts), r(Slots), error, stream, tokens, layer, world_, rank_};
  auto& shared = x.tail.experts.shared;
  shared.gate = {r(Norm), {}, {}, r(GateQuant), r(GateScale), r(GateOut), error, stream, tokens, 5120, 2304};
  shared.up = {r(Norm), {}, {}, r(UpQuant), r(UpScale), r(UpOut), error, stream, tokens, 5120, 2304};
  shared.activation = {r(GateOut), r(UpOut), {}, r(Activation), error, stream, tokens};
  shared.down = {r(Activation), {}, {}, r(DownQuant), r(DownScale), r(DownOut), error, stream, tokens, 2304, 5120};
  x.tail.experts.merge = {accumulator, r(DownOut), r(Merge), error, stream, tokens};
  x.tail.residual = {r(Merge), residual, r(Post), r(Comb), r(Output), error, stream, tokens};
  auto mhc = BindUploadedMhc(weights, layer, UploadedSublayer::kFfn, x.route.input); if (!mhc.ok()) return mhc.status();
  x.route.input = *mhc;
  auto router = BindUploadedRouter(weights, x.route.router); if (!router.ok()) return router.status();
  x.route.router = *router;
  auto expert = BindUploadedSharedExpert(weights, layer, shared); if (!expert.ok()) return expert.status();
  shared = *expert;
  status = ValidateFfnRoute(config, x.route); if (!status.ok()) return status;
  status = ValidateExpertResidual(x.tail); if (!status.ok()) return status;
  return x;
}
}  // namespace pih::deepseek_v41
