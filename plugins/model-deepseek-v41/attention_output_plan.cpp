#include "attention_output_plan.h"
#include <algorithm>
#include <limits>

namespace pih::deepseek_v41 {
namespace {
enum Slot : unsigned { Kv, Indices, Attention, Inverse, Grouped, Quantized, Scales, Linear, Reduction, Residual };
bool Valid(EngramDeviceRegion r) {
  return r.address && r.bytes && r.bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Result<AttentionOutputPlan> AttentionOutputPlan::Create(const FlashConfig& config, std::uint32_t tokens,
    std::uint32_t maximum, std::uint32_t world, std::uint32_t rank, std::uint64_t budget) {
  if (config.config_sha256() == Sha256Digest{} || !tokens || tokens > 4096 || maximum < tokens || maximum > 1048576 ||
      (world != 1 && world != 2 && world != 4 && world != 8) || rank >= world)
    return Status::InvalidArgument("Attention output plan config, capacity or rank invalid");
  AttentionOutputPlan plan; plan.config_sha256_ = config.config_sha256();
  plan.tokens_ = tokens; plan.maximum_ = maximum; plan.world_ = world; plan.rank_ = rank;
  const std::uint64_t heads = 64 / world, groups = 8 / world;
  const std::array<std::uint64_t, 10> sizes{std::max(2ULL * tokens, 128ULL + maximum) * 1024,
      tokens * 640ULL * 4, tokens * heads * 1024, tokens * heads * 1024,
      tokens * groups * 2048, tokens * groups * 1024, tokens * groups * 32,
      tokens * 10240ULL, tokens * 20480ULL, tokens * 40960ULL};
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    const auto aligned = (sizes[i] + 255) & ~std::uint64_t{255};
    if (aligned > budget - plan.bytes_) return Status::ResourceExhausted("Attention output arena exceeds budget");
    plan.segments_[i] = {plan.bytes_, sizes[i]}; plan.bytes_ += aligned;
  }
  return plan;
}
Result<AttentionResidualLaunch> AttentionOutputPlan::Bind(const BackboneWeightUpload& weights, const FlashConfig& config,
    EngramDeviceRegion arena, const AttentionPrepareLaunch& source,
    EngramDeviceRegion compressed, EngramDeviceRegion selected) const {
  if (config.config_sha256() != config_sha256_ || weights.catalog().config_sha256() != config_sha256_ ||
      weights.catalog().world_size() != world_ || weights.catalog().rank() != rank_ || source.world_size != world_ ||
      source.layer >= 40 || !Valid(arena) || arena.address % 256 || arena.bytes != bytes_)
    return Status::InvalidArgument("Attention output plan binding configuration, rank or arena differs");
  const auto& m = source.input.mix;
  const auto source_valid = ValidateAttentionPrepare(config, source); if (!source_valid.ok()) return source_valid;
  const auto& window = source.window.prepare.cache;
  const auto tokens = window.tokens, start = window.start;
  if (!tokens || tokens > tokens_ || start >= maximum_ || tokens > maximum_ - start)
    return Status::InvalidArgument("Attention output step exceeds planned capacity");
  const auto ratio = config.attention_sharing()[source.layer].compression_ratio;
  const auto shape = GetAttentionAssemblyShape(start, tokens, ratio); if (!shape.ok()) return shape.status();
  const auto window_input = start ? window.ring : window.kv;
  for (const auto input : {window_input, compressed, selected, source.query.rope.output, source.query.rope.phases,
                          m.residual, m.post, m.comb, m.error_flag}) {
    if ((!input.address != !input.bytes) || (input.bytes && !Valid(input)) || Overlap(arena, input))
      return Status::InvalidArgument("Attention output arena overlaps or has malformed external inputs");
  }
  auto status = weights.ValidateScratch(std::array{arena, m.error_flag}); if (!status.ok()) return status;
  const std::uint32_t heads = 64 / world_, groups = 8 / world_;
  const auto rows = shape->window_rows + shape->compressed_rows;
  const auto picks = shape->window_columns + shape->compressed_columns;
  const std::array<std::uint64_t, 10> sizes{rows * 1024ULL, tokens * std::uint64_t(picks) * 4,
      tokens * std::uint64_t(heads) * 1024, tokens * std::uint64_t(heads) * 1024,
      tokens * std::uint64_t(groups) * 2048, tokens * std::uint64_t(groups) * 1024,
      tokens * std::uint64_t(groups) * 32, tokens * 10240ULL, tokens * 20480ULL, tokens * 40960ULL};
  const auto r = [&](unsigned i) -> EngramDeviceRegion { return {arena.address + segments_[i].offset, sizes[i]}; };
  AttentionResidualLaunch x;
  x.attention.assembly = {window_input, compressed, selected, r(Kv), r(Indices), m.error_flag, m.stream, start, tokens, ratio};
  auto& o = x.attention.output;
  o.grouped.attention = {source.query.rope.output, r(Kv), {}, r(Indices), r(Attention), m.error_flag, m.stream, tokens, heads, rows, picks};
  o.grouped.inverse_rope = {r(Attention), source.query.rope.phases, r(Inverse), m.error_flag, m.stream, tokens, heads, 512, true};
  o.grouped.projection = {r(Inverse), {}, r(Grouped), m.error_flag, m.stream, tokens, groups};
  o.linear = {r(Grouped), {}, {}, r(Quantized), r(Scales), r(Linear), m.error_flag, m.stream, tokens, groups * 1024, 5120};
  o.reduction = r(Reduction);
  x.residual = {r(Linear), m.residual, m.post, m.comb, r(Residual), m.error_flag, m.stream, tokens};
  auto bound = BindUploadedAttentionOutput(weights, config, source.layer, x.attention); if (!bound.ok()) return bound.status();
  x.attention = *bound;
  status = ValidateAttentionResidual(x); if (!status.ok()) return status;
  return x;
}
}  // namespace pih::deepseek_v41
