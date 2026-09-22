#include "engram_launch.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion region, std::uint64_t required, unsigned alignment) {
  return region.address && region.address % alignment == 0 && region.bytes == required &&
      required <= std::numeric_limits<std::uintptr_t>::max() - region.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
template<std::size_t N>
bool OutputsDisjoint(const std::array<EngramDeviceRegion, N>& inputs,
    EngramDeviceRegion output, EngramDeviceRegion error) {
  if (Overlap(output, error)) return false;
  for (const auto& input : inputs)
    if (Overlap(input, output) || Overlap(input, error)) return false;
  return true;
}
}
Status ValidateEngramLookup(const EngramLookupLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 || (x.layer != 1 && x.layer != 14) ||
      !x.world_size || x.world_size > 128 || 384 % x.world_size || 128 % x.world_size ||
      x.rank >= x.world_size)
    return Status::InvalidArgument("Engram lookup sequence or partition invalid");
  const std::uint64_t rows = ((x.layer == 1 ? 384006168ULL : 384016682ULL) + x.world_size - 1) / x.world_size;
  if (!Valid(x.table, rows * 256, 1) || !Valid(x.scales, rows * 8, 1) ||
      !Valid(x.ids, x.tokens * 24ULL * 4, 4) ||
      !Valid(x.output, x.tokens * 6144ULL * 2, 2) || !Valid(x.error_flag, 4, 4) ||
      !OutputsDisjoint(std::array{x.table, x.scales, x.ids}, x.output, x.error_flag))
    return Status::InvalidArgument("Engram lookup buffer layout or alias invalid");
  return Status::Ok();
}
Status ValidateEngramGate(const EngramGateLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 ||
      (x.gate_storage != EngramStorage::kBF16 && x.gate_storage != EngramStorage::kF32))
    return Status::InvalidArgument("Engram gate sequence or storage invalid");
  const unsigned gate_bytes = x.gate_storage == EngramStorage::kBF16 ? 2 : 4;
  if (!Valid(x.input, x.tokens * 20480ULL * 2, 2) ||
      !Valid(x.projected_kv, x.tokens * 25600ULL * 2, 2) ||
      !Valid(x.q_weight, 20480ULL * gate_bytes, gate_bytes) ||
      !Valid(x.k_weight, 20480ULL * gate_bytes, gate_bytes) ||
      !Valid(x.output, x.tokens * 20480ULL * 2, 2) || !Valid(x.error_flag, 4, 4) ||
      ((x.mask.address || x.mask.bytes) && !Valid(x.mask, x.tokens, 1)) ||
      !OutputsDisjoint(std::array{x.input, x.projected_kv, x.q_weight, x.k_weight, x.mask},
                      x.output, x.error_flag))
    return Status::InvalidArgument("Engram gate buffer layout or alias invalid");
  return Status::Ok();
}
Status ValidateEngramProjection(const EngramProjectionLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096)
    return Status::InvalidArgument("Engram projection sequence invalid");
  if (!Valid(x.input, x.tokens * 6144ULL * 2, 2) ||
      !Valid(x.weight, 25600ULL * 6144, 1) || !Valid(x.weight_scales, 800ULL * 192, 1) ||
      !Valid(x.quantized, x.tokens * 6144ULL, 1) ||
      !Valid(x.activation_scales, x.tokens * 192ULL, 1) ||
      !Valid(x.output, x.tokens * 25600ULL * 2, 2) || !Valid(x.error_flag, 4, 4))
    return Status::InvalidArgument("Engram projection buffer layout invalid");
  const std::array inputs{x.input, x.weight, x.weight_scales};
  const std::array outputs{x.quantized, x.activation_scales, x.output, x.error_flag};
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    for (const auto& input : inputs)
      if (Overlap(outputs[i], input)) return Status::InvalidArgument("Engram projection input/output alias");
    for (std::size_t j = 0; j < i; ++j)
      if (Overlap(outputs[i], outputs[j])) return Status::InvalidArgument("Engram projection output/scratch alias");
  }
  return Status::Ok();
}
Status ValidateEngramChain(const EngramLaunch& x) {
  const auto lookup = ValidateEngramLookup(x.lookup);
  if (!lookup.ok()) return lookup;
  const auto projection = ValidateEngramProjection(x.projection);
  if (!projection.ok()) return projection;
  const auto gate = ValidateEngramGate(x.gate);
  if (!gate.ok()) return gate;
  const auto same = [](EngramDeviceRegion a, EngramDeviceRegion b) {
    return a.address == b.address && a.bytes == b.bytes;
  };
  if (x.lookup.tokens != x.projection.tokens || x.lookup.tokens != x.gate.tokens ||
      x.lookup.stream != x.projection.stream || x.lookup.stream != x.gate.stream ||
      !same(x.lookup.error_flag, x.projection.error_flag) || !same(x.lookup.error_flag, x.gate.error_flag) ||
      !same(x.lookup.output, x.projection.input) || !same(x.projection.output, x.gate.projected_kv))
    return Status::InvalidArgument("Engram stage connections invalid");
  // Keep all external inputs alive throughout this initial conservative chain.
  // Per-stage validation alone would miss scratch overwriting a later gate input.
  const std::array inputs{x.lookup.table, x.lookup.scales, x.lookup.ids,
      x.projection.weight, x.projection.weight_scales, x.gate.input,
      x.gate.q_weight, x.gate.k_weight, x.gate.mask};
  const std::array writes{x.lookup.output, x.projection.quantized,
      x.projection.activation_scales, x.projection.output, x.gate.output, x.lookup.error_flag};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto& input : inputs)
      if (Overlap(writes[i], input)) return Status::InvalidArgument("Engram chain overwrites a live input");
    for (std::size_t j = 0; j < i; ++j)
      if (Overlap(writes[i], writes[j])) return Status::InvalidArgument("Engram chain scratch/output alias");
  }
  return Status::Ok();
}
Status ValidateEngramSingleRank(const EngramLaunch& x) {
  if (x.lookup.world_size != 1 || x.lookup.rank != 0)
    return Status::InvalidArgument("Engram single-rank entry requires TP=1");
  return ValidateEngramChain(x);
}
}  // namespace pih::deepseek_v41
