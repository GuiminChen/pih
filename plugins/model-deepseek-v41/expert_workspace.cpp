#include "expert_workspace.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
// Gathered input/weight, gate scratch/output, up scratch/output, activation,
// down scratch/output, and FP32 accumulator. Each segment starts at 256 bytes.
constexpr std::array<std::uint64_t, 13> kStride{10240, 4, 5120, 160, 4608,
    5120, 160, 4608, 4608, 2304, 72, 10240, 20480};
std::uint64_t Align(std::uint64_t x) { return (x + 255) & ~std::uint64_t(255); }
bool Valid(EngramDeviceRegion x, std::uint64_t bytes, unsigned alignment) {
  return x.address && x.address % alignment == 0 && x.bytes == bytes &&
      bytes <= std::numeric_limits<std::uintptr_t>::max() - x.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Result<std::uint64_t> ExpertWorkspaceBytes(std::uint32_t tokens) {
  if (!tokens || tokens > 4096) return Status::InvalidArgument("Expert workspace token capacity invalid");
  std::uint64_t bytes = 0;
  for (const auto stride : kStride) bytes += Align(tokens * stride);
  return bytes;
}
Result<EngramDeviceRegion> ExpertWorkspaceAccumulator(EngramDeviceRegion workspace, std::uint32_t tokens) {
  const auto bytes = ExpertWorkspaceBytes(tokens); if (!bytes.ok()) return bytes.status();
  if (!Valid(workspace, *bytes, 256)) return Status::InvalidArgument("Expert workspace extent or alignment invalid");
  std::uint64_t offset = 0;
  for (std::size_t i = 0; i < 12; ++i) offset += Align(tokens * kStride[i]);
  return EngramDeviceRegion{workspace.address + offset, tokens * kStride[12]};
}
Result<ExpertWorkspaceBatch> BuildExpertWorkspaceBatch(const ExpertDispatchLaunch& d,
    const ExpertCounts& counts, EngramDeviceRegion input, EngramDeviceRegion route_weights,
    std::span<const ExpertWeights> weights, EngramDeviceRegion workspace) {
  const auto dispatch = ValidateExpertDispatch(d); if (!dispatch.ok()) return dispatch;
  const auto observed = counts.ValidatePlan(d); if (!observed.ok()) return observed;
  const auto bytes = ExpertWorkspaceBytes(d.tokens); if (!bytes.ok()) return bytes.status();
  const unsigned local = (d.layer < 40 ? 384U : 128U) / d.world_size, picks = d.layer < 40 ? 6U : 3U;
  if (weights.size() != local || !Valid(workspace, *bytes, 256) ||
      !Valid(input, d.tokens * 10240ULL, 2) || !Valid(route_weights, d.tokens * std::uint64_t(picks) * 4, 4))
    return Status::InvalidArgument("Expert workspace storage, weights or source layout invalid");
  for (const auto read : {input, route_weights, d.indices, d.counts, d.slots, d.error_flag})
    if (Overlap(workspace, read)) return Status::InvalidArgument("Expert workspace overlaps source or dispatch storage");
  for (const auto& w : weights) {
    const std::array projections{w.gate, w.up, w.down};
    for (std::size_t i = 0; i < projections.size(); ++i) {
      const auto& p = projections[i];
      // All three matrices contain 5120*2304 elements. FP4 scales are per row,
      // per 32 input elements, so all three also have the same scale byte count.
      if (!Valid(p.weight, 5120ULL * 2304 / 2, 1) || !Valid(p.scales, 5120ULL * 2304 / 32, 1))
        return Status::InvalidArgument("Expert workspace FP4 weight layout invalid");
      if (Overlap(workspace, p.weight) || Overlap(workspace, p.scales))
        return Status::InvalidArgument("Expert workspace overlaps persistent expert weights");
    }
  }
  std::array<std::uintptr_t, 13> bases{};
  std::uint64_t offset = 0;
  for (std::size_t i = 0; i < bases.size(); ++i) {
    bases[i] = workspace.address + offset; offset += Align(d.tokens * kStride[i]);
  }
  ExpertWorkspaceBatch result;
  result.accumulator = {bases[12], d.tokens * kStride[12]};
  result.experts.reserve(local);
  for (unsigned i = 0; i < local; ++i) {
    const unsigned expert = d.rank * local + i;
    const auto count = counts.Rows(expert); if (!count.ok()) return count.status();
    const unsigned rows = *count;
    const auto region = [&](unsigned slot) -> EngramDeviceRegion {
      return rows ? EngramDeviceRegion{bases[slot], rows * kStride[slot]} : EngramDeviceRegion{};
    };
    ExpertTokenChainLaunch chain;
    chain.gather = {d, input, route_weights, region(0), region(1), expert, rows};
    chain.accumulator = result.accumulator;
    if (rows) {
      const auto& w = weights[i];
      RoutedExpertLaunch execution;
      execution.gate = {region(0), w.gate.weight, w.gate.scales, region(2), region(3), region(4), d.error_flag,
          d.stream, rows, 5120, 2304};
      execution.up = {region(0), w.up.weight, w.up.scales, region(5), region(6), region(7), d.error_flag,
          d.stream, rows, 5120, 2304};
      execution.activation = {region(4), region(7), region(1), region(8), d.error_flag, d.stream, rows};
      execution.down = {region(8), w.down.weight, w.down.scales, region(9), region(10), region(11), d.error_flag,
          d.stream, rows, 2304, 5120};
      chain.expert = execution;
    }
    result.experts.push_back(chain);
  }
  const auto validation = ValidateExpertBatch(result.experts, counts); if (!validation.ok()) return validation;
  return result;
}
Status ValidateExpertWorkspaceAllocation(const pih_cuda_allocation_v1& a,
    std::int32_t device_ordinal, std::uint32_t tokens) {
  const auto bytes = ExpertWorkspaceBytes(tokens); if (!bytes.ok()) return bytes.status();
  if (device_ordinal < 0 || a.struct_size != sizeof(a) || a.abi_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
      a.memory_kind != PIH_CUDA_MEMORY_DEVICE_V1 || a.device_ordinal != device_ordinal || !a.generation ||
      a.alignment < 256 || (a.alignment & (a.alignment - 1)) ||
      !Valid({a.address, a.bytes}, *bytes, 256) || a.address % a.alignment)
    return Status::FailedPrecondition("Expert workspace allocation identity, device or extent invalid");
  return Status::Ok();
}
Status AllocateExpertWorkspace(const pih_nvidia_cuda_memory_api_v1& memory,
    std::int32_t device_ordinal, std::uint32_t tokens, pih_cuda_allocation_v1& output) {
  const auto bytes = ExpertWorkspaceBytes(tokens); if (!bytes.ok()) return bytes.status();
  if (memory.struct_size != sizeof(memory) || memory.contract_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
      !memory.context || !memory.allocate_device || !memory.deallocate_device || device_ordinal < 0 ||
      output.struct_size != sizeof(output) || output.abi_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
      output.address || output.bytes || output.alignment || output.generation || output.memory_kind || output.device_ordinal)
    return Status::InvalidArgument("Expert workspace capability or empty output ledger invalid");
  // Preserve the provider's exact output for ownership reconciliation; never
  // replace it with an inferred device/address or silently discard it on error.
  const auto status = memory.allocate_device(memory.context, device_ordinal, *bytes, 256, &output);
  if (!pih_status_is_valid_v1(&status)) return Status::Internal("Expert workspace provider returned malformed status; quarantine ledger");
  if (!pih_status_is_ok_v1(&status)) {
    if (output.address || output.bytes || output.generation)
      return Status::Internal("Expert workspace allocation failed with ambiguous ownership; quarantine ledger");
    switch (status.code) {
      case PIH_STATUS_RESOURCE_EXHAUSTED_V1: return Status::ResourceExhausted("Expert workspace allocation exhausted backend memory");
      case PIH_STATUS_UNAVAILABLE_V1: return Status::Unavailable("Expert workspace memory provider unavailable");
      case PIH_STATUS_DEADLINE_EXCEEDED_V1: return Status::DeadlineExceeded("Expert workspace allocation deadline exceeded");
      default: return Status::FailedPrecondition("Expert workspace memory provider rejected allocation");
    }
  }
  return ValidateExpertWorkspaceAllocation(output, device_ordinal, tokens);
}
}  // namespace pih::deepseek_v41
