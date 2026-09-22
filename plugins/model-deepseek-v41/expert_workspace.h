#pragma once
#include "expert_batch.h"
#include "pih/contracts/nvidia_cuda_memory_v1.h"
#include <vector>

namespace pih::deepseek_v41 {
struct ExpertProjectionWeights final { EngramDeviceRegion weight, scales; };
struct ExpertWeights final { ExpertProjectionWeights gate, up, down; };
// Capacity is known before routing. One 256-byte-aligned allocation, reused
// sequentially across experts; includes the persistent FP32 token accumulator.
Result<std::uint64_t> ExpertWorkspaceBytes(std::uint32_t tokens);
Result<EngramDeviceRegion> ExpertWorkspaceAccumulator(EngramDeviceRegion workspace, std::uint32_t tokens);
// Uses the admitted backend memory capability. output must be an empty,
// ABI-initialized ledger slot. It retains provider output even on malformed
// success/failure: quarantine ambiguous ownership, never free guessed handles.
// Caller owns the returned allocation and releases through the same provider
// only after all work using it is safely retired.
Status AllocateExpertWorkspace(const pih_nvidia_cuda_memory_api_v1& memory,
    std::int32_t device_ordinal, std::uint32_t tokens, pih_cuda_allocation_v1& output);
Status ValidateExpertWorkspaceAllocation(const pih_cuda_allocation_v1& allocation,
    std::int32_t device_ordinal, std::uint32_t tokens);
struct ExpertWorkspaceBatch final {
  EngramDeviceRegion accumulator;
  std::vector<ExpertTokenChainLaunch> experts;
};
Result<ExpertWorkspaceBatch> BuildExpertWorkspaceBatch(const ExpertDispatchLaunch& dispatch,
    const ExpertCounts& counts, EngramDeviceRegion input, EngramDeviceRegion route_weights,
    std::span<const ExpertWeights> weights, EngramDeviceRegion workspace);
}  // namespace pih::deepseek_v41
