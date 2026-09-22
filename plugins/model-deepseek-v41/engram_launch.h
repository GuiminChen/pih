#pragma once
#include "engram_weights.h"

namespace pih::deepseek_v41 {
struct EngramDeviceRegion final {
  std::uintptr_t address = 0;
  std::uint64_t bytes = 0;
};
// All regions are caller-owned, contiguous device allocations on the stream's
// device. Callers retain them through completion and initialize error_flag to 0.
// Validation checks address arithmetic/layout, not CUDA allocation provenance.
struct EngramLookupLaunch final {
  EngramDeviceRegion table, scales, ids, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0, layer = 0, world_size = 0, rank = 0;
};
struct EngramGateLaunch final {
  // BF16 input [tokens,4,5120], projected KV [tokens,25600], output like input.
  // q/k [4,5120] may be BF16 or F32. Mask is optional U8 [tokens], values 0/1.
  EngramDeviceRegion input, projected_kv, q_weight, k_weight, mask, output, error_flag;
  EngramStorage gate_storage = EngramStorage::kBF16;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0;
};
struct EngramProjectionLaunch final {
  // BF16 input [tokens,6144]; FP8 weight [25600,6144], E8M0 scale [800,192].
  // Caller-owned scratch: E4M3FN [tokens,6144], E8M0 [tokens,192].
  // BF16 output [tokens,25600], directly consumable by the gate launcher.
  EngramDeviceRegion input, weight, weight_scales, quantized, activation_scales, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0;
};
Status ValidateEngramProjection(const EngramProjectionLaunch& launch);
Status LaunchEngramProjection(const EngramProjectionLaunch& launch);
Status ValidateEngramLookup(const EngramLookupLaunch& launch);
Status ValidateEngramGate(const EngramGateLaunch& launch);
// Asynchronous launches; success is not completion. Read error_flag after the
// stream's completion: bit 1 invalid data/ID/mask, bit 2 non-finite arithmetic.
// Lookup produces a rank-local contribution; caller performs BF16 SUM before
// projection when world_size > 1. No implicit collective or CPU fallback.
Status LaunchEngramLookup(const EngramLookupLaunch& launch);
Status LaunchEngramGate(const EngramGateLaunch& launch);
struct EngramLaunch final {
  EngramLookupLaunch lookup;
  EngramProjectionLaunch projection;
  EngramGateLaunch gate;
};
// Validates the complete chain before enqueuing any stage. TP > 1 must use the
// separate launches with an admitted collective, not this single-rank entry.
Status ValidateEngramChain(const EngramLaunch& launch);
Status ValidateEngramSingleRank(const EngramLaunch& launch);
Status LaunchEngramSingleRank(const EngramLaunch& launch);
}  // namespace pih::deepseek_v41
