#pragma once
#include "attention_output.h"
#include "engram_reduce.h"

namespace pih::deepseek_v41 {
Status ValidateAttentionReduction(const AttentionLocalOutputLaunch& launch,
    std::uint32_t rank, std::uintptr_t communicator);
// Submit after wo_b and FP32 promotion. Poll until kEnqueued, then call
// RoundAttentionOutput before mHC on the same stream;
// completion/error admission and generation-wide abort remain caller-owned.
Result<EngramReduction> ReduceAttentionOutput(const AttentionLocalOutputLaunch& launch,
    std::uint32_t rank, std::uintptr_t communicator);
}  // namespace pih::deepseek_v41
