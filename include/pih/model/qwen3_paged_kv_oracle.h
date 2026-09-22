#pragma once

#include <cstdint>
#include <span>

#include "pih/core/bfloat16.h"
#include "pih/core/status.h"
#include "pih/model/qwen3_kv_slot_pool.h"

namespace pih {

// Materializes one layer of committed paged KV into token-major [T,8,128]
// tensors. This is a bounded CPU correctness oracle, not a production path.
Status qwen_materialize_paged_kv_oracle(
    std::span<const BFloat16> physical_backing,
    std::span<const QwenKvSlotState> slot_states,
    std::span<const QwenKvBlockHandle> handles,
    std::uint32_t owner_sequence_index, std::uint32_t layer,
    std::uint32_t token_count, std::span<BFloat16> key_output,
    std::span<BFloat16> value_output);

}  // namespace pih
