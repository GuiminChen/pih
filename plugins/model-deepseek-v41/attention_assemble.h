#pragma once
#include "sparse_attention.h"
#include "attention_output.h"

namespace pih::deepseek_v41 {
struct AttentionAssemblyLaunch final {
  // BF16 window [tokens,512] at prefill or ring [128,512] at decode;
  // BF16 compressed prefix [end/ratio,512], already-offset I32 selected rows.
  // Compressed and selected regions are absent for ratio 0 or an empty prefix.
  EngramDeviceRegion window, compressed, selected, kv, indices, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t start = 0, tokens = 0, ratio = 0;
};
struct AttentionAssemblyShape final {
  std::uint32_t window_rows, compressed_rows, window_columns, compressed_columns;
};
Result<AttentionAssemblyShape> GetAttentionAssemblyShape(std::uint32_t start,
    std::uint32_t tokens, std::uint32_t ratio);
Status ValidateAttentionAssembly(const AttentionAssemblyLaunch& launch);
Status LaunchAttentionAssembly(const AttentionAssemblyLaunch& launch);
struct AssembledAttentionLaunch final {
  AttentionAssemblyLaunch assembly;
  SparseAttentionLaunch attention;
};
Status ValidateAssembledAttention(const AssembledAttentionLaunch& launch);
Status LaunchAssembledAttention(const AssembledAttentionLaunch& launch);
struct AssembledAttentionOutputLaunch final {
  AttentionAssemblyLaunch assembly;
  AttentionLocalOutputLaunch output;
};
// Rank-local [tokens,5120]; TP>1 still requires SUM before mHC expansion.
Status ValidateAssembledAttentionOutput(const AssembledAttentionOutputLaunch& launch);
Status LaunchAssembledAttentionOutput(const AssembledAttentionOutputLaunch& launch);
}  // namespace pih::deepseek_v41
