#pragma once
#include "attention_input.h"
#include "mhc_launch.h"
#include "config.h"

namespace pih::deepseek_v41 {
struct AttentionPrepareLaunch final {
  MhcSublayerInputLaunch input;
  AttentionQueryLaunch query;
  AttentionWindowLaunch window;
  std::uint32_t layer = 0, world_size = 0;
};
Status ValidateAttentionPrepare(const FlashConfig& config, const AttentionPrepareLaunch& launch);
// Enqueues mHC input, query, window projection/normalization/RoPE/cache update.
// Retains normalized hidden and qr for the compressor/indexer. Caller owns
// phase provenance, cache lifetime and completion admission.
Status LaunchAttentionPrepare(const FlashConfig& config, const AttentionPrepareLaunch& launch);
}  // namespace pih::deepseek_v41
