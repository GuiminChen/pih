#pragma once
#include "attention_prepare.h"
#include "compressed_prepare.h"

namespace pih::deepseek_v41 {
struct AttentionSourcesLaunch final {
  AttentionPrepareLaunch attention;
  // Exactly the configured compressed-KV owner runs this producer. Consumers
  // retain their already admitted source cache and do not repeat compression.
  std::optional<CompressedPrepareLaunch> compressed;
};
Status ValidateAttentionSources(const FlashConfig& config, const AttentionSourcesLaunch& launch);
Status LaunchAttentionSources(const FlashConfig& config, const AttentionSourcesLaunch& launch);
}  // namespace pih::deepseek_v41
