#pragma once
#include "config.h"
#include "compressor_pool.h"
#include "compressed_kv.h"

namespace pih::deepseek_v41 {
struct CompressedPrepareLaunch final {
  CompressorLaunch compressor;
  // Absent exactly when this step emits no complete compressed row.
  std::optional<CompressedKvPrepareLaunch> cache;
  std::uint32_t layer = 0, start = 0;
};
Status ValidateCompressedPrepare(const FlashConfig& config, const CompressedPrepareLaunch& launch);
Status LaunchCompressedPrepare(const FlashConfig& config, const CompressedPrepareLaunch& launch);
}  // namespace pih::deepseek_v41
