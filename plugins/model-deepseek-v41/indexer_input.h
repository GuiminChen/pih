#pragma once
#include "linear_fp8.h"
#include "rope_launch.h"
#include "norm_launch.h"

namespace pih::deepseek_v41 {
struct IndexerQuantizeLaunch final {
  // Mutable BF16 [tokens,heads,128], group-32 E2M1/E8M0 simulated values.
  // heads=1 for index keys; 4/8/16/32 for rank-local index queries.
  EngramDeviceRegion values, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0, heads = 0;
};
Status ValidateIndexerQuantize(const IndexerQuantizeLaunch& launch);
Status LaunchIndexerQuantize(const IndexerQuantizeLaunch& launch);
struct IndexerQueryLaunch final {
  Fp8LinearLaunch projection;
  RopeApplyLaunch rope;
  IndexerQuantizeLaunch quantize;
};
Status ValidateIndexerQuery(const IndexerQueryLaunch& launch);
Status LaunchIndexerQuery(const IndexerQueryLaunch& launch);
struct IndexerKeyProjectionLaunch final {
  // BF16 unrotated compressor latent [rows,512], BF16 weight [128,512],
  // BF16 output [rows,128], FP32 accumulation without bias.
  EngramDeviceRegion input, weight, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t rows = 0;
};
Status ValidateIndexerKeyProjection(const IndexerKeyProjectionLaunch& launch);
Status LaunchIndexerKeyProjection(const IndexerKeyProjectionLaunch& launch);
struct IndexerKeyCacheLaunch final {
  IndexerQuantizeLaunch quantize;
  EngramDeviceRegion cache;
  std::uint32_t first_slot = 0, capacity = 0;
};
Status ValidateIndexerKeyCache(const IndexerKeyCacheLaunch& launch);
Status LaunchIndexerKeyCache(const IndexerKeyCacheLaunch& launch);
struct IndexerKeyLaunch final {
  IndexerKeyProjectionLaunch projection;
  RmsNormLaunch norm;
  RopeApplyLaunch rope;
  IndexerKeyCacheLaunch cache;
};
Status ValidateIndexerKey(const IndexerKeyLaunch& launch);
Status LaunchIndexerKey(const IndexerKeyLaunch& launch);
}  // namespace pih::deepseek_v41
