#pragma once
#include "mhc_launch.h"

namespace pih::deepseek_v41 {
struct HeadProjectionLaunch final {
  // Last normalized BF16 hidden [5120], promoted FP32 checkpoint weight
  // [129280/world_size,5120], rank-local FP32 logits [129280/world_size].
  EngramDeviceRegion input, weight, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t world_size = 0, rank = 0;
  // Packed projection gathers each sequence's last normalized token row. The
  // index table is U32 device storage in canonical PackedRequest order.
  EngramDeviceRegion row_indices;
  std::uint32_t sequences = 1, input_rows = 1;
};
struct ModelHeadLaunch final {
  MhcInputLaunch input;
  HeadProjectionLaunch projection;
};
Status ValidateHeadProjection(const HeadProjectionLaunch& launch);
Status LaunchHeadProjection(const HeadProjectionLaunch& launch);
Status ValidateModelHead(const ModelHeadLaunch& launch);
Status LaunchModelHead(const ModelHeadLaunch& launch);
struct PackedLogitsTransposeLaunch final {
  EngramDeviceRegion rank_major, sequence_major;
  std::uintptr_t stream{};
  std::uint32_t sequences{}, world_size{};
};
Status ValidatePackedLogitsTranspose(const PackedLogitsTransposeLaunch& launch);
Status LaunchPackedLogitsTranspose(const PackedLogitsTransposeLaunch& launch);
}  // namespace pih::deepseek_v41
