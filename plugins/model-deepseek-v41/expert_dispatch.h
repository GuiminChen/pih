#pragma once
#include "engram_launch.h"

namespace pih::deepseek_v41 {
struct ExpertDispatchLaunch final {
  // U32 router indices [tokens,picks]. Output counts [local_experts] and
  // flattened routing slots [local_experts,tokens], unused slots UINT32_MAX.
  // A routing slot is token*picks+pick, not just a token index.
  EngramDeviceRegion indices, counts, slots, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0, layer = 0, world_size = 0, rank = 0;
};
Status ValidateExpertDispatch(const ExpertDispatchLaunch& launch);
Status LaunchExpertDispatch(const ExpertDispatchLaunch& launch);
struct ExpertGatherLaunch final {
  ExpertDispatchLaunch dispatch;
  // Original BF16 [tokens,5120] and FP32 [tokens,picks]; gathered BF16
  // [rows,5120] and FP32 [rows]. rows must equal the device dispatch count.
  EngramDeviceRegion input, route_weights, output, gathered_weights;
  std::uint32_t expert = 0, rows = 0;
};
Status ValidateExpertGather(const ExpertGatherLaunch& launch);
Status LaunchExpertGather(const ExpertGatherLaunch& launch);
struct ExpertScatterLaunch final {
  ExpertDispatchLaunch dispatch;
  // Already routing-weighted BF16 expert output [rows,5120]. Accumulator is
  // FP32 [tokens,5120], initialized to zero by its owner before the first expert.
  // Submit each expert exactly once on dispatch.stream; no concurrent writers.
  EngramDeviceRegion input, accumulator;
  std::uint32_t expert = 0, rows = 0;
};
Status ValidateExpertScatter(const ExpertScatterLaunch& launch);
Status LaunchExpertScatter(const ExpertScatterLaunch& launch);
}  // namespace pih::deepseek_v41
