#pragma once
#include "engram_launch.h"

namespace pih::deepseek_v41 {
struct RouterLaunch final {
  // BF16 hidden [tokens,5120], promoted FP32 weight [experts,5120],
  // FP32 text bias [experts]; optional U8 image mask [tokens] requires
  // FP32 image bias [experts]. FP32 logits scratch [tokens,experts].
  // U32 indices and FP32 route_weights [tokens,picks].
  EngramDeviceRegion input, weight, bias, image_bias, image_mask, logits, indices, route_weights, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0, layer = 0;
};
Status ValidateRouter(const RouterLaunch& launch);
Status LaunchRouter(const RouterLaunch& launch);
}  // namespace pih::deepseek_v41
