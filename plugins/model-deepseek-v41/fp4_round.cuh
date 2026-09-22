#pragma once
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
// E2M1 magnitude codes 0..7; midpoint ties choose the even encoded magnitude.
__device__ inline float RoundE2M1(float value) {
  const float magnitude = fabsf(value);
  const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  unsigned best = 0;
  float distance = magnitude;
  for (unsigned i = 1; i < 8; ++i) {
    const float candidate = fabsf(magnitude - levels[i]);
    if (candidate < distance || (candidate == distance && !(i & 1U))) {
      best = i; distance = candidate;
    }
  }
  return copysignf(levels[best], value);
}
}  // namespace pih::deepseek_v41
