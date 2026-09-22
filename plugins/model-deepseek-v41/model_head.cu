#include "model_head.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__global__ void Project(const __nv_bfloat16* input, const float* weight, float* output, unsigned* error, unsigned rows) {
  const unsigned row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= rows) return;
  float sum = 0.0f;
  for (unsigned d = lane; d < 5120; d += 32) {
    const float a = __bfloat162float(input[d]), b = weight[static_cast<unsigned long long>(row) * 5120 + d];
    if (!isfinite(a) || !isfinite(b)) atomicOr(error, 1U);
    sum += a * b;
  }
  for (unsigned delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, delta);
  if (!lane) {
    if (!isfinite(sum)) { atomicOr(error, 2U); sum = 0.0f; }
    output[row] = sum;
  }
}
}
Status LaunchHeadProjection(const HeadProjectionLaunch& x) {
  const auto validation = ValidateHeadProjection(x); if (!validation.ok()) return validation;
  const unsigned rows = 129280 / x.world_size;
  Project<<<(rows + 3) / 4, 128, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(Ptr<const __nv_bfloat16>(x.input),
      Ptr<const float>(x.weight), Ptr<float>(x.output), Ptr<unsigned>(x.error_flag), rows);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
}  // namespace pih::deepseek_v41
