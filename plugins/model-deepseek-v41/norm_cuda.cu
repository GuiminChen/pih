#include "norm_launch.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion region) { return reinterpret_cast<T*>(region.address); }
template<class T> __device__ float Weight(const T* w, unsigned column) { return float(w[column]); }
template<> __device__ float Weight(const __nv_bfloat16* w, unsigned column) { return __bfloat162float(w[column]); }
template<class T>
__global__ void Norm(const __nv_bfloat16* input, const T* weight, __nv_bfloat16* output,
    unsigned* error, unsigned width) {
  __shared__ float scratch[256];
  const auto offset = static_cast<unsigned long long>(blockIdx.x) * width;
  float square = 0;
  for (unsigned column = threadIdx.x; column < width; column += 256) {
    const float value = __bfloat162float(input[offset + column]);
    if (!isfinite(value)) atomicOr(error, 1U);
    square += value * value;
  }
  scratch[threadIdx.x] = square;
  __syncthreads();
  for (unsigned stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (!threadIdx.x) {
    const float inverse = rsqrtf(scratch[0] / float(width) + 1e-20F);
    if (!isfinite(scratch[0]) || !isfinite(inverse)) atomicOr(error, 2U);
    scratch[0] = inverse;
  }
  __syncthreads();
  for (unsigned column = threadIdx.x; column < width; column += 256) {
    const float weight_value = Weight(weight, column);
    if (!isfinite(weight_value)) atomicOr(error, 1U);
    const float normalized = __bfloat162float(input[offset + column]) * scratch[0];
    const float value = weight_value * normalized;
    const auto rounded = __float2bfloat16_rn(value);
    if (!isfinite(value) || !isfinite(__bfloat162float(rounded))) atomicOr(error, 2U);
    output[offset + column] = rounded;
  }
}
}
Status LaunchRmsNorm(const RmsNormLaunch& x) {
  const auto validation = ValidateRmsNorm(x);
  if (!validation.ok()) return validation;
  const auto stream = reinterpret_cast<cudaStream_t>(x.stream);
  if (x.weight_storage == EngramStorage::kBF16)
    Norm<<<x.rows, 256, 0, stream>>>(Ptr<const __nv_bfloat16>(x.input), Ptr<const __nv_bfloat16>(x.weight),
        Ptr<__nv_bfloat16>(x.output), Ptr<unsigned>(x.error_flag), x.width);
  else
    Norm<<<x.rows, 256, 0, stream>>>(Ptr<const __nv_bfloat16>(x.input), Ptr<const float>(x.weight),
        Ptr<__nv_bfloat16>(x.output), Ptr<unsigned>(x.error_flag), x.width);
  const auto result = cudaGetLastError();
  return result == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(result));
}
}  // namespace pih::deepseek_v41
