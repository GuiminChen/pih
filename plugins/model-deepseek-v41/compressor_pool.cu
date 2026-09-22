#include "compressor_pool.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__device__ float Read(float value) { return value; }
__device__ float Read(__nv_bfloat16 value) { return __bfloat162float(value); }
template<class T> __device__ T Round(float value);
template<> __device__ float Round<float>(float value) { return value; }
template<> __device__ __nv_bfloat16 Round<__nv_bfloat16>(float value) { return __float2bfloat16_rn(value); }
template<class T>
__global__ void Projection(const __nv_bfloat16* input, const T* value_weight, const T* gate_weight,
    T* values, T* scores, unsigned* error, unsigned tokens) {
  __shared__ float a[16][32], b[16][32];
  const unsigned row = blockIdx.y * 16 + threadIdx.y;
  const unsigned column = blockIdx.x * 16 + threadIdx.x;
  const unsigned lane = threadIdx.y * 16 + threadIdx.x;
  const T* weight = blockIdx.z ? gate_weight : value_weight;
  T* output = blockIdx.z ? scores : values;
  float sum = 0.0f;
  for (unsigned base = 0; base < 5120; base += 32) {
    for (unsigned i = lane; i < 512; i += 256) {
      const unsigned r = i / 32, k = i % 32;
      const unsigned source_row = blockIdx.y * 16 + r;
      a[r][k] = source_row < tokens ? __bfloat162float(input[static_cast<unsigned long long>(source_row) * 5120 + base + k]) : 0.0f;
      b[r][k] = Read(weight[static_cast<unsigned long long>(blockIdx.x * 16 + r) * 5120 + base + k]);
      if (!isfinite(a[r][k]) || !isfinite(b[r][k])) atomicOr(error, 1U);
    }
    __syncthreads();
    for (unsigned k = 0; k < 32; ++k) sum += a[threadIdx.y][k] * b[threadIdx.x][k];
    __syncthreads();
  }
  if (row < tokens) {
    const T rounded = Round<T>(sum);
    if (!isfinite(sum) || !isfinite(Read(rounded))) atomicOr(error, 2U);
    output[static_cast<unsigned long long>(row) * 512 + column] = rounded;
  }
}
__device__ __nv_bfloat16 Pool(float v0, float v1, float s0, float s1, unsigned* error) {
  if (!isfinite(v0) || !isfinite(v1) || !isfinite(s0) || !isfinite(s1)) atomicOr(error, 1U);
  const float maximum = fmaxf(s0, s1);
  const float e0 = expf(s0 - maximum), e1 = expf(s1 - maximum);
  const float denominator = e0 + e1;
  const float value = v0 * (e0 / denominator) + v1 * (e1 / denominator);
  const auto rounded = __float2bfloat16_rn(value);
  if (!isfinite(value) || !isfinite(__bfloat162float(rounded))) atomicOr(error, 2U);
  return rounded;
}
__global__ void Prefill(const float* values, const float* scores, float* state_values,
    float* state_scores, __nv_bfloat16* output, unsigned* error, unsigned tokens) {
  const unsigned first = blockIdx.x * 2;
  for (unsigned d = threadIdx.x; d < 512; d += 256) {
    const auto offset = static_cast<unsigned long long>(first) * 512 + d;
    const float v0 = values[offset], s0 = scores[offset];
    if (first + 1 < tokens) {
      output[static_cast<unsigned long long>(blockIdx.x) * 512 + d] =
          Pool(v0, values[offset + 512], s0, scores[offset + 512], error);
    } else {
      if (!isfinite(v0) || !isfinite(s0)) atomicOr(error, 1U);
      state_values[d] = v0; state_scores[d] = s0;
    }
  }
}
__global__ void Decode(const float* values, const float* scores, float* state_values,
    float* state_scores, __nv_bfloat16* output, unsigned* error, unsigned slot) {
  for (unsigned d = threadIdx.x; d < 512; d += 256) {
    const float value = values[d], score = scores[d];
    if (!isfinite(value) || !isfinite(score)) atomicOr(error, 1U);
    state_values[slot * 512 + d] = value; state_scores[slot * 512 + d] = score;
    // Each lane owns its dimensions. Slot zero came from the preceding token;
    // slot one is this token. No cross-lane state read or barrier is necessary.
    if (slot == 1) output[d] = Pool(state_values[d], value, state_scores[d], score, error);
  }
}
}
Status LaunchCompressorPool(const CompressorPoolLaunch& x) {
  const auto validation = ValidateCompressorPool(x); if (!validation.ok()) return validation;
  const auto stream = reinterpret_cast<cudaStream_t>(x.stream);
  if (!x.start)
    Prefill<<<(x.tokens + 1) / 2, 256, 0, stream>>>(Ptr<const float>(x.values), Ptr<const float>(x.scores),
        Ptr<float>(x.state_values), Ptr<float>(x.state_scores), Ptr<__nv_bfloat16>(x.output), Ptr<unsigned>(x.error_flag), x.tokens);
  else
    Decode<<<1, 256, 0, stream>>>(Ptr<const float>(x.values), Ptr<const float>(x.scores),
        Ptr<float>(x.state_values), Ptr<float>(x.state_scores), Ptr<__nv_bfloat16>(x.output), Ptr<unsigned>(x.error_flag), x.start % 2);
  const auto result = cudaGetLastError();
  return result == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(result));
}
Status LaunchCompressorProjection(const CompressorProjectionLaunch& x) {
  const auto validation = ValidateCompressorProjection(x); if (!validation.ok()) return validation;
  const dim3 block(16, 16), grid(32, (x.tokens + 15) / 16, x.ratio);
  const auto stream = reinterpret_cast<cudaStream_t>(x.stream);
  if (x.ratio == 1)
    Projection<<<grid, block, 0, stream>>>(Ptr<const __nv_bfloat16>(x.input), Ptr<const __nv_bfloat16>(x.value_weight),
        Ptr<const __nv_bfloat16>(x.gate_weight), Ptr<__nv_bfloat16>(x.values), Ptr<__nv_bfloat16>(x.scores), Ptr<unsigned>(x.error_flag), x.tokens);
  else
    Projection<<<grid, block, 0, stream>>>(Ptr<const __nv_bfloat16>(x.input), Ptr<const float>(x.value_weight),
        Ptr<const float>(x.gate_weight), Ptr<float>(x.values), Ptr<float>(x.scores), Ptr<unsigned>(x.error_flag), x.tokens);
  const auto result = cudaGetLastError();
  return result == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(result));
}
}  // namespace pih::deepseek_v41
