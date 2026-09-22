#include "indexer_score.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__device__ float Read(__nv_bfloat16 value, unsigned* error) {
  const float result = __bfloat162float(value);
  if (!isfinite(result)) atomicOr(error, 1U);
  return result;
}
__device__ float Round(float value, unsigned* error) {
  const float rounded = __bfloat162float(__float2bfloat16_rn(value));
  if (!isfinite(value) || !isfinite(rounded)) atomicOr(error, 2U);
  return rounded;
}
__device__ float Sum(float value) {
  for (unsigned delta = 16; delta; delta >>= 1) value += __shfl_down_sync(0xffffffffU, value, delta);
  return value;
}
__global__ void Weights(const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, unsigned* error, unsigned heads) {
  const unsigned head = blockIdx.x % heads, token = blockIdx.x / heads;
  float sum = 0.0f;
  for (unsigned d = threadIdx.x; d < 5120; d += 32)
    sum += Read(input[static_cast<unsigned long long>(token) * 5120 + d], error) * Read(weight[head * 5120 + d], error);
  sum = Sum(sum);
  if (!threadIdx.x) {
    // 128^-0.5 * 32^-0.5 = 1/64. BF16 projection precedes scalar multiplication.
    const float scaled = Round(Round(sum, error) * (1.0f / 64.0f), error);
    output[blockIdx.x] = __float2bfloat16_rn(scaled);
  }
}
__global__ void Score(const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* weights,
    __nv_bfloat16* output, unsigned* error, unsigned heads, unsigned positions) {
  const unsigned position = blockIdx.x, token = blockIdx.y;
  float total = 0.0f;
  for (unsigned head = 0; head < heads; ++head) {
    float dot = 0.0f;
    for (unsigned d = threadIdx.x; d < 128; d += 32)
      dot += Read(query[(static_cast<unsigned long long>(token) * heads + head) * 128 + d], error) *
          Read(key[static_cast<unsigned long long>(position) * 128 + d], error);
    dot = Sum(dot);
    if (!threadIdx.x) {
      // Reference einsum and elementwise product each materialize BF16 before
      // the FP32 accumulation of the head sum and its final BF16 cast.
      const float rounded_dot = Round(dot, error);
      const float weight = Read(weights[static_cast<unsigned long long>(token) * heads + head], error);
      total += Round(fmaxf(rounded_dot, 0.0f) * weight, error);
    }
  }
  if (!threadIdx.x) output[static_cast<unsigned long long>(token) * positions + position] =
      __float2bfloat16_rn(Round(total, error));
}
Status LastError() {
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
}
Status LaunchIndexerWeights(const IndexerWeightsLaunch& x) {
  const auto validation = ValidateIndexerWeights(x); if (!validation.ok()) return validation;
  Weights<<<x.tokens * x.heads, 32, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(
      Ptr<const __nv_bfloat16>(x.input), Ptr<const __nv_bfloat16>(x.weight), Ptr<__nv_bfloat16>(x.output), Ptr<unsigned>(x.error_flag), x.heads);
  return LastError();
}
Status LaunchIndexerScore(const IndexerScoreLaunch& x) {
  const auto validation = ValidateIndexerScore(x); if (!validation.ok()) return validation;
  Score<<<dim3(x.positions, x.tokens), 32, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(
      Ptr<const __nv_bfloat16>(x.query), Ptr<const __nv_bfloat16>(x.key), Ptr<const __nv_bfloat16>(x.weights),
      Ptr<__nv_bfloat16>(x.output), Ptr<unsigned>(x.error_flag), x.heads, x.positions);
  return LastError();
}
}  // namespace pih::deepseek_v41
