#include "indexer_input.h"
#include "fp4_round.cuh"
#include <cuda_bf16.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__global__ void KeyProjection(const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, unsigned* error, unsigned rows) {
  __shared__ float a[16][32], b[16][32];
  const unsigned row = blockIdx.y * 16 + threadIdx.y, column = blockIdx.x * 16 + threadIdx.x;
  const unsigned lane = threadIdx.y * 16 + threadIdx.x;
  float sum = 0.0f;
  for (unsigned base = 0; base < 512; base += 32) {
    for (unsigned i = lane; i < 512; i += 256) {
      const unsigned r = i / 32, k = i % 32, source_row = blockIdx.y * 16 + r;
      a[r][k] = source_row < rows ? __bfloat162float(input[static_cast<unsigned long long>(source_row) * 512 + base + k]) : 0.0f;
      b[r][k] = __bfloat162float(weight[(blockIdx.x * 16 + r) * 512 + base + k]);
      if (!isfinite(a[r][k]) || !isfinite(b[r][k])) atomicOr(error, 1U);
    }
    __syncthreads();
    for (unsigned k = 0; k < 32; ++k) sum += a[threadIdx.y][k] * b[threadIdx.x][k];
    __syncthreads();
  }
  if (row < rows) {
    const auto rounded = __float2bfloat16_rn(sum);
    if (!isfinite(sum) || !isfinite(__bfloat162float(rounded))) atomicOr(error, 2U);
    output[static_cast<unsigned long long>(row) * 128 + column] = rounded;
  }
}
__global__ void Quantize(__nv_bfloat16* values, unsigned* error) {
  const auto offset = static_cast<unsigned long long>(blockIdx.x) * 32 + threadIdx.x;
  const float value = __bfloat162float(values[offset]);
  if (!isfinite(value)) atomicOr(error, 1U);
  float maximum = isfinite(value) ? fabsf(value) : 0.0f;
  for (unsigned delta = 16; delta; delta >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, delta));
  const float raw_scale = fmaxf(maximum, 6.0f * 0x1p-126f) * (1.0f / 6.0f);
  const unsigned bits = __float_as_uint(raw_scale);
  const int exponent = int((bits >> 23) & 255U) - 127 + ((bits & 0x7fffffU) != 0);
  const float scale = ldexpf(1.0f, exponent);
  if (!isfinite(scale) || scale <= 0.0f) atomicOr(error, 2U);
  const float normalized = isfinite(value) ? fminf(fmaxf(value / scale, -6.0f), 6.0f) : 0.0f;
  const float result = RoundE2M1(normalized) * scale;
  const auto rounded = __float2bfloat16_rn(result);
  if (!isfinite(result) || !isfinite(__bfloat162float(rounded))) atomicOr(error, 2U);
  values[offset] = rounded;
}
}
Status LaunchIndexerQuantize(const IndexerQuantizeLaunch& x) {
  const auto validation = ValidateIndexerQuantize(x); if (!validation.ok()) return validation;
  Quantize<<<x.tokens * x.heads * 4, 32, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(
      Ptr<__nv_bfloat16>(x.values), Ptr<unsigned>(x.error_flag));
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
Status LaunchIndexerKeyProjection(const IndexerKeyProjectionLaunch& x) {
  const auto validation = ValidateIndexerKeyProjection(x); if (!validation.ok()) return validation;
  KeyProjection<<<dim3(8, (x.rows + 15) / 16), dim3(16, 16), 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(
      Ptr<const __nv_bfloat16>(x.input), Ptr<const __nv_bfloat16>(x.weight),
      Ptr<__nv_bfloat16>(x.output), Ptr<unsigned>(x.error_flag), x.rows);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
Status LaunchIndexerKeyCache(const IndexerKeyCacheLaunch& x) {
  const auto validation = ValidateIndexerKeyCache(x); if (!validation.ok()) return validation;
  const auto quantize = LaunchIndexerQuantize(x.quantize); if (!quantize.ok()) return quantize;
  const auto error = cudaMemcpyAsync(Ptr<__nv_bfloat16>(x.cache) + static_cast<std::uint64_t>(x.first_slot) * 128,
      Ptr<const __nv_bfloat16>(x.quantize.values), x.quantize.values.bytes, cudaMemcpyDeviceToDevice,
      reinterpret_cast<cudaStream_t>(x.quantize.stream));
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
}  // namespace pih::deepseek_v41
