#include "linear_fp8.h"
#include "linear_fp4.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Pointer(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__device__ float DecodeE4M3(unsigned char bits) {
  const unsigned exponent = (bits >> 3) & 15U, fraction = bits & 7U;
  if (exponent == 15 && fraction == 7) return CUDART_NAN_F;
  const float magnitude = exponent ? ldexpf(1.0F + fraction * 0.125F, int(exponent) - 7)
                                   : ldexpf(float(fraction), -9);
  return bits & 128U ? -magnitude : magnitude;
}
__global__ void Quantize(const __nv_bfloat16* input, unsigned char* output,
    unsigned char* scales, unsigned* error) {
  const unsigned group = blockIdx.x, lane = threadIdx.x;
  const auto index = static_cast<unsigned long long>(group) * 32 + lane;
  const float value = __bfloat162float(input[index]);
  if (!isfinite(value)) atomicOr(error, 1U);
  float maximum = isfinite(value) ? fabsf(value) : 0.0F;
  for (unsigned delta = 16; delta; delta >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, delta));
  const float scaled = fmaxf(maximum, 1e-4F) * (1.0F / 448.0F);
  const unsigned bits = __float_as_uint(scaled);
  const int exponent = int((bits >> 23) & 255U) - 127 + ((bits & 0x7fffffU) != 0);
  // A finite BF16 input with the 1e-4 floor keeps this in E8M0 range.
  const float scale = ldexpf(1.0F, exponent);
  if (!lane) scales[group] = static_cast<unsigned char>(exponent + 127);
  const float normalized = isfinite(value) ? fminf(fmaxf(value / scale, -448.0F), 448.0F) : 0.0F;
  output[index] = __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3);
}
// Functional native tiled implementation. Scale correction happens after each
// 32-wide dot, matching the reference grouping, not per individual operand.
// This is not a tensor-core throughput claim; numerical/performance qualification
// remains necessary before registering a production kernel capability.
template<bool Fp4Weight>
__global__ void Projection(const unsigned char* a, const unsigned char* b,
    const unsigned char* as, const unsigned char* bs, __nv_bfloat16* output,
    unsigned* error, unsigned tokens, unsigned in_features, unsigned out_features) {
  __shared__ float left[16][32], right[16][32];
  const unsigned tx = threadIdx.x, ty = threadIdx.y, linear = ty * 16 + tx;
  const unsigned token = blockIdx.y * 16 + ty, column = blockIdx.x * 16 + tx;
  float accumulated = 0.0F;
  for (unsigned group = 0; group < (in_features / 32); ++group) {
    for (unsigned element = linear; element < 512; element += 256) {
      const unsigned row = element / 32, k = element % 32;
      const unsigned input_row = blockIdx.y * 16 + row;
      const unsigned weight_row = blockIdx.x * 16 + row;
      const float av = input_row < tokens ? DecodeE4M3(a[static_cast<unsigned long long>(input_row) * in_features + group * 32 + k]) : 0.0F;
      float bv;
      if constexpr (Fp4Weight) {
        const auto offset = static_cast<unsigned long long>(weight_row) * (in_features / 2) + group * 16 + k / 2;
        const unsigned code = (b[offset] >> ((k & 1U) * 4)) & 15U;
        const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        bv = code & 8U ? -levels[code & 7U] : levels[code & 7U];
      } else {
        bv = DecodeE4M3(b[static_cast<unsigned long long>(weight_row) * in_features + group * 32 + k]);
      }
      if (!isfinite(av) || !isfinite(bv)) atomicOr(error, 1U);
      left[row][k] = av; right[row][k] = bv;
    }
    __syncthreads();
    if (token < tokens) {
      float dot = 0.0F;
      for (unsigned k = 0; k < 32; ++k) dot += left[ty][k] * right[tx][k];
      const unsigned char sa = as[static_cast<unsigned long long>(token) * (in_features / 32) + group];
      const unsigned char sb = bs[static_cast<unsigned long long>(Fp4Weight ? column : column / 32) * (in_features / 32) + group];
      if (sa == 255 || sb == 255) atomicOr(error, 1U);
      const float term = (dot * ldexpf(1.0F, int(sa) - 127)) * ldexpf(1.0F, int(sb) - 127);
      accumulated += term;
      if (!isfinite(term) || !isfinite(accumulated)) atomicOr(error, 2U);
    }
    __syncthreads();
  }
  if (token < tokens) {
    const auto rounded = __float2bfloat16_rn(accumulated);
    if (!isfinite(__bfloat162float(rounded))) atomicOr(error, 2U);
    output[static_cast<unsigned long long>(token) * out_features + column] = rounded;
  }
}
Status LaunchStatus() { const auto e = cudaGetLastError(); return e == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(e)); }
}
Status LaunchFp8Linear(const Fp8LinearLaunch& x) {
  const auto validation = ValidateFp8Linear(x); if (!validation.ok()) return validation;
  const auto stream = reinterpret_cast<cudaStream_t>(x.stream);
  Quantize<<<x.rows * (x.in_features / 32), 32, 0, stream>>>(Pointer<const __nv_bfloat16>(x.input),
      Pointer<unsigned char>(x.quantized), Pointer<unsigned char>(x.activation_scales), Pointer<unsigned>(x.error_flag));
  const auto quantized = LaunchStatus(); if (!quantized.ok()) return quantized;
  Projection<false><<<dim3(x.out_features / 16, (x.rows + 15) / 16), dim3(16,16), 0, stream>>>(
      Pointer<const unsigned char>(x.quantized), Pointer<const unsigned char>(x.weight),
      Pointer<const unsigned char>(x.activation_scales), Pointer<const unsigned char>(x.weight_scales),
      Pointer<__nv_bfloat16>(x.output), Pointer<unsigned>(x.error_flag), x.rows, x.in_features, x.out_features);
  return LaunchStatus();
}
Status LaunchFp4Linear(const Fp4LinearLaunch& x) {
  const auto validation = ValidateFp4Linear(x); if (!validation.ok()) return validation;
  const auto stream = reinterpret_cast<cudaStream_t>(x.stream);
  Quantize<<<x.rows * (x.in_features / 32), 32, 0, stream>>>(Pointer<const __nv_bfloat16>(x.input),
      Pointer<unsigned char>(x.quantized), Pointer<unsigned char>(x.activation_scales), Pointer<unsigned>(x.error_flag));
  const auto quantized = LaunchStatus(); if (!quantized.ok()) return quantized;
  Projection<true><<<dim3(x.out_features / 16, (x.rows + 15) / 16), dim3(16,16), 0, stream>>>(
      Pointer<const unsigned char>(x.quantized), Pointer<const unsigned char>(x.weight),
      Pointer<const unsigned char>(x.activation_scales), Pointer<const unsigned char>(x.weight_scales),
      Pointer<__nv_bfloat16>(x.output), Pointer<unsigned>(x.error_flag), x.rows, x.in_features, x.out_features);
  return LaunchStatus();
}
}  // namespace pih::deepseek_v41
