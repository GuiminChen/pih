#include "window_kv.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__global__ void QuantizeDequantize(__nv_bfloat16* kv, unsigned* error) {
  const auto index = static_cast<unsigned long long>(blockIdx.x) * 32 + threadIdx.x;
  const float value = __bfloat162float(kv[index]);
  if (!isfinite(value)) atomicOr(error, 1U);
  float maximum = isfinite(value) ? fabsf(value) : 0.0F;
  for (unsigned delta = 16; delta; delta >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, delta));
  const unsigned bits = __float_as_uint(fmaxf(maximum, 1e-4F) * (1.0F / 448.0F));
  const int exponent = int((bits >> 23) & 255U) - 127 + ((bits & 0x7fffffU) != 0);
  const float scale = ldexpf(1.0F, exponent);
  const float normalized = isfinite(value) ? fminf(fmaxf(value / scale, -448.0F), 448.0F) : 0.0F;
  __nv_fp8_e4m3 encoded;
  encoded.__x = __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3);
  const float dequantized = float(encoded) * scale;
  const auto rounded = __float2bfloat16_rn(dequantized);
  if (!isfinite(dequantized) || !isfinite(__bfloat162float(rounded))) atomicOr(error, 2U);
  kv[index] = rounded;
}
__global__ void WriteRing(const __nv_bfloat16* kv, __nv_bfloat16* ring, unsigned first, unsigned start) {
  const unsigned token = first + blockIdx.x, slot = (start + token) % 128;
  for (unsigned column = threadIdx.x; column < 512; column += 256)
    ring[slot * 512 + column] = kv[static_cast<unsigned long long>(token) * 512 + column];
}
Status LastError() { const auto e = cudaGetLastError(); return e == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(e)); }
}
Status LaunchWindowKv(const WindowKvLaunch& x) {
  const auto validation = ValidateWindowKv(x); if (!validation.ok()) return validation;
  const auto stream = reinterpret_cast<cudaStream_t>(x.stream);
  QuantizeDequantize<<<x.tokens * 16, 32, 0, stream>>>(Ptr<__nv_bfloat16>(x.kv), Ptr<unsigned>(x.error_flag));
  const auto quantized = LastError(); if (!quantized.ok()) return quantized;
  const unsigned retained = x.tokens < 128 ? x.tokens : 128;
  WriteRing<<<retained, 256, 0, stream>>>(Ptr<const __nv_bfloat16>(x.kv), Ptr<__nv_bfloat16>(x.ring),
      x.tokens - retained, x.start);
  return LastError();
}
}  // namespace pih::deepseek_v41
