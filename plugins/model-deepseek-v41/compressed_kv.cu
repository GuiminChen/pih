#include "compressed_kv.h"
#include "fp4_round.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__global__ void QuantizeStore(__nv_bfloat16* kv, __nv_bfloat16* cache, unsigned* error, unsigned first_slot) {
  // One full warp owns two independent 16-element groups. All lanes are live.
  const auto offset = static_cast<unsigned long long>(blockIdx.x) * 32 + threadIdx.x;
  const float value = __bfloat162float(kv[offset]);
  if (!isfinite(value)) atomicOr(error, 1U);
  float maximum = isfinite(value) ? fabsf(value) : 0.0f;
  for (unsigned delta = 8; delta; delta >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, delta, 16));
  const float unrounded_scale = fmaxf(maximum, 6.0f * 0x1p-9f) / 6.0f;
  __nv_fp8_e4m3 encoded_scale;
  encoded_scale.__x = __nv_cvt_float_to_fp8(unrounded_scale, __NV_NOSAT, __NV_E4M3);
  const float scale = float(encoded_scale);
  // Match the scale cast, including non-saturating overflow. Error-marked
  // output is never admissible even when later arithmetic happens to be finite.
  if (!isfinite(scale) || scale <= 0.0f) atomicOr(error, 2U);
  const float normalized = isfinite(value) ? fminf(fmaxf(value / scale, -6.0f), 6.0f) : 0.0f;
  const float result = RoundE2M1(normalized) * scale;
  const auto rounded = __float2bfloat16_rn(result);
  if (!isfinite(result) || !isfinite(__bfloat162float(rounded))) atomicOr(error, 2U);
  kv[offset] = rounded;
  cache[static_cast<unsigned long long>(first_slot) * 512 + offset] = rounded;
}
}
Status LaunchCompressedKv(const CompressedKvLaunch& x) {
  const auto validation = ValidateCompressedKv(x); if (!validation.ok()) return validation;
  QuantizeStore<<<x.rows * 16, 32, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(
      Ptr<__nv_bfloat16>(x.kv), Ptr<__nv_bfloat16>(x.cache), Ptr<unsigned>(x.error_flag), x.first_slot);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
}  // namespace pih::deepseek_v41
