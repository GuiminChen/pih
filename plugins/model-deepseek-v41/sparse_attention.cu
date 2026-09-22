#include "sparse_attention.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__global__ void Attention(const __nv_bfloat16* query, const __nv_bfloat16* kv, const float* sink,
    const int* indices, __nv_bfloat16* output, unsigned* error, unsigned heads, unsigned kv_rows, unsigned picks) {
  const unsigned vector = blockIdx.x, token = vector / heads, head = vector % heads, lane = threadIdx.x;
  const auto q_offset = static_cast<unsigned long long>(vector) * 512;
  const float q0 = __bfloat162float(query[q_offset + lane]);
  const float q1 = __bfloat162float(query[q_offset + lane + 256]);
  if (!isfinite(q0) || !isfinite(q1)) atomicOr(error, 1U);
  __shared__ float scratch[256], scores[64], probabilities[64], maximum, denominator, rescale;
  __shared__ int gathered[64];
  __shared__ bool any_valid;
  if (!lane) { maximum = -1e30F; denominator = 0; any_valid = false; }
  float out0 = 0, out1 = 0;
  __syncthreads();
  for (unsigned block = 0; block < (picks + 63) / 64; ++block) {
    if (lane < 64) {
      const unsigned pick = block * 64 + lane;
      const int index = pick < picks ? indices[static_cast<unsigned long long>(token) * picks + pick] : -1;
      if (index < -1 || (index >= 0 && static_cast<unsigned>(index) >= kv_rows)) atomicOr(error, 1U);
      gathered[lane] = index >= 0 && static_cast<unsigned>(index) < kv_rows ? index : -1;
    }
    __syncthreads();
    for (unsigned pick = 0; pick < 64; ++pick) {
      const int index = gathered[pick];
      float dot = 0;
      if (index >= 0) {
        const auto k_offset = static_cast<unsigned long long>(index) * 512;
        const float k0 = __bfloat162float(kv[k_offset + lane]);
        const float k1 = __bfloat162float(kv[k_offset + lane + 256]);
        if (!isfinite(k0) || !isfinite(k1)) atomicOr(error, 1U);
        dot = q0 * k0 + q1 * k1;
      }
      scratch[lane] = dot;
      __syncthreads();
      for (unsigned stride = 128; stride; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
      }
      if (!lane) {
        const float score = scratch[0] * rsqrtf(512.0F);
        if (index >= 0 && !isfinite(score)) atomicOr(error, 2U);
        scores[pick] = index >= 0 && isfinite(score) ? score : -CUDART_INF_F;
        any_valid = any_valid || index >= 0;
      }
      __syncthreads();
    }
    if (!lane) {
      const float previous = maximum;
      for (unsigned pick = 0; pick < 64; ++pick) maximum = fmaxf(maximum, scores[pick]);
      rescale = expf(previous - maximum);
      float sum = 0;
      for (unsigned pick = 0; pick < 64; ++pick) {
        const float probability = expf(scores[pick] - maximum);
        sum += probability;
        // Reference rounds the value-GEMM probabilities to BF16, but keeps
        // softmax's denominator in FP32. Do not round the denominator.
        probabilities[pick] = __bfloat162float(__float2bfloat16_rn(probability));
      }
      denominator = denominator * rescale + sum;
      if (!isfinite(denominator)) atomicOr(error, 2U);
    }
    __syncthreads();
    out0 *= rescale; out1 *= rescale;
    for (unsigned pick = 0; pick < 64; ++pick) {
      if (gathered[pick] < 0) continue;
      const auto offset = static_cast<unsigned long long>(gathered[pick]) * 512;
      out0 += probabilities[pick] * __bfloat162float(kv[offset + lane]);
      out1 += probabilities[pick] * __bfloat162float(kv[offset + lane + 256]);
    }
    __syncthreads();
  }
  if (!lane) {
    const float bias = sink[head];
    if (!isfinite(bias)) atomicOr(error, 1U);
    denominator += expf(bias - maximum);
    // +infinity here is permitted: a dominant finite sink yields zero output.
    // The reference convention for an entirely masked row is also zero.
    if (any_valid && (!(denominator > 0) || isnan(denominator))) atomicOr(error, 2U);
  }
  __syncthreads();
  if (!isfinite(out0) || !isfinite(out1)) atomicOr(error, 2U);
  out0 = any_valid ? out0 / denominator : 0.0F;
  out1 = any_valid ? out1 / denominator : 0.0F;
  const auto rounded0 = __float2bfloat16_rn(out0), rounded1 = __float2bfloat16_rn(out1);
  if (!isfinite(__bfloat162float(rounded0)) || !isfinite(__bfloat162float(rounded1))) atomicOr(error, 2U);
  output[q_offset + lane] = rounded0; output[q_offset + lane + 256] = rounded1;
}
}
Status LaunchSparseAttention(const SparseAttentionLaunch& x) {
  const auto validation = ValidateSparseAttention(x); if (!validation.ok()) return validation;
  Attention<<<x.tokens * x.heads, 256, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(
      Ptr<const __nv_bfloat16>(x.query), Ptr<const __nv_bfloat16>(x.kv), Ptr<const float>(x.sink),
      Ptr<const int>(x.indices), Ptr<__nv_bfloat16>(x.output), Ptr<unsigned>(x.error_flag), x.heads, x.kv_rows, x.picks);
  const auto result = cudaGetLastError();
  return result == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(result));
}
}  // namespace pih::deepseek_v41
