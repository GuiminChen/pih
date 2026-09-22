#include "attention_output.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
template<bool Round>
__global__ void ConvertOutput(__nv_bfloat16* local, float* reduced, unsigned* error, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  if constexpr (Round) {
    auto value = __float2bfloat16_rn(reduced[i]);
    if (!isfinite(reduced[i]) || !isfinite(__bfloat162float(value))) {
      atomicOr(error, 2U); value = __float2bfloat16_rn(0.0F);
    }
    local[i] = value;
  } else {
    float value = __bfloat162float(local[i]);
    if (!isfinite(value)) { atomicOr(error, 1U); value = 0.0F; }
    reduced[i] = value;
  }
}
__global__ void GroupedOutput(const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, unsigned* error, unsigned tokens, unsigned groups) {
  __shared__ float left[16][32], right[16][32];
  const unsigned tx = threadIdx.x, ty = threadIdx.y, thread = ty * 16 + tx;
  const unsigned group = blockIdx.z, token = blockIdx.y * 16 + ty, channel = blockIdx.x * 16 + tx;
  float accumulated = 0;
  for (unsigned tile = 0; tile < 128; ++tile) {
    for (unsigned element = thread; element < 512; element += 256) {
      const unsigned row = element / 32, column = tile * 32 + element % 32;
      const unsigned input_token = blockIdx.y * 16 + row, output_channel = blockIdx.x * 16 + row;
      const float a = input_token < tokens ? __bfloat162float(input[
          (static_cast<unsigned long long>(input_token) * groups + group) * 4096 + column]) : 0.0F;
      const float b = __bfloat162float(weight[(static_cast<unsigned long long>(group) * 1024 + output_channel) * 4096 + column]);
      if (!isfinite(a) || !isfinite(b)) atomicOr(error, 1U);
      left[row][element % 32] = a; right[row][element % 32] = b;
    }
    __syncthreads();
    if (token < tokens)
      for (unsigned k = 0; k < 32; ++k) accumulated += left[ty][k] * right[tx][k];
    __syncthreads();
  }
  if (token < tokens) {
    const auto rounded = __float2bfloat16_rn(accumulated);
    if (!isfinite(accumulated) || !isfinite(__bfloat162float(rounded))) atomicOr(error, 2U);
    output[(static_cast<unsigned long long>(token) * groups + group) * 1024 + channel] = rounded;
  }
}
}
Status LaunchGroupedOutput(const GroupedOutputLaunch& x) {
  const auto validation = ValidateGroupedOutput(x); if (!validation.ok()) return validation;
  GroupedOutput<<<dim3(64, (x.tokens + 15) / 16, x.groups), dim3(16, 16), 0,
      reinterpret_cast<cudaStream_t>(x.stream)>>>(Ptr<const __nv_bfloat16>(x.input), Ptr<const __nv_bfloat16>(x.weight),
      Ptr<__nv_bfloat16>(x.output), Ptr<unsigned>(x.error_flag), x.tokens, x.groups);
  const auto result = cudaGetLastError();
  return result == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(result));
}
namespace {
template<bool Round> Status ConvertAttentionOutput(const AttentionLocalOutputLaunch& x) {
  const auto validation = ValidateAttentionLocalOutput(x); if (!validation.ok()) return validation;
  const unsigned count = x.linear.rows * 5120;
  ConvertOutput<Round><<<(count + 255) / 256, 256, 0,
      reinterpret_cast<cudaStream_t>(x.linear.stream)>>>(Ptr<__nv_bfloat16>(x.linear.output),
      Ptr<float>(x.reduction), Ptr<unsigned>(x.linear.error_flag), count);
  const auto result = cudaGetLastError();
  return result == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(result));
}
}
Status PromoteAttentionOutput(const AttentionLocalOutputLaunch& x) { return ConvertAttentionOutput<false>(x); }
Status RoundAttentionOutput(const AttentionLocalOutputLaunch& x) { return ConvertAttentionOutput<true>(x); }
}  // namespace pih::deepseek_v41
