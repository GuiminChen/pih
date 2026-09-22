#include "mhc_launch.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion region) { return reinterpret_cast<T*>(region.address); }
__device__ float Sum(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (unsigned width = 128; width; width >>= 1) {
    if (threadIdx.x < width) scratch[threadIdx.x] += scratch[threadIdx.x + width];
    __syncthreads();
  }
  return scratch[0];
}
__device__ float Sigmoid(float x) { return 1.0F / (1.0F + expf(-x)); }
__global__ void Mix(const __nv_bfloat16* residual, const float* fn, const float* scale,
    const float* base, float* pre, float* post, float* comb, unsigned* error) {
  __shared__ float scratch[256], mixes[24], inverse;
  const unsigned token = blockIdx.x;
  const auto offset = static_cast<unsigned long long>(token) * 20480;
  float square = 0;
  for (unsigned d = threadIdx.x; d < 20480; d += 256) {
    const float x = __bfloat162float(residual[offset + d]);
    if (!isfinite(x)) atomicOr(error, 1U);
    square += x * x;
  }
  const float total = Sum(square, scratch);
  if (!threadIdx.x) {
    inverse = rsqrtf(total / 20480.0F + 1e-20F);
    if (!isfinite(total) || !isfinite(inverse)) atomicOr(error, 2U);
  }
  __syncthreads();
  for (unsigned row = 0; row < 24; ++row) {
    float dot = 0;
    for (unsigned d = threadIdx.x; d < 20480; d += 256) {
      const float weight = fn[row * 20480 + d];
      if (!isfinite(weight)) atomicOr(error, 1U);
      dot += __bfloat162float(residual[offset + d]) * weight;
    }
    const float projected = Sum(dot, scratch) * inverse;
    if (!threadIdx.x) {
      mixes[row] = projected;
      if (!isfinite(projected)) atomicOr(error, 2U);
    }
    __syncthreads();
  }
  if (threadIdx.x) return;
  for (unsigned i = 0; i < 3; ++i) if (!isfinite(scale[i])) atomicOr(error, 1U);
  for (unsigned i = 0; i < 24; ++i) if (!isfinite(base[i])) atomicOr(error, 1U);
  for (unsigned i = 0; i < 4; ++i) {
    const float pre_logit = mixes[i] * scale[0] + base[i];
    const float post_logit = mixes[4 + i] * scale[1] + base[4 + i];
    if (!isfinite(pre_logit) || !isfinite(post_logit)) atomicOr(error, 2U);
    pre[token * 4 + i] = Sigmoid(pre_logit) + 1e-6F;
    post[token * 4 + i] = 2 * Sigmoid(post_logit);
  }
  float matrix[16];
  for (unsigned row = 0; row < 4; ++row) {
    float maximum = -FLT_MAX, sum = 0;
    for (unsigned col = 0; col < 4; ++col) {
      const unsigned i = row * 4 + col;
      matrix[i] = mixes[8 + i] * scale[2] + base[8 + i];
      if (!isfinite(matrix[i])) atomicOr(error, 2U);
      maximum = fmaxf(maximum, matrix[i]);
    }
    for (unsigned col = 0; col < 4; ++col) { matrix[row * 4 + col] = expf(matrix[row * 4 + col] - maximum); sum += matrix[row * 4 + col]; }
    for (unsigned col = 0; col < 4; ++col) matrix[row * 4 + col] = matrix[row * 4 + col] / sum + 1e-6F;
  }
  for (unsigned iteration = 0; iteration < 20; ++iteration) {
    if (iteration) {
      for (unsigned row = 0; row < 4; ++row) {
        float sum = 0;
        for (unsigned col = 0; col < 4; ++col) sum += matrix[row * 4 + col];
        for (unsigned col = 0; col < 4; ++col) matrix[row * 4 + col] /= sum + 1e-6F;
      }
    }
    for (unsigned col = 0; col < 4; ++col) {
      float sum = 0;
      for (unsigned row = 0; row < 4; ++row) sum += matrix[row * 4 + col];
      for (unsigned row = 0; row < 4; ++row) matrix[row * 4 + col] /= sum + 1e-6F;
    }
  }
  for (unsigned i = 0; i < 16; ++i) {
    if (!isfinite(matrix[i])) atomicOr(error, 2U);
    comb[token * 16 + i] = matrix[i];
  }
}
__global__ void Pre(const __nv_bfloat16* residual, const float* pre, __nv_bfloat16* output, unsigned* error) {
  const unsigned token = blockIdx.x;
  for (unsigned d = threadIdx.x; d < 5120; d += 256) {
    float value = 0;
    for (unsigned copy = 0; copy < 4; ++copy) {
      const float weight = pre[token * 4 + copy];
      const float x = __bfloat162float(residual[(static_cast<unsigned long long>(token) * 4 + copy) * 5120 + d]);
      if (!isfinite(x) || !isfinite(weight)) atomicOr(error, 1U);
      value += weight * x;
    }
    const auto rounded = __float2bfloat16_rn(value);
    if (!isfinite(value) || !isfinite(__bfloat162float(rounded))) atomicOr(error, 2U);
    output[static_cast<unsigned long long>(token) * 5120 + d] = rounded;
  }
}
__global__ void Post(const __nv_bfloat16* sublayer, const __nv_bfloat16* residual,
    const float* post, const float* comb, __nv_bfloat16* output, unsigned* error) {
  const unsigned token = blockIdx.x;
  for (unsigned d = threadIdx.x; d < 5120; d += 256) {
    const float x = __bfloat162float(sublayer[static_cast<unsigned long long>(token) * 5120 + d]);
    if (!isfinite(x)) atomicOr(error, 1U);
    for (unsigned target = 0; target < 4; ++target) {
      float residual_sum = 0;
      const float weight = post[token * 4 + target];
      if (!isfinite(weight)) atomicOr(error, 1U);
      for (unsigned source = 0; source < 4; ++source) {
        const float mixing = comb[token * 16 + source * 4 + target];
        const float r = __bfloat162float(residual[(static_cast<unsigned long long>(token) * 4 + source) * 5120 + d]);
        if (!isfinite(mixing) || !isfinite(r)) atomicOr(error, 1U);
        residual_sum += mixing * r;
      }
      const float value = weight * x + residual_sum;
      const auto rounded = __float2bfloat16_rn(value);
      if (!isfinite(value) || !isfinite(__bfloat162float(rounded))) atomicOr(error, 2U);
      output[(static_cast<unsigned long long>(token) * 4 + target) * 5120 + d] = rounded;
    }
  }
}
__global__ void InitialPre(float* pre, unsigned tokens) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < tokens * 4) pre[index] = index % 4 == 0 ? 1.0F : 0.0F;
}
Status LastError() { const auto e = cudaGetLastError(); return e == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(e)); }
}
Status LaunchMhcMix(const MhcMixLaunch& x) {
  auto status = ValidateMhcMix(x); if (!status.ok()) return status;
  Mix<<<x.tokens, 256, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(Ptr<const __nv_bfloat16>(x.residual),
      Ptr<const float>(x.fn), Ptr<const float>(x.scale), Ptr<const float>(x.base), Ptr<float>(x.pre),
      Ptr<float>(x.post), Ptr<float>(x.comb), Ptr<unsigned>(x.error_flag));
  return LastError();
}
Status LaunchMhcPre(const MhcPreLaunch& x) {
  auto status = ValidateMhcPre(x); if (!status.ok()) return status;
  Pre<<<x.tokens, 256, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(Ptr<const __nv_bfloat16>(x.residual),
      Ptr<const float>(x.pre), Ptr<__nv_bfloat16>(x.output), Ptr<unsigned>(x.error_flag));
  return LastError();
}
Status LaunchMhcPost(const MhcPostLaunch& x) {
  auto status = ValidateMhcPost(x); if (!status.ok()) return status;
  Post<<<x.tokens, 256, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(Ptr<const __nv_bfloat16>(x.sublayer),
      Ptr<const __nv_bfloat16>(x.residual), Ptr<const float>(x.post), Ptr<const float>(x.comb),
      Ptr<__nv_bfloat16>(x.output), Ptr<unsigned>(x.error_flag));
  return LastError();
}
Status LaunchMhcInitialPre(EngramDeviceRegion pre, std::uint32_t tokens, std::uintptr_t stream) {
  auto status = ValidateMhcInitialPre(pre, tokens, stream); if (!status.ok()) return status;
  InitialPre<<<(tokens * 4 + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(Ptr<float>(pre), tokens);
  return LastError();
}
}  // namespace pih::deepseek_v41
