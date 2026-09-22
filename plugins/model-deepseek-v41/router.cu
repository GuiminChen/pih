#include "router.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__global__ void Logits(const __nv_bfloat16* input, const float* weight, float* output, unsigned* error, unsigned experts) {
  const unsigned expert = blockIdx.x % experts, token = blockIdx.x / experts;
  float sum = 0.0f;
  for (unsigned d = threadIdx.x; d < 5120; d += 32) {
    const float a = __bfloat162float(input[static_cast<unsigned long long>(token) * 5120 + d]);
    const float b = weight[expert * 5120 + d];
    if (!isfinite(a) || !isfinite(b)) atomicOr(error, 1U);
    sum += a * b;
  }
  for (unsigned delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, delta);
  if (!threadIdx.x) {
    if (!isfinite(sum)) atomicOr(error, 2U);
    output[blockIdx.x] = sum;
  }
}
__global__ void Select(const float* logits, const float* bias, const float* image_bias, const unsigned char* mask,
    unsigned* indices, float* weights, unsigned* error, unsigned experts, unsigned picks) {
  const unsigned token = blockIdx.x;
  float best[6], raw[6]; unsigned ids[6];
  unsigned count = 0;
  const unsigned image = mask ? mask[token] : 0;
  if (image > 1) atomicOr(error, 1U);
  const float* correction = image == 1 ? image_bias : bias;
  for (unsigned expert = 0; expert < experts; ++expert) {
    const float logit = logits[static_cast<unsigned long long>(token) * experts + expert];
    const float b = correction[expert];
    if (!isfinite(logit) || !isfinite(b)) atomicOr(error, 1U);
    // Frozen temperature=1; PyTorch softplus defaults to the linear branch >20.
    const float score = sqrtf(logit > 20.0f ? logit : log1pf(expf(logit)));
    const float adjusted = score + b;
    if (!isfinite(score) || !isfinite(adjusted)) atomicOr(error, 2U);
    unsigned at = 0;
    while (at < count && (best[at] > adjusted || (best[at] == adjusted && ids[at] < expert))) ++at;
    if (at >= picks) continue;
    const unsigned last = count < picks ? count++ : picks - 1;
    for (unsigned i = last; i > at; --i) { best[i] = best[i - 1]; raw[i] = raw[i - 1]; ids[i] = ids[i - 1]; }
    best[at] = adjusted; raw[at] = score; ids[at] = expert;
  }
  float sum = 0.0f;
  for (unsigned i = 0; i < picks; ++i) sum += raw[i];
  for (unsigned i = 0; i < picks; ++i) {
    const float weight = (raw[i] / (sum + 1e-20f)) * 1.5f;
    if (!isfinite(weight)) atomicOr(error, 2U);
    const auto offset = static_cast<unsigned long long>(token) * picks + i;
    indices[offset] = ids[i]; weights[offset] = weight;
  }
}
Status LastError() { const auto e = cudaGetLastError(); return e == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(e)); }
}
Status LaunchRouter(const RouterLaunch& x) {
  const auto validation = ValidateRouter(x); if (!validation.ok()) return validation;
  const unsigned experts = x.layer < 40 ? 384 : 128, picks = x.layer < 40 ? 6 : 3;
  const auto stream = reinterpret_cast<cudaStream_t>(x.stream);
  Logits<<<x.tokens * experts, 32, 0, stream>>>(Ptr<const __nv_bfloat16>(x.input), Ptr<const float>(x.weight),
      Ptr<float>(x.logits), Ptr<unsigned>(x.error_flag), experts);
  const auto logits = LastError(); if (!logits.ok()) return logits;
  Select<<<x.tokens, 1, 0, stream>>>(Ptr<const float>(x.logits), Ptr<const float>(x.bias), Ptr<const float>(x.image_bias),
      Ptr<const unsigned char>(x.image_mask), Ptr<unsigned>(x.indices), Ptr<float>(x.route_weights), Ptr<unsigned>(x.error_flag), experts, picks);
  return LastError();
}
}  // namespace pih::deepseek_v41
