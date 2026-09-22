#include "token_embedding.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__global__ void Lookup(const unsigned* ids, const __nv_bfloat16* weight, __nv_bfloat16* hidden,
    unsigned* error, unsigned tokens, unsigned first, unsigned rows) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= tokens * 5120) return;
  const unsigned id = ids[i / 5120]; float value = 0;
  if (id >= 129280) atomicOr(error, 1U);
  else if (id >= first && id - first < rows) {
    value = __bfloat162float(weight[static_cast<unsigned long long>(id - first) * 5120 + i % 5120]);
    if (!isfinite(value)) { atomicOr(error, 1U); value = 0; }
  }
  hidden[i] = __float2bfloat16_rn(value);
}
__global__ void Expand(const __nv_bfloat16* hidden, __nv_bfloat16* residual, unsigned* error, unsigned tokens) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= tokens * 4 * 5120) return;
  float value = __bfloat162float(hidden[(i / (4 * 5120)) * 5120 + i % 5120]);
  if (!isfinite(value)) { atomicOr(error, 2U); value = 0; }
  residual[i] = __float2bfloat16_rn(value);
}
Status Last() { const auto e = cudaGetLastError(); return e == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(e)); }
}
Status LaunchTokenEmbeddingLookup(const TokenEmbeddingLaunch& x) {
  const auto valid = ValidateTokenEmbedding(x); if (!valid.ok()) return valid;
  const unsigned rows = 129280 / x.world_size;
  Lookup<<<(x.tokens * 5120 + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(
      Ptr<const unsigned>(x.ids), Ptr<const __nv_bfloat16>(x.weight), Ptr<__nv_bfloat16>(x.hidden),
      Ptr<unsigned>(x.error_flag), x.tokens, x.rank * rows, rows);
  return Last();
}
Status LaunchTokenEmbeddingExpand(const TokenEmbeddingLaunch& x) {
  const auto valid = ValidateTokenEmbedding(x); if (!valid.ok()) return valid;
  Expand<<<(x.tokens * 4 * 5120 + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(
      Ptr<const __nv_bfloat16>(x.hidden), Ptr<__nv_bfloat16>(x.residual), Ptr<unsigned>(x.error_flag), x.tokens);
  const auto expanded = Last(); if (!expanded.ok()) return expanded;
  return LaunchMhcInitialPre(x.pre, x.tokens, x.stream);
}
}  // namespace pih::deepseek_v41
