#include "expert_fp8.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__global__ void Activation(const __nv_bfloat16* gate, const __nv_bfloat16* up,
    const float* route_weights, __nv_bfloat16* output, unsigned* error) {
  const unsigned row = blockIdx.x;
  const float weight = route_weights ? route_weights[row] : 1.0f;
  if (!isfinite(weight) || weight < 0.0f) atomicOr(error, 1U);
  for (unsigned column = threadIdx.x; column < 2304; column += 256) {
    const auto offset = static_cast<unsigned long long>(row) * 2304 + column;
    float g = __bfloat162float(gate[offset]), u = __bfloat162float(up[offset]);
    if (!isfinite(g) || !isfinite(u)) atomicOr(error, 1U);
    g = fminf(g, 10.0f); u = fminf(fmaxf(u, -10.0f), 10.0f);
    const float activated = (g / (1.0f + expf(-g))) * u;
    // Routed expert weighting occurs before BF16 cast/down projection.
    const float value = route_weights ? weight * activated : activated;
    const auto rounded = __float2bfloat16_rn(value);
    if (!isfinite(value) || !isfinite(__bfloat162float(rounded))) atomicOr(error, 2U);
    output[offset] = rounded;
  }
}
}
Status LaunchExpertActivation(const ExpertActivationLaunch& x) {
  const auto validation = ValidateExpertActivation(x); if (!validation.ok()) return validation;
  Activation<<<x.rows, 256, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(Ptr<const __nv_bfloat16>(x.gate),
      Ptr<const __nv_bfloat16>(x.up), Ptr<const float>(x.route_weights), Ptr<__nv_bfloat16>(x.output), Ptr<unsigned>(x.error_flag));
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
namespace {
__global__ void Merge(const float* routed, const __nv_bfloat16* shared,
    __nv_bfloat16* output, unsigned* error, unsigned elements) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float left = routed[index], right = __bfloat162float(shared[index]);
  const float sum = left + right;
  const auto value = __float2bfloat16_rn(sum);
  if (!isfinite(left) || !isfinite(right) || !isfinite(sum) || !isfinite(__bfloat162float(value))) {
    atomicOr(error, 2U);
    output[index] = __float2bfloat16_rn(0.0f);
  } else output[index] = value;
}
}
Status LaunchExpertMerge(const ExpertMergeLaunch& x) {
  const auto validation = ValidateExpertMerge(x); if (!validation.ok()) return validation;
  const unsigned elements = x.tokens * 5120U;
  Merge<<<(elements + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(
      reinterpret_cast<const float*>(x.routed.address), reinterpret_cast<const __nv_bfloat16*>(x.shared.address),
      reinterpret_cast<__nv_bfloat16*>(x.output.address), reinterpret_cast<unsigned*>(x.error_flag.address), elements);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
}  // namespace pih::deepseek_v41
