#include "pih/backend/cuda/deepseek_expert_swiglu.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

__global__ void expert_swiglu_kernel(const __nv_bfloat16* gate,
                                     const __nv_bfloat16* up,
                                     const float* route_weights,
                                     __nv_bfloat16* output,
                                     std::uint32_t* error_flag,
                                     std::uint64_t element_count) {
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const auto token = static_cast<std::uint32_t>(
        index / DeepSeekExpertSwiGluLaunch::kIntermediateSize);
    const auto gate_value = __bfloat162float(gate[index]);
    const auto up_value = __bfloat162float(up[index]);
    const auto route_weight = route_weights[token];
    if (!isfinite(gate_value) || !isfinite(up_value) ||
        !isfinite(route_weight) || route_weight < 0.0F) {
      atomicOr(error_flag, 1U);
      continue;
    }
    const auto clamped_gate = fminf(gate_value,
                                    DeepSeekExpertSwiGluLaunch::kLimit);
    const auto clamped_up =
        fminf(fmaxf(up_value, -DeepSeekExpertSwiGluLaunch::kLimit),
              DeepSeekExpertSwiGluLaunch::kLimit);
    const auto value =
        (clamped_gate / (1.0F + expf(-clamped_gate))) * clamped_up *
        route_weight;
    if (!isfinite(value)) {
      atomicOr(error_flag, 2U);
      continue;
    }
    output[index] = __float2bfloat16_rn(value);
  }
}

__global__ void shared_expert_swiglu_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* output, std::uint32_t* error_flag,
    std::uint64_t element_count) {
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const auto gate_value = __bfloat162float(gate[index]);
    const auto up_value = __bfloat162float(up[index]);
    if (!isfinite(gate_value) || !isfinite(up_value)) {
      atomicOr(error_flag, 1U);
      continue;
    }
    const auto clamped_gate = fminf(
        gate_value, DeepSeekSharedExpertSwiGluLaunch::kLimit);
    const auto clamped_up = fminf(
        fmaxf(up_value, -DeepSeekSharedExpertSwiGluLaunch::kLimit),
        DeepSeekSharedExpertSwiGluLaunch::kLimit);
    const auto value =
        (clamped_gate / (1.0F + expf(-clamped_gate))) * clamped_up;
    if (!isfinite(value)) {
      atomicOr(error_flag, 2U);
      continue;
    }
    output[index] = __float2bfloat16_rn(value);
  }
}

}  // namespace

Status launch_deepseek_expert_swiglu(DeepSeekExpertSwiGluLaunch launch) {
  const auto valid = validate_deepseek_expert_swiglu_launch(launch);
  if (!valid.ok()) return valid;
  const auto clean = cuda_status(
      cudaPeekAtLastError(), "cudaPeekAtLastError before DeepSeek SwiGLU");
  if (!clean.ok()) return clean;
  const auto elements = static_cast<std::uint64_t>(launch.token_count) *
                        DeepSeekExpertSwiGluLaunch::kIntermediateSize;
  const auto blocks = static_cast<std::uint32_t>((elements + 255U) / 256U);
  expert_swiglu_kernel<<<
      blocks, 256, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.gate_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.up_bf16),
      reinterpret_cast<const float*>(launch.route_weights_f32),
      reinterpret_cast<__nv_bfloat16*>(launch.output_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag), elements);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek expert SwiGLU launch");
}

Status launch_deepseek_shared_expert_swiglu(
    DeepSeekSharedExpertSwiGluLaunch launch) {
  const auto valid = validate_deepseek_shared_expert_swiglu_launch(launch);
  if (!valid.ok()) return valid;
  const auto clean = cuda_status(
      cudaPeekAtLastError(),
      "cudaPeekAtLastError before DeepSeek shared expert SwiGLU");
  if (!clean.ok()) return clean;
  const auto elements = static_cast<std::uint64_t>(launch.token_count) *
                        DeepSeekSharedExpertSwiGluLaunch::kIntermediateSize;
  const auto blocks = static_cast<std::uint32_t>((elements + 255U) / 256U);
  shared_expert_swiglu_kernel<<<
      blocks, 256, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.gate_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.up_bf16),
      reinterpret_cast<__nv_bfloat16*>(launch.output_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag), elements);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek shared expert SwiGLU launch");
}

}  // namespace pih
