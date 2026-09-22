#include "pih/backend/cuda/deepseek_expert_accumulate.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

constexpr std::uint32_t kHiddenSize = 4096;

__global__ void ordered_expert_accumulate_kernel(
    const __nv_bfloat16* expert_output, const std::uint32_t* token_indices,
    float* accumulator, std::uint32_t* error_flag,
    std::uint32_t route_count, std::uint32_t token_count) {
  for (std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
       column < kHiddenSize; column += blockDim.x * gridDim.x) {
    std::uint32_t previous_token = 0;
    for (std::uint32_t route = 0; route < route_count; ++route) {
      const auto token = token_indices[route];
      if (token >= token_count || (route != 0 && token <= previous_token)) {
        atomicOr(error_flag, 1U);
        return;
      }
      previous_token = token;
      const auto source = static_cast<std::uint64_t>(route) * kHiddenSize + column;
      const auto destination =
          static_cast<std::uint64_t>(token) * kHiddenSize + column;
      const auto value = __bfloat162float(expert_output[source]);
      const auto next = accumulator[destination] + value;
      if (!isfinite(value) || !isfinite(next)) {
        atomicOr(error_flag, 2U);
        return;
      }
      accumulator[destination] = next;
    }
  }
}

__global__ void expert_finalize_kernel(
    const float* accumulator, const __nv_bfloat16* shared_output,
    __nv_bfloat16* output, std::uint32_t* error_flag,
    std::uint64_t element_count) {
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count; index +=
           static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const auto value = accumulator[index] + __bfloat162float(shared_output[index]);
    if (!isfinite(value)) {
      atomicOr(error_flag, 4U);
      continue;
    }
    output[index] = __float2bfloat16_rn(value);
  }
}

}  // namespace

Status launch_deepseek_ordered_expert_accumulate(
    DeepSeekExpertAccumulateLaunch launch) {
  if (launch.expert_output_bf16 == 0 || launch.token_indices == 0 ||
      launch.accumulator_f32 == 0 || launch.error_flag == 0 ||
      launch.stream == 0 ||
      launch.route_count == 0 || launch.token_count == 0 ||
      launch.route_count > launch.token_count) {
    return Status::InvalidArgument(
        "DeepSeek ordered expert accumulate launch is invalid");
  }
  const auto clean = cuda_status(
      cudaPeekAtLastError(), "cudaPeekAtLastError before expert accumulate");
  if (!clean.ok()) return clean;
  ordered_expert_accumulate_kernel<<<
      16, 256, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.expert_output_bf16),
      reinterpret_cast<const std::uint32_t*>(launch.token_indices),
      reinterpret_cast<float*>(launch.accumulator_f32),
      reinterpret_cast<std::uint32_t*>(launch.error_flag), launch.route_count,
      launch.token_count);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek ordered expert accumulate launch");
}

Status launch_deepseek_expert_finalize(DeepSeekExpertFinalizeLaunch launch) {
  if (launch.accumulator_f32 == 0 || launch.shared_output_bf16 == 0 ||
      launch.output_bf16 == 0 || launch.error_flag == 0 ||
      launch.stream == 0 || launch.token_count == 0) {
    return Status::InvalidArgument("DeepSeek expert finalize launch is invalid");
  }
  const auto clean = cuda_status(
      cudaPeekAtLastError(), "cudaPeekAtLastError before expert finalize");
  if (!clean.ok()) return clean;
  const auto elements =
      static_cast<std::uint64_t>(launch.token_count) * kHiddenSize;
  const auto blocks = static_cast<std::uint32_t>((elements + 255U) / 256U);
  expert_finalize_kernel<<<
      blocks, 256, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const float*>(launch.accumulator_f32),
      reinterpret_cast<const __nv_bfloat16*>(launch.shared_output_bf16),
      reinterpret_cast<__nv_bfloat16*>(launch.output_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag), elements);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek expert finalize launch");
}

}  // namespace pih
