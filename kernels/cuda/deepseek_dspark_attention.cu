#include "pih/backend/cuda/deepseek_dspark_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

__device__ constexpr float kNegativeInfinity = -__builtin_huge_valf();

__global__ void dspark_attention_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* recent_kv,
    const __nv_bfloat16* draft_kv, const float* attention_sink,
    __nv_bfloat16* output, std::uint32_t* error_flag,
    std::uint32_t recent_count) {
  __shared__ float reduction[256];
  __shared__ float old_scale;
  __shared__ float row_weight;
  __shared__ float running_max;
  __shared__ float running_sum;

  const auto query_ordinal = blockIdx.x /
                             DeepSeekDsparkAttentionLaunch::kHeadCount;
  const auto head = blockIdx.x %
                    DeepSeekDsparkAttentionLaunch::kHeadCount;
  const auto first_column = threadIdx.x;
  const auto second_column = threadIdx.x + blockDim.x;
  const auto query_base =
      (static_cast<std::uint64_t>(query_ordinal) *
           DeepSeekDsparkAttentionLaunch::kHeadCount +
       head) * DeepSeekDsparkAttentionLaunch::kHeadDimension;
  float first_accumulator = 0.0F;
  float second_accumulator = 0.0F;
  if (threadIdx.x == 0) {
    running_max = kNegativeInfinity;
    running_sum = 0.0F;
  }
  __syncthreads();

  const auto row_count = recent_count +
                         DeepSeekDsparkAttentionLaunch::kBlockSize;
  for (std::uint32_t row = 0; row < row_count; ++row) {
    const auto* kv = row < recent_count
                         ? recent_kv + static_cast<std::uint64_t>(row) *
                                           DeepSeekDsparkAttentionLaunch::
                                               kHeadDimension
                         : draft_kv +
                               static_cast<std::uint64_t>(row - recent_count) *
                                   DeepSeekDsparkAttentionLaunch::
                                       kHeadDimension;
    reduction[threadIdx.x] =
        __bfloat162float(query[query_base + first_column]) *
            __bfloat162float(kv[first_column]) +
        __bfloat162float(query[query_base + second_column]) *
            __bfloat162float(kv[second_column]);
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride != 0; stride /= 2) {
      if (threadIdx.x < stride) {
        reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      auto score = reduction[0] * rsqrtf(
          static_cast<float>(DeepSeekDsparkAttentionLaunch::kHeadDimension));
      if (!isfinite(score)) {
        atomicOr(error_flag, 1U);
        score = kNegativeInfinity;
      }
      if (isinf(score) && score < 0.0F) {
        old_scale = 1.0F;
        row_weight = 0.0F;
      } else {
        const auto next_max = fmaxf(running_max, score);
        old_scale = isinf(running_max) ? 0.0F
                                       : expf(running_max - next_max);
        row_weight = expf(score - next_max);
        running_sum = running_sum * old_scale + row_weight;
        running_max = next_max;
      }
    }
    __syncthreads();
    first_accumulator = first_accumulator * old_scale +
                        row_weight * __bfloat162float(kv[first_column]);
    second_accumulator = second_accumulator * old_scale +
                         row_weight * __bfloat162float(kv[second_column]);
    __syncthreads();
  }

  const auto sink = attention_sink[head];
  if (!isfinite(sink) || !isfinite(running_max) ||
      !isfinite(running_sum) || running_sum <= 0.0F) {
    if (threadIdx.x == 0) atomicOr(error_flag, 2U);
    return;
  }
  const auto final_max = fmaxf(running_max, sink);
  const auto numerator_scale = expf(running_max - final_max);
  const auto denominator = running_sum * numerator_scale +
                           expf(sink - final_max);
  if (!isfinite(denominator) || denominator <= 0.0F) {
    if (threadIdx.x == 0) atomicOr(error_flag, 4U);
    return;
  }
  output[query_base + first_column] = __float2bfloat16_rn(
      first_accumulator * numerator_scale / denominator);
  output[query_base + second_column] = __float2bfloat16_rn(
      second_accumulator * numerator_scale / denominator);
}

__global__ void dspark_positions_kernel(
    std::uint32_t* draft_positions, std::uint32_t current_position) {
  if (threadIdx.x < DeepSeekDsparkAttentionLaunch::kBlockSize) {
    draft_positions[threadIdx.x] = current_position + threadIdx.x + 1U;
  }
}

}  // namespace

Status launch_deepseek_dspark_attention(
    DeepSeekDsparkAttentionLaunch launch) {
  auto status = validate_deepseek_dspark_attention_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(
      cudaPeekAtLastError(),
      "cudaPeekAtLastError before DeepSeek DSpark attention");
  if (!status.ok()) return status;
  constexpr auto threads = 256U;
  constexpr auto blocks = DeepSeekDsparkAttentionLaunch::kBlockSize *
                          DeepSeekDsparkAttentionLaunch::kHeadCount;
  dspark_attention_kernel<<<
      blocks, threads, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.query_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.recent_kv_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.draft_kv_bf16),
      reinterpret_cast<const float*>(launch.attention_sink_f32),
      reinterpret_cast<__nv_bfloat16*>(launch.output_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.recent_count);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek DSpark attention launch");
}

Status launch_deepseek_dspark_positions(
    DeepSeekDsparkPositionLaunch launch) {
  auto status = validate_deepseek_dspark_position_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(
      cudaPeekAtLastError(),
      "cudaPeekAtLastError before DeepSeek DSpark positions");
  if (!status.ok()) return status;
  dspark_positions_kernel<<<
      1, 32, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<std::uint32_t*>(launch.draft_positions_u32),
      launch.current_position);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek DSpark positions launch");
}

}  // namespace pih
