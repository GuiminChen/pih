#include "pih/backend/cuda/deepseek_compressor_pooling.h"

#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

__global__ void compressor_pooling_kernel(
    const float* kv_projection, const float* gate_projection,
    const float* ape_row, float* kv_state, float* score_state, float* output,
    std::uint32_t* error_flag, std::uint32_t ratio,
    std::uint32_t head_dim, std::uint32_t absolute_position) {
  const auto batch = blockIdx.x;
  const auto overlap = ratio == 4U;
  const auto projection_dim = (overlap ? 2U : 1U) * head_dim;
  const auto state_rows = (overlap ? 2U : 1U) * ratio;
  const auto remainder = absolute_position % ratio;
  const auto destination_row = (overlap ? ratio : 0U) + remainder;
  const auto projection_base =
      static_cast<std::uint64_t>(batch) * projection_dim;
  const auto state_base = static_cast<std::uint64_t>(batch) * state_rows *
                          projection_dim;
  for (std::uint32_t column = threadIdx.x; column < projection_dim;
       column += blockDim.x) {
    const auto kv = kv_projection[projection_base + column];
    const auto score = gate_projection[projection_base + column] +
                       ape_row[column];
    if (!isfinite(kv) || !isfinite(score)) atomicOr(error_flag, 1U);
    kv_state[state_base +
             static_cast<std::uint64_t>(destination_row) * projection_dim +
             column] = kv;
    score_state[state_base +
                static_cast<std::uint64_t>(destination_row) * projection_dim +
                column] = score;
  }
  __syncthreads();
  if (remainder + 1U != ratio) return;
  for (std::uint32_t column = threadIdx.x; column < head_dim;
       column += blockDim.x) {
    const auto candidates = overlap ? 2U * ratio : ratio;
    // The first group has no preceding overlap. Fixed-state banks are raw
    // device storage, so even multiplying an absent KV by a zero softmax
    // weight would read indeterminate memory (and can propagate NaNs).
    const auto first_candidate = overlap && absolute_position < ratio
        ? ratio : 0U;
    float maximum = -__builtin_huge_valf();
    for (std::uint32_t candidate = first_candidate;
         candidate < candidates; ++candidate) {
      const auto source_column =
          overlap && candidate >= ratio ? column + head_dim : column;
      maximum = fmaxf(
          maximum,
          score_state[state_base +
                      static_cast<std::uint64_t>(candidate) * projection_dim +
                      source_column]);
    }
    float numerator = 0.0F;
    float denominator = 0.0F;
    for (std::uint32_t candidate = first_candidate;
         candidate < candidates; ++candidate) {
      const auto source_column =
          overlap && candidate >= ratio ? column + head_dim : column;
      const auto offset =
          state_base + static_cast<std::uint64_t>(candidate) * projection_dim +
          source_column;
      const auto weight = expf(score_state[offset] - maximum);
      denominator += weight;
      numerator += weight * kv_state[offset];
    }
    const auto value = numerator / denominator;
    if (!isfinite(value)) atomicOr(error_flag, 2U);
    output[static_cast<std::uint64_t>(batch) * head_dim + column] = value;
  }
  __syncthreads();
  if (overlap) {
    const auto rows = static_cast<std::uint64_t>(ratio) * projection_dim;
    for (std::uint64_t index = threadIdx.x; index < rows;
         index += blockDim.x) {
      kv_state[state_base + index] = kv_state[state_base + rows + index];
      score_state[state_base + index] =
          score_state[state_base + rows + index];
    }
  }
}

}  // namespace

Status launch_deepseek_compressor_pooling(
    DeepSeekCompressorPoolingLaunch launch) {
  auto status = validate_deepseek_compressor_pooling_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(cudaPeekAtLastError(),
                       "cudaPeekAtLastError before DeepSeek compressor pooling");
  if (!status.ok()) return status;
  compressor_pooling_kernel<<<
      launch.batch_count, 256, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const float*>(launch.kv_projection_f32),
      reinterpret_cast<const float*>(launch.gate_projection_f32),
      reinterpret_cast<const float*>(launch.ape_row_f32),
      reinterpret_cast<float*>(launch.kv_state_f32),
      reinterpret_cast<float*>(launch.score_state_f32),
      reinterpret_cast<float*>(launch.output_f32),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32), launch.ratio,
      launch.head_dim, launch.absolute_position);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek compressor pooling launch");
}

}  // namespace pih
