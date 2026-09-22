#include "pih/backend/cuda/deepseek_compressor_bf16_projection.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih { namespace {

__global__ void compressor_bf16_projection_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* kv_weight, const __nv_bfloat16* gate_weight,
    float* kv_projection, float* gate_projection, std::uint32_t* error,
    std::uint32_t projection_dim, std::uint32_t hidden_size) {
  const auto token = blockIdx.x;
  const auto fused_column = blockIdx.y;
  const auto* weight = fused_column < projection_dim ? kv_weight : gate_weight;
  const auto weight_row = fused_column % projection_dim;
  float partial = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    partial += __bfloat162float(input[token * hidden_size + column]) *
               __bfloat162float(weight[weight_row * hidden_size + column]);
  }
  __shared__ float reduction[256];
  reduction[threadIdx.x] = partial;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x != 0) return;
  const auto value = reduction[0];
  if (!isfinite(value)) atomicOr(error, 1U);
  const auto output_column = fused_column % projection_dim;
  auto* output = fused_column < projection_dim ? kv_projection
                                                : gate_projection;
  output[static_cast<std::uint64_t>(token) * projection_dim + output_column] =
      value;
}

}  // namespace

Status launch_deepseek_compressor_bf16_projection(
    DeepSeekCompressorBf16ProjectionLaunch launch) {
  auto status = validate_deepseek_compressor_bf16_projection_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(
      cudaPeekAtLastError(),
      "cudaPeekAtLastError before DeepSeek BF16 compressor projection");
  if (!status.ok()) return status;
  const auto coefficient = launch.ratio == 4 ? 2U : 1U;
  const auto projection_dim = coefficient * launch.head_dim;
  compressor_bf16_projection_kernel<<<
      dim3(launch.token_count, 2U * projection_dim), 256, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.input_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.kv_weight_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.gate_weight_bf16),
      reinterpret_cast<float*>(launch.kv_projection_f32),
      reinterpret_cast<float*>(launch.gate_projection_f32),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32), projection_dim,
      launch.hidden_size);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek BF16 compressor projection launch");
}

}  // namespace pih
