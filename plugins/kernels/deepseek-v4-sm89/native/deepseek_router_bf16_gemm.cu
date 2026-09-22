#include "pih/backend/cuda/deepseek_router_bf16_gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

__global__ void router_bf16_gemm_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, float* scores,
    std::uint32_t* error, std::uint32_t expert_count,
    std::uint32_t hidden_size) {
  const auto row = blockIdx.x;
  const auto expert = blockIdx.y;
  float partial = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    partial += __bfloat162float(input[row * hidden_size + column]) *
               __bfloat162float(weight[expert * hidden_size + column]);
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
  if (threadIdx.x == 0) {
    const auto value = reduction[0];
    if (!isfinite(value)) atomicOr(error, 1U);
    scores[row * expert_count + expert] = value;
  }
}

}  // namespace

Status launch_deepseek_router_bf16_gemm(
    DeepSeekRouterBf16GemmLaunch launch) {
  auto status = validate_deepseek_router_bf16_gemm_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(
      cudaPeekAtLastError(),
      "cudaPeekAtLastError before DeepSeek BF16 router GEMM");
  if (!status.ok()) return status;
  router_bf16_gemm_kernel<<<dim3(launch.token_count, launch.expert_count),
                            256, 0,
                            reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.input_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.weight_bf16),
      reinterpret_cast<float*>(launch.scores_f32),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.expert_count, launch.hidden_size);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek BF16 router GEMM launch");
}

}  // namespace pih
