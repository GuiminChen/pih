#include "pih/backend/cuda/deepseek_compressor_bf16_store.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

__global__ void compressor_bf16_store_kernel(
    const float* compressed, const __nv_bfloat16* rms_weight,
    const float* cos_sin_cache, __nv_bfloat16* destination,
    std::uint32_t* error_flag, std::uint32_t head_dim,
    std::uint32_t rope_head_dim, std::uint32_t rope_position,
    float epsilon) {
  __shared__ float reductions[256];
  float sum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < head_dim;
       column += blockDim.x) {
    const auto value = compressed[column];
    sum += value * value;
  }
  reductions[threadIdx.x] = sum;
  __syncthreads();
  for (std::uint32_t width = blockDim.x / 2; width != 0; width /= 2) {
    if (threadIdx.x < width) {
      reductions[threadIdx.x] += reductions[threadIdx.x + width];
    }
    __syncthreads();
  }
  const auto inverse = rsqrtf(reductions[0] / head_dim + epsilon);
  const auto rope_begin = head_dim - rope_head_dim;
  const auto* rope = cos_sin_cache +
                     static_cast<std::uint64_t>(rope_position) * rope_head_dim;
  for (std::uint32_t column = threadIdx.x; column < head_dim;
       column += blockDim.x) {
    float value = compressed[column] * inverse *
                  __bfloat162float(rms_weight[column]);
    if (column >= rope_begin) {
      const auto local = column - rope_begin;
      const auto pair = local / 2U;
      const auto even = rope_begin + pair * 2U;
      const auto x0 = compressed[even] * inverse *
                      __bfloat162float(rms_weight[even]);
      const auto x1 = compressed[even + 1U] * inverse *
                      __bfloat162float(rms_weight[even + 1U]);
      const auto cosine = rope[pair];
      const auto sine = rope[rope_head_dim / 2U + pair];
      value = (local & 1U) == 0U ? x0 * cosine - x1 * sine
                                 : x1 * cosine + x0 * sine;
    }
    if (!isfinite(value)) atomicOr(error_flag, 4U);
    destination[column] = __float2bfloat16_rn(value);
  }
}

}  // namespace

Status launch_deepseek_compressor_bf16_store(
    DeepSeekCompressorBf16StoreLaunch launch) {
  auto status = validate_deepseek_compressor_bf16_store_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(
      cudaPeekAtLastError(),
      "cudaPeekAtLastError before DeepSeek compressor BF16 store");
  if (!status.ok()) return status;
  compressor_bf16_store_kernel<<<
      1, 256, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const float*>(launch.compressed_f32),
      reinterpret_cast<const __nv_bfloat16*>(launch.rms_weight_bf16),
      reinterpret_cast<const float*>(launch.cos_sin_cache_f32),
      reinterpret_cast<__nv_bfloat16*>(launch.destination_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32), launch.head_dim,
      launch.rope_head_dim, launch.rope_position, launch.rms_epsilon);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek compressor BF16 store launch");
}

}  // namespace pih
