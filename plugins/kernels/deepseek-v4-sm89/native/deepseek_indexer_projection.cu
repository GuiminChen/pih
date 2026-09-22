#include "pih/backend/cuda/deepseek_indexer_projection.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"
#include "pih/backend/cuda/deepseek_fp8_activation_quant.h"
#include "pih/backend/cuda/deepseek_fp8_gemm.h"

namespace pih { namespace {

__global__ void q_rope(__nv_bfloat16* query, const float* frequencies,
                       const std::uint32_t* positions,
                       std::uint32_t* error,
                       std::uint32_t table_position_count,
                       std::uint32_t elements) {
  const auto ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= elements) return;
  const auto token = ordinal / (64U * 32U);
  const auto remainder = ordinal % (64U * 32U);
  const auto head = remainder / 32U;
  const auto pair = remainder % 32U;
  const auto position = positions[token];
  if (position >= table_position_count) { atomicOr(error, 2U); return; }
  const auto base = (static_cast<std::uint64_t>(token) * 64U + head) * 128U;
  const auto even = base + 64U + pair * 2U;
  const auto odd = even + 1U;
  const float x = __bfloat162float(query[even]);
  const float y = __bfloat162float(query[odd]);
  const auto frequency = static_cast<std::uint64_t>(position) * 64U;
  const float cosine = frequencies[frequency + pair];
  const float sine = frequencies[frequency + pair + 32U];
  const float rotated_even = x * cosine - y * sine;
  const float rotated_odd = y * cosine + x * sine;
  if (!isfinite(rotated_even) || !isfinite(rotated_odd)) {
    atomicOr(error, 4U);
    query[even] = __float2bfloat16_rn(0.0F);
    query[odd] = __float2bfloat16_rn(0.0F);
    return;
  }
  query[even] = __float2bfloat16_rn(rotated_even);
  query[odd] = __float2bfloat16_rn(rotated_odd);
}

__global__ void weight_gemm(const __nv_bfloat16* hidden,
                            const __nv_bfloat16* weight, float* output,
                            std::uint32_t* error, std::uint32_t elements) {
  const auto ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= elements) return;
  const auto token = ordinal / 64U;
  const auto head = ordinal % 64U;
  float sum = 0.0F;
  for (std::uint32_t inner = 0; inner < 4096U; ++inner) {
    sum += __bfloat162float(hidden[static_cast<std::uint64_t>(token) * 4096U + inner]) *
           __bfloat162float(weight[static_cast<std::uint64_t>(head) * 4096U + inner]);
  }
  constexpr float scale = 0.011048543456039806F;  // 1 / sqrt(128 * 64)
  sum *= scale;
  if (!isfinite(sum)) { atomicOr(error, 8U); sum = 0.0F; }
  output[ordinal] = sum;
}

}  // namespace

Status launch_deepseek_indexer_projection(
    DeepSeekIndexerProjectionLaunch launch) {
  auto status = validate_deepseek_indexer_projection_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(cudaPeekAtLastError(),
                       "cudaPeekAtLastError before DeepSeek indexer projection");
  if (!status.ok()) return status;
  constexpr std::uint32_t threads = 256;
  const auto rope_elements = launch.token_count * 64U * 32U;
  const auto weight_elements = launch.token_count * 64U;
  const auto stream = reinterpret_cast<cudaStream_t>(launch.stream);
  status = launch_deepseek_fp8_activation_quant(
      {launch.qr_bf16, launch.qr_e4m3, launch.qr_scale_bits,
       launch.error_flag_u32, launch.stream, launch.token_count, 1024});
  if (!status.ok()) return status;
  status = launch_deepseek_fp8_gemm(
      {launch.qr_e4m3, launch.qr_scale_bits, launch.wq_b_e4m3,
       launch.wq_b_scale_bits, launch.query_bf16, launch.error_flag_u32,
       launch.stream, launch.token_count, 8192, 1024});
  if (!status.ok()) return status;
  q_rope<<<(rope_elements + threads - 1U) / threads, threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(launch.query_bf16),
      reinterpret_cast<const float*>(launch.frequencies_f32),
      reinterpret_cast<const std::uint32_t*>(launch.positions_u32),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.table_position_count, rope_elements);
  status = cuda_status(cudaPeekAtLastError(), "DeepSeek indexer RoPE launch");
  if (!status.ok()) return status;
  weight_gemm<<<(weight_elements + threads - 1U) / threads, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.hidden_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.weights_proj_bf16),
      reinterpret_cast<float*>(launch.head_weight_f32),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32), weight_elements);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek indexer projection launch");
}

}  // namespace pih
