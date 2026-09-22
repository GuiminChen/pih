#include "pih/backend/cuda/deepseek_fp4_gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

__device__ float decode_e4m3(std::uint8_t bits, std::uint32_t* error_flag) {
  const auto exponent = static_cast<std::uint8_t>((bits >> 3U) & 0x0FU);
  const auto mantissa = static_cast<std::uint8_t>(bits & 0x07U);
  if (exponent == 0x0FU && mantissa == 0x07U) {
    atomicOr(error_flag, 1U);
    return 0.0F;
  }
  float value = exponent == 0
                    ? ldexpf(static_cast<float>(mantissa), -9)
                    : ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F,
                             static_cast<int>(exponent) - 7);
  return (bits & 0x80U) != 0 ? -value : value;
}

__device__ float decode_e2m1(std::uint8_t bits) {
  constexpr float values[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                               2.0F, 3.0F, 4.0F, 6.0F};
  const auto value = values[bits & 0x07U];
  return (bits & 0x08U) != 0 ? -value : value;
}

__global__ void fp4_gemm_kernel(
    const std::uint8_t* activation,
    const std::uint8_t* activation_scales,
    const std::uint8_t* packed_weight,
    const std::uint8_t* weight_scales, __nv_bfloat16* output,
    std::uint32_t* error_flag, std::uint32_t m, std::uint32_t n,
    std::uint32_t k) {
  const auto output_elements = static_cast<std::uint64_t>(m) * n;
  for (std::uint64_t output_index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       output_index < output_elements;
       output_index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const auto row_m = static_cast<std::uint32_t>(output_index / n);
    const auto row_n = static_cast<std::uint32_t>(output_index % n);
    float accumulator = 0.0F;
    for (std::uint32_t block_k = 0; block_k < k / 32U; ++block_k) {
      const auto activation_scale_bits =
          activation_scales[static_cast<std::uint64_t>(row_m) * (k / 128U) +
                            block_k / 4U];
      const auto weight_scale_bits =
          weight_scales[static_cast<std::uint64_t>(row_n) * (k / 32U) +
                        block_k];
      if (activation_scale_bits == 0xFFU || weight_scale_bits == 0xFFU) {
        atomicOr(error_flag, 2U);
        continue;
      }
      float local = 0.0F;
      for (std::uint32_t offset = 0; offset < 32U; ++offset) {
        const auto column = block_k * 32U + offset;
        const auto a = decode_e4m3(
            activation[static_cast<std::uint64_t>(row_m) * k + column],
            error_flag);
        const auto packed =
            packed_weight[static_cast<std::uint64_t>(row_n) * (k / 2U) +
                          column / 2U];
        const auto nibble = static_cast<std::uint8_t>(
            (column & 1U) == 0 ? packed & 0x0FU : packed >> 4U);
        local += a * decode_e2m1(nibble);
      }
      accumulator +=
          local * ldexpf(1.0F,
                         static_cast<int>(activation_scale_bits) - 127) *
          ldexpf(1.0F, static_cast<int>(weight_scale_bits) - 127);
    }
    if (!isfinite(accumulator)) {
      atomicOr(error_flag, 4U);
      continue;
    }
    output[output_index] = __float2bfloat16_rn(accumulator);
  }
}

}  // namespace

Status launch_deepseek_fp4_gemm(DeepSeekFp4GemmLaunch launch) {
  const auto valid = validate_deepseek_fp4_gemm_launch(launch);
  if (!valid.ok()) return valid;
  const auto clean = cuda_status(
      cudaPeekAtLastError(), "cudaPeekAtLastError before DeepSeek FP4 GEMM");
  if (!clean.ok()) return clean;
  const auto elements = static_cast<std::uint64_t>(launch.m) * launch.n;
  const auto blocks = static_cast<std::uint32_t>((elements + 255U) / 256U);
  fp4_gemm_kernel<<<
      blocks, 256, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const std::uint8_t*>(launch.activation_e4m3),
      reinterpret_cast<const std::uint8_t*>(launch.activation_scale_bits),
      reinterpret_cast<const std::uint8_t*>(launch.packed_weight),
      reinterpret_cast<const std::uint8_t*>(launch.weight_scale_bits),
      reinterpret_cast<__nv_bfloat16*>(launch.output_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag), launch.m, launch.n,
      launch.k);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek FP4 GEMM launch");
}

}  // namespace pih
