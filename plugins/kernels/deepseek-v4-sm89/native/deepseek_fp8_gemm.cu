#include "pih/backend/cuda/deepseek_fp8_gemm.h"

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

template <bool kOutputFp32>
__global__ void fp8_gemm_kernel(
    const std::uint8_t* activation,
    const std::uint8_t* activation_scales,
    const std::uint8_t* weight, const std::uint8_t* weight_scales,
    void* output, std::uint32_t* error_flag,
    std::uint32_t m, std::uint32_t n, std::uint32_t k) {
  const auto output_elements = static_cast<std::uint64_t>(m) * n;
  const auto k_blocks = k / 128U;
  for (std::uint64_t output_index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       output_index < output_elements;
       output_index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const auto row_m = static_cast<std::uint32_t>(output_index / n);
    const auto row_n = static_cast<std::uint32_t>(output_index % n);
    float accumulator = 0.0F;
    for (std::uint32_t block = 0; block < k_blocks; ++block) {
      const auto activation_scale =
          activation_scales[static_cast<std::uint64_t>(row_m) * k_blocks +
                            block];
      const auto weight_scale =
          weight_scales[static_cast<std::uint64_t>(row_n / 128U) * k_blocks +
                        block];
      if (activation_scale == 0xFFU || weight_scale == 0xFFU) {
        atomicOr(error_flag, 2U);
        continue;
      }
      float local = 0.0F;
      for (std::uint32_t offset = 0; offset < 128U; ++offset) {
        const auto column = block * 128U + offset;
        local += decode_e4m3(
                     activation[static_cast<std::uint64_t>(row_m) * k +
                                column],
                     error_flag) *
                 decode_e4m3(
                     weight[static_cast<std::uint64_t>(row_n) * k + column],
                     error_flag);
      }
      accumulator += local *
                     ldexpf(1.0F, static_cast<int>(activation_scale) - 127) *
                     ldexpf(1.0F, static_cast<int>(weight_scale) - 127);
    }
    if (!isfinite(accumulator)) {
      atomicOr(error_flag, 4U);
      continue;
    }
    if constexpr (kOutputFp32) {
      static_cast<float*>(output)[output_index] = accumulator;
    } else {
      static_cast<__nv_bfloat16*>(output)[output_index] =
          __float2bfloat16_rn(accumulator);
    }
  }
}

}  // namespace

Status launch_deepseek_fp8_gemm(DeepSeekFp8GemmLaunch launch) {
  const auto valid = validate_deepseek_fp8_gemm_launch(launch);
  if (!valid.ok()) return valid;
  const auto clean = cuda_status(
      cudaPeekAtLastError(), "cudaPeekAtLastError before DeepSeek FP8 GEMM");
  if (!clean.ok()) return clean;
  const auto elements = static_cast<std::uint64_t>(launch.m) * launch.n;
  const auto blocks = static_cast<std::uint32_t>((elements + 255U) / 256U);
  auto launch_kernel = [&]<bool kFp32>() {
    fp8_gemm_kernel<kFp32><<<
        blocks, 256, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
        reinterpret_cast<const std::uint8_t*>(launch.activation_e4m3),
        reinterpret_cast<const std::uint8_t*>(launch.activation_scale_bits),
        reinterpret_cast<const std::uint8_t*>(launch.weight_e4m3),
        reinterpret_cast<const std::uint8_t*>(launch.weight_scale_bits),
        reinterpret_cast<void*>(launch.output_bf16),
        reinterpret_cast<std::uint32_t*>(launch.error_flag), launch.m,
        launch.n, launch.k);
  };
  if (launch.output_type == DeepSeekFp8GemmOutputType::kFp32) {
    launch_kernel.template operator()<true>();
  } else {
    launch_kernel.template operator()<false>();
  }
  return cuda_status(cudaPeekAtLastError(), "DeepSeek FP8 GEMM launch");
}

}  // namespace pih
