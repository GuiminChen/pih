#include "pih/backend/cuda/deepseek_fp8_activation_quant.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

__global__ void fp8_activation_quant_kernel(
    const __nv_bfloat16* input, std::uint8_t* output,
    std::uint8_t* scale_bits, std::uint32_t* error_flag,
    std::uint32_t logical_k) {
  __shared__ float magnitudes[DeepSeekFp8ActivationQuantLaunch::kGroupSize];
  __shared__ float scale;
  __shared__ std::uint8_t encoded_scale;
  const auto group = blockIdx.x;
  const auto lane = threadIdx.x;
  const auto groups_per_token =
      logical_k / DeepSeekFp8ActivationQuantLaunch::kGroupSize;
  const auto token = group / groups_per_token;
  const auto k_group = group % groups_per_token;
  const auto index = static_cast<std::uint64_t>(token) * logical_k +
                     k_group * DeepSeekFp8ActivationQuantLaunch::kGroupSize +
                     lane;
  const auto value = __bfloat162float(input[index]);
  if (!isfinite(value)) atomicOr(error_flag, 1U);
  magnitudes[lane] = isfinite(value) ? fabsf(value) : 0.0F;
  __syncthreads();
  for (std::uint32_t stride =
           DeepSeekFp8ActivationQuantLaunch::kGroupSize / 2U;
       stride != 0; stride /= 2U) {
    if (lane < stride) {
      magnitudes[lane] = fmaxf(magnitudes[lane], magnitudes[lane + stride]);
    }
    __syncthreads();
  }
  if (lane == 0) {
    const auto amax = fmaxf(magnitudes[0], 1.0e-4F);
    const auto scaled = amax * (1.0F / 448.0F);
    const auto raw = __float_as_uint(scaled);
    const auto exponent = static_cast<int>((raw >> 23U) & 0xFFU) - 127;
    const auto mantissa = raw & 0x7FFFFFU;
    const auto scale_exponent = exponent + (mantissa != 0 ? 1 : 0);
    if (scale_exponent < -127 || scale_exponent > 127) {
      atomicOr(error_flag, 2U);
      scale = 1.0F;
      encoded_scale = 127U;
    } else {
      scale = ldexpf(1.0F, scale_exponent);
      encoded_scale = static_cast<std::uint8_t>(scale_exponent + 127);
    }
    scale_bits[group] = encoded_scale;
  }
  __syncthreads();
  const auto normalized = fminf(fmaxf(value / scale, -448.0F), 448.0F);
  output[index] = __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3);
}

}  // namespace

Status launch_deepseek_fp8_activation_quant(
    DeepSeekFp8ActivationQuantLaunch launch) {
  const auto valid = validate_deepseek_fp8_activation_quant_launch(launch);
  if (!valid.ok()) return valid;
  const auto clean = cuda_status(
      cudaPeekAtLastError(), "cudaPeekAtLastError before DeepSeek FP8 quant");
  if (!clean.ok()) return clean;
  const auto groups = static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(launch.token_count) *
      (launch.logical_k /
       DeepSeekFp8ActivationQuantLaunch::kGroupSize));
  fp8_activation_quant_kernel<<<
      groups, DeepSeekFp8ActivationQuantLaunch::kGroupSize, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.input_bf16),
      reinterpret_cast<std::uint8_t*>(launch.output_e4m3),
      reinterpret_cast<std::uint8_t*>(launch.scale_bits),
      reinterpret_cast<std::uint32_t*>(launch.error_flag), launch.logical_k);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek FP8 activation quant launch");
}

}  // namespace pih
