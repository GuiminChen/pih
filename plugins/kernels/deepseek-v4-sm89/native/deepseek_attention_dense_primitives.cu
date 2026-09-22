#include "pih/backend/cuda/deepseek_attention_dense_primitives.h"
#include "pih/backend/cuda/deepseek_rope_table.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"
#include "pih/backend/cuda/deepseek_fp8_activation_quant.h"

namespace pih { namespace {

__global__ void rope_table_kernel(
    float* output, std::uint32_t* error, std::uint32_t position_count,
    bool yarn) {
  const auto index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  constexpr std::uint32_t kPairs = 32;
  if (index >= position_count * kPairs) return;
  const auto position = index / kPairs;
  const auto pair = index % kPairs;
  const auto positional_frequency =
      pow(10000.0, static_cast<double>(pair * 2U) / 64.0);
  double inverse_frequency = 1.0 / positional_frequency;
  double magnitude = 1.0;
  if (yarn) {
    constexpr double kPi = 3.14159265358979323846264338327950288;
    const auto correction = [](double rotations) {
      return 64.0 * log(65536.0 / (rotations * 2.0 * kPi)) /
             (2.0 * log(10000.0));
    };
    const auto low = floor(correction(32.0));
    const auto high = ceil(correction(1.0));
    const auto ramp = fmin(fmax((static_cast<double>(pair) - low) /
                                (high - low), 0.0), 1.0);
    const auto extrapolation_mask = 1.0 - ramp;
    const auto interpolated = inverse_frequency / 16.0;
    inverse_frequency = interpolated * (1.0 - extrapolation_mask) +
                        inverse_frequency * extrapolation_mask;
    magnitude = 0.1 * log(16.0) + 1.0;
  }
  const auto angle = static_cast<double>(position) * inverse_frequency;
  const auto cosine = static_cast<float>(cos(angle) * magnitude);
  const auto sine = static_cast<float>(sin(angle) * magnitude);
  if (!isfinite(cosine) || !isfinite(sine)) atomicOr(error, 1U);
  const auto row = static_cast<std::uint64_t>(position) * 64U;
  output[row + pair] = cosine;
  output[row + kPairs + pair] = sine;
}

__global__ void head_rms_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    std::uint32_t* error, std::uint32_t head_dimension) {
  __shared__ float scratch[256];
  const auto offset = static_cast<std::uint64_t>(blockIdx.x) * head_dimension;
  float sum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < head_dimension;
       column += blockDim.x) {
    const auto value = __bfloat162float(input[offset + column]);
    if (!isfinite(value)) atomicOr(error, 1U);
    sum += value * value;
  }
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (std::uint32_t width = blockDim.x / 2; width != 0; width >>= 1U) {
    if (threadIdx.x < width) scratch[threadIdx.x] += scratch[threadIdx.x + width];
    __syncthreads();
  }
  const auto inverse = rsqrtf(scratch[0] / head_dimension + 1.0e-6F);
  if (!isfinite(inverse)) atomicOr(error, 2U);
  for (std::uint32_t column = threadIdx.x; column < head_dimension;
       column += blockDim.x) {
    const auto value = __bfloat162float(input[offset + column]) * inverse;
    if (!isfinite(value)) atomicOr(error, 4U);
    output[offset + column] = __float2bfloat16_rn(value);
  }
}

__global__ void rotary_kernel(
    __nv_bfloat16* input, const float* frequencies,
    const std::uint32_t* positions, std::uint32_t* error,
    std::uint32_t token_count, std::uint32_t head_count, std::uint32_t head_dimension,
    std::uint32_t rope_dimension, std::uint32_t table_position_count,
    bool inverse) {
  const auto pair = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  const auto pairs_per_token = head_count * rope_dimension / 2U;
  const auto total_pairs = token_count * pairs_per_token;
  if (pair >= total_pairs) return;
  const auto token = pair / pairs_per_token;
  const auto within_token = pair % pairs_per_token;
  const auto head = within_token / (rope_dimension / 2U);
  const auto rope_pair = within_token % (rope_dimension / 2U);
  const auto offset = (static_cast<std::uint64_t>(token) * head_count + head) *
      head_dimension + (head_dimension - rope_dimension) + rope_pair * 2U;
  const auto position = positions[token];
  if (position >= table_position_count) {
    atomicOr(error, 32U);
    return;
  }
  const auto frequency_base =
      static_cast<std::uint64_t>(position) * rope_dimension;
  const auto cosine = frequencies[frequency_base + rope_pair];
  auto sine = frequencies[frequency_base + rope_dimension / 2U + rope_pair];
  if (inverse) sine = -sine;
  const auto real = __bfloat162float(input[offset]);
  const auto imag = __bfloat162float(input[offset + 1]);
  if (!isfinite(real) || !isfinite(imag) || !isfinite(cosine) ||
      !isfinite(sine)) atomicOr(error, 8U);
  const auto rotated_real = real * cosine - imag * sine;
  const auto rotated_imag = real * sine + imag * cosine;
  if (!isfinite(rotated_real) || !isfinite(rotated_imag)) atomicOr(error, 16U);
  input[offset] = __float2bfloat16_rn(rotated_real);
  input[offset + 1] = __float2bfloat16_rn(rotated_imag);
}

__device__ float decode_e4m3(std::uint8_t bits) {
  const auto exponent = static_cast<std::uint8_t>((bits >> 3U) & 0x0FU);
  const auto mantissa = static_cast<std::uint8_t>(bits & 0x07U);
  float value = exponent == 0
      ? ldexpf(static_cast<float>(mantissa), -9)
      : ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F,
               static_cast<int>(exponent) - 7);
  return (bits & 0x80U) != 0 ? -value : value;
}

__global__ void kv_fp8_simulate_kernel(
    __nv_bfloat16* kv, std::uint32_t* error,
    std::uint32_t vector_dimension, std::uint32_t quantized_dimension) {
  __shared__ float magnitudes[64];
  __shared__ float scale;
  const auto group = blockIdx.x;
  const auto groups_per_token = quantized_dimension / 64U;
  const auto token = group / groups_per_token;
  const auto within = group % groups_per_token;
  const auto index = static_cast<std::uint64_t>(token) * vector_dimension +
                     within * 64U + threadIdx.x;
  const auto input = __bfloat162float(kv[index]);
  if (!isfinite(input)) atomicOr(error, 32U);
  magnitudes[threadIdx.x] = isfinite(input) ? fabsf(input) : 0.0F;
  __syncthreads();
  for (std::uint32_t width = 32; width != 0; width >>= 1U) {
    if (threadIdx.x < width)
      magnitudes[threadIdx.x] =
          fmaxf(magnitudes[threadIdx.x], magnitudes[threadIdx.x + width]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const auto amax = fmaxf(magnitudes[0], 1.0e-4F);
    const auto scaled = amax / 448.0F;
    const auto raw = __float_as_uint(scaled);
    const auto exponent = static_cast<int>((raw >> 23U) & 0xFFU) - 127;
    const auto scale_exponent = exponent + ((raw & 0x7FFFFFU) != 0 ? 1 : 0);
    if (scale_exponent < -127 || scale_exponent > 127) {
      atomicOr(error, 64U);
      scale = 1.0F;
    } else {
      scale = ldexpf(1.0F, scale_exponent);
    }
  }
  __syncthreads();
  const auto normalized = fminf(fmaxf(input / scale, -448.0F), 448.0F);
  const auto encoded = __nv_cvt_float_to_fp8(
      isfinite(normalized) ? normalized : 0.0F, __NV_SATFINITE, __NV_E4M3);
  const auto output = decode_e4m3(encoded) * scale;
  if (!isfinite(output)) atomicOr(error, 128U);
  kv[index] = __float2bfloat16_rn(output);
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
__global__ void grouped_bf16_gemm_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, std::uint32_t* error,
    std::uint32_t group_count, std::uint32_t outputs,
    std::uint32_t inputs) {
  __shared__ float scratch[256];
  const auto output_index = blockIdx.x;
  const auto row = output_index % outputs;
  const auto group_token = output_index / outputs;
  const auto group = group_token % group_count;
  const auto input_base = static_cast<std::uint64_t>(group_token) * inputs;
  const auto weight_base =
      (static_cast<std::uint64_t>(group) * outputs + row) * inputs;
  float partial = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < inputs;
       column += blockDim.x) {
    const auto x = __bfloat162float(input[input_base + column]);
    const auto w = __bfloat162float(weight[weight_base + column]);
    if (!isfinite(x) || !isfinite(w)) atomicOr(error, 256U);
    partial += x * w;
  }
  scratch[threadIdx.x] = partial;
  __syncthreads();
  for (std::uint32_t width = blockDim.x / 2; width != 0; width >>= 1U) {
    if (threadIdx.x < width) scratch[threadIdx.x] += scratch[threadIdx.x + width];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (!isfinite(scratch[0])) atomicOr(error, 512U);
    output[output_index] = __float2bfloat16_rn(scratch[0]);
  }
}
#endif

__global__ void grouped_fp8_gemm_kernel(
    const std::uint8_t* activation, const std::uint8_t* activation_scales,
    const std::uint8_t* weight, const std::uint8_t* weight_scales,
    __nv_bfloat16* output, std::uint32_t* error,
    std::uint32_t group_count, std::uint32_t outputs,
    std::uint32_t inputs) {
  __shared__ float scratch[256];
  const auto output_index = blockIdx.x;
  const auto row = output_index % outputs;
  const auto group_token = output_index / outputs;
  const auto group = group_token % group_count;
  const auto global_weight_row = group * outputs + row;
  const auto k_blocks = inputs / 128U;
  float partial = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < inputs;
       column += blockDim.x) {
    const auto block = column / 128U;
    const auto activation_scale =
        activation_scales[static_cast<std::uint64_t>(group_token) * k_blocks +
                          block];
    const auto weight_scale =
        weight_scales[static_cast<std::uint64_t>(global_weight_row / 128U) *
                          k_blocks + block];
    if (activation_scale == 0xFFU || weight_scale == 0xFFU) {
      atomicOr(error, 1024U);
      continue;
    }
    partial += decode_e4m3(
                   activation[static_cast<std::uint64_t>(group_token) *
                                  inputs + column]) *
               decode_e4m3(
                   weight[static_cast<std::uint64_t>(global_weight_row) *
                              inputs + column]) *
               ldexpf(1.0F, static_cast<int>(activation_scale) - 127) *
               ldexpf(1.0F, static_cast<int>(weight_scale) - 127);
  }
  scratch[threadIdx.x] = partial;
  __syncthreads();
  for (std::uint32_t width = blockDim.x / 2; width != 0; width >>= 1U) {
    if (threadIdx.x < width) scratch[threadIdx.x] += scratch[threadIdx.x + width];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (!isfinite(scratch[0])) atomicOr(error, 2048U);
    output[output_index] = __float2bfloat16_rn(scratch[0]);
  }
}

}  // namespace pih::<anonymous>

Status launch_deepseek_rope_table(DeepSeekRopeTableLaunch launch) {
  auto status = validate_deepseek_rope_table_launch(launch);
  if (!status.ok()) return status;
  auto* stream = reinterpret_cast<cudaStream_t>(launch.stream);
  status = cuda_status(cudaMemsetAsync(
      reinterpret_cast<void*>(launch.error_flag_u32), 0,
      sizeof(std::uint32_t), stream), "DeepSeek RoPE error clear");
  if (!status.ok()) return status;
  const auto values = launch.position_count * 32U;
  rope_table_kernel<<<(values + 255U) / 256U, 256, 0, stream>>>(
      reinterpret_cast<float*>(launch.output_f32),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.position_count, launch.yarn);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek RoPE table launch");
}

Status launch_deepseek_head_rms(DeepSeekHeadRmsLaunch launch) {
  auto status = validate_deepseek_head_rms_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(cudaPeekAtLastError(),
                       "cudaPeekAtLastError before DeepSeek head RMS");
  if (!status.ok()) return status;
  head_rms_kernel<<<launch.token_count * launch.head_count, 256, 0,
                    reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.input_bf16),
      reinterpret_cast<__nv_bfloat16*>(launch.output_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.head_dimension);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek head RMS launch");
}

Status launch_deepseek_rotary(DeepSeekRotaryLaunch launch) {
  auto status = validate_deepseek_rotary_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(cudaPeekAtLastError(),
                       "cudaPeekAtLastError before DeepSeek rotary");
  if (!status.ok()) return status;
  const auto pairs = launch.token_count * launch.head_count *
                     launch.rope_dimension / 2U;
  const auto blocks = (pairs + 255U) / 256U;
  rotary_kernel<<<blocks, 256, 0,
                  reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<__nv_bfloat16*>(launch.input_bf16),
      reinterpret_cast<const float*>(launch.frequencies_f32),
      reinterpret_cast<const std::uint32_t*>(launch.positions_u32),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.token_count, launch.head_count, launch.head_dimension, launch.rope_dimension,
      launch.table_position_count, launch.inverse);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek rotary launch");
}

Status launch_deepseek_kv_fp8_simulate(DeepSeekKvFp8SimulateLaunch launch) {
  auto status = validate_deepseek_kv_fp8_simulate_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(
      cudaPeekAtLastError(),
      "cudaPeekAtLastError before DeepSeek KV FP8 simulation");
  if (!status.ok()) return status;
  const auto groups = launch.token_count *
                      (launch.quantized_dimension / launch.group_size);
  kv_fp8_simulate_kernel<<<groups, 64, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<__nv_bfloat16*>(launch.kv_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.vector_dimension, launch.quantized_dimension);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek KV FP8 simulation launch");
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Status launch_deepseek_grouped_bf16_gemm(
    DeepSeekGroupedBf16GemmLaunch launch) {
  auto status = validate_deepseek_grouped_bf16_gemm_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(
      cudaPeekAtLastError(),
      "cudaPeekAtLastError before DeepSeek grouped BF16 GEMM");
  if (!status.ok()) return status;
  const auto blocks = launch.token_count * launch.group_count *
                      launch.output_per_group;
  grouped_bf16_gemm_kernel<<<blocks, 256, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.input_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.weight_bf16),
      reinterpret_cast<__nv_bfloat16*>(launch.output_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.group_count, launch.output_per_group, launch.input_per_group);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek grouped BF16 GEMM launch");
}
#endif

Status launch_deepseek_grouped_fp8_gemm(
    DeepSeekGroupedFp8GemmLaunch launch) {
  auto status = validate_deepseek_grouped_fp8_gemm_launch(launch);
  if (!status.ok()) return status;
  status = launch_deepseek_fp8_activation_quant({
      launch.input_bf16, launch.activation_e4m3,
      launch.activation_scale_bits, launch.error_flag_u32, launch.stream,
      launch.token_count * launch.group_count, launch.input_per_group});
  if (!status.ok()) return status;
  const auto blocks = launch.token_count * launch.group_count *
                      launch.output_per_group;
  grouped_fp8_gemm_kernel<<<blocks, 256, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const std::uint8_t*>(launch.activation_e4m3),
      reinterpret_cast<const std::uint8_t*>(launch.activation_scale_bits),
      reinterpret_cast<const std::uint8_t*>(launch.weight_e4m3),
      reinterpret_cast<const std::uint8_t*>(launch.weight_scale_bits),
      reinterpret_cast<__nv_bfloat16*>(launch.output_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.group_count, launch.output_per_group, launch.input_per_group);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek grouped FP8 GEMM launch");
}

}  // namespace pih
