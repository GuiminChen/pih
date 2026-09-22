#include "pih/backend/cuda/deepseek_mhc.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cfloat>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

constexpr std::uint32_t kStreams = 4;
constexpr std::uint32_t kMixRows = 24;

__device__ float reduce_block(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (std::uint32_t width = blockDim.x / 2; width != 0; width >>= 1U) {
    if (threadIdx.x < width) scratch[threadIdx.x] += scratch[threadIdx.x + width];
    __syncthreads();
  }
  const auto result = scratch[0];
  __syncthreads();
  return result;
}

__device__ float stable_sigmoid(float value) {
  if (value >= 0.0F) {
    const auto e = expf(-value);
    return 1.0F / (1.0F + e);
  }
  const auto e = expf(value);
  return e / (1.0F + e);
}

__global__ void mhc_pre_kernel(
    const __nv_bfloat16* residual, const float* fn, const float* scale,
    const float* base, const __nv_bfloat16* norm_weight, float* post_mix,
    float* residual_mix,
    __nv_bfloat16* layer_input, std::uint32_t* error_flag,
    std::uint32_t hidden_size, float rms_epsilon, float pre_epsilon,
    float sinkhorn_epsilon, float post_multiplier,
    std::uint32_t sinkhorn_iterations) {
  __shared__ float scratch[256];
  __shared__ float mixes[kMixRows];
  __shared__ float pre[kStreams];
  const auto token = blockIdx.x;
  const auto width = kStreams * hidden_size;
  const auto residual_base = static_cast<std::uint64_t>(token) * width;
  float square_sum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < width;
       column += blockDim.x) {
    const auto value = __bfloat162float(residual[residual_base + column]);
    if (!isfinite(value)) atomicOr(error_flag, 1U);
    square_sum += value * value;
  }
  const auto total = reduce_block(square_sum, scratch);
  if (threadIdx.x == 0) {
    mixes[0] = rsqrtf(total / static_cast<float>(width) + rms_epsilon);
    if (!isfinite(mixes[0])) atomicOr(error_flag, 2U);
  }
  __syncthreads();
  const auto inverse_rms = mixes[0];
  for (std::uint32_t row = 0; row < kMixRows; ++row) {
    float partial = 0.0F;
    const auto fn_base = static_cast<std::uint64_t>(row) * width;
    for (std::uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
      const auto weight = fn[fn_base + column];
      if (!isfinite(weight)) atomicOr(error_flag, 4U);
      partial += __bfloat162float(residual[residual_base + column]) * weight;
    }
    const auto value = reduce_block(partial, scratch);
    if (threadIdx.x == 0) mixes[row] = value * inverse_rms;
    __syncthreads();
  }
  const auto post_base = static_cast<std::uint64_t>(token) * kStreams;
  const auto comb_base = static_cast<std::uint64_t>(token) * kStreams * kStreams;
  if (threadIdx.x == 0) {
    for (std::uint32_t index = 0; index < 3; ++index) {
      if (!isfinite(scale[index])) atomicOr(error_flag, 32U);
    }
    for (std::uint32_t index = 0; index < kMixRows; ++index) {
      if (!isfinite(base[index])) atomicOr(error_flag, 32U);
    }
    for (std::uint32_t stream = 0; stream < kStreams; ++stream) {
      pre[stream] = stable_sigmoid(mixes[stream] * scale[0] + base[stream]) +
                    pre_epsilon;
      post_mix[post_base + stream] =
          stable_sigmoid(mixes[kStreams + stream] * scale[1] +
                         base[kStreams + stream]) * post_multiplier;
    }
    for (std::uint32_t input = 0; input < kStreams; ++input) {
      float maximum = -FLT_MAX;
      for (std::uint32_t output = 0; output < kStreams; ++output) {
        const auto index = 2 * kStreams + input * kStreams + output;
        mixes[index] = mixes[index] * scale[2] + base[index];
        maximum = fmaxf(maximum, mixes[index]);
      }
      float sum = 0.0F;
      for (std::uint32_t output = 0; output < kStreams; ++output) {
        const auto index = 2 * kStreams + input * kStreams + output;
        const auto value = expf(mixes[index] - maximum);
        residual_mix[comb_base + input * kStreams + output] = value;
        sum += value;
      }
      for (std::uint32_t output = 0; output < kStreams; ++output) {
        residual_mix[comb_base + input * kStreams + output] =
            residual_mix[comb_base + input * kStreams + output] / sum +
            sinkhorn_epsilon;
      }
    }
    for (std::uint32_t iteration = 0; iteration < sinkhorn_iterations;
         ++iteration) {
      if (iteration != 0) {
        for (std::uint32_t input = 0; input < kStreams; ++input) {
          float sum = sinkhorn_epsilon;
          for (std::uint32_t output = 0; output < kStreams; ++output)
            sum += residual_mix[comb_base + input * kStreams + output];
          for (std::uint32_t output = 0; output < kStreams; ++output)
            residual_mix[comb_base + input * kStreams + output] /= sum;
        }
      }
      for (std::uint32_t output = 0; output < kStreams; ++output) {
        float sum = sinkhorn_epsilon;
        for (std::uint32_t input = 0; input < kStreams; ++input)
          sum += residual_mix[comb_base + input * kStreams + output];
        for (std::uint32_t input = 0; input < kStreams; ++input)
          residual_mix[comb_base + input * kStreams + output] /= sum;
      }
    }
  }
  __syncthreads();
  const auto input_base = static_cast<std::uint64_t>(token) * hidden_size;
  for (std::uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    float value = 0.0F;
    for (std::uint32_t stream = 0; stream < kStreams; ++stream) {
      value += pre[stream] * __bfloat162float(
          residual[residual_base + static_cast<std::uint64_t>(stream) *
                                   hidden_size + column]);
    }
    if (!isfinite(value)) atomicOr(error_flag, 8U);
    layer_input[input_base + column] = __float2bfloat16_rn(value);
  }
  __syncthreads();
  float layer_square_sum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    const auto value = __bfloat162float(layer_input[input_base + column]);
    layer_square_sum += value * value;
  }
  const auto layer_total = reduce_block(layer_square_sum, scratch);
  if (threadIdx.x == 0) {
    mixes[0] = rsqrtf(layer_total / static_cast<float>(hidden_size) +
                      rms_epsilon);
    if (!isfinite(mixes[0])) atomicOr(error_flag, 64U);
  }
  __syncthreads();
  for (std::uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    const auto weight = __bfloat162float(norm_weight[column]);
    const auto value = __bfloat162float(layer_input[input_base + column]) *
                       mixes[0] * weight;
    if (!isfinite(value) || !isfinite(weight)) atomicOr(error_flag, 64U);
    layer_input[input_base + column] = __float2bfloat16_rn(value);
  }
}

__global__ void mhc_post_kernel(
    const __nv_bfloat16* layer_output, const __nv_bfloat16* residual,
    const float* post_mix, const float* residual_mix,
    __nv_bfloat16* output, std::uint32_t* error_flag,
    std::uint32_t hidden_size) {
  const auto token = blockIdx.x;
  const auto hidden_base = static_cast<std::uint64_t>(token) * hidden_size;
  const auto residual_base = hidden_base * kStreams;
  const auto mix_base = static_cast<std::uint64_t>(token) * kStreams * kStreams;
  for (std::uint32_t target = 0; target < kStreams; ++target) {
    for (std::uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      float value = post_mix[token * kStreams + target] *
                    __bfloat162float(layer_output[hidden_base + column]);
      for (std::uint32_t source = 0; source < kStreams; ++source) {
        value += residual_mix[mix_base + source * kStreams + target] *
                 __bfloat162float(residual[
                     residual_base + static_cast<std::uint64_t>(source) *
                                         hidden_size + column]);
      }
      if (!isfinite(value)) atomicOr(error_flag, 16U);
      output[residual_base + static_cast<std::uint64_t>(target) * hidden_size +
             column] = __float2bfloat16_rn(value);
    }
  }
}

__global__ void mhc_target_hidden_tap_kernel(
    const __nv_bfloat16* residual, __nv_bfloat16* target_hidden,
    std::uint32_t* error_flag, std::uint32_t hidden_size,
    std::uint32_t target_stage_index) {
  const auto token = blockIdx.x;
  const auto source_base =
      static_cast<std::uint64_t>(token) * kStreams * hidden_size;
  const auto target_base =
      static_cast<std::uint64_t>(token) * 3U * hidden_size +
      static_cast<std::uint64_t>(target_stage_index) * hidden_size;
  for (std::uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    float value = 0.0F;
    for (std::uint32_t source = 0; source < kStreams; ++source) {
      value += __bfloat162float(
          residual[source_base +
                   static_cast<std::uint64_t>(source) * hidden_size + column]);
    }
    value *= 0.25F;
    if (!isfinite(value)) atomicOr(error_flag, 128U);
    target_hidden[target_base + column] = __float2bfloat16_rn(value);
  }
}

}  // namespace

Status launch_deepseek_mhc_pre(DeepSeekMhcPreLaunch launch) {
  auto status = validate_deepseek_mhc_pre_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(cudaPeekAtLastError(),
                       "cudaPeekAtLastError before DeepSeek mHC pre");
  if (!status.ok()) return status;
  mhc_pre_kernel<<<launch.token_count, 256, 0,
                   reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.residual_bf16),
      reinterpret_cast<const float*>(launch.fn_f32),
      reinterpret_cast<const float*>(launch.scale_f32),
      reinterpret_cast<const float*>(launch.base_f32),
      reinterpret_cast<const __nv_bfloat16*>(launch.norm_weight_bf16),
      reinterpret_cast<float*>(launch.post_mix_f32),
      reinterpret_cast<float*>(launch.residual_mix_f32),
      reinterpret_cast<__nv_bfloat16*>(launch.layer_input_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.hidden_size, launch.rms_epsilon, launch.pre_epsilon,
      launch.sinkhorn_epsilon, launch.post_multiplier,
      launch.sinkhorn_iterations);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek mHC pre launch");
}

Status launch_deepseek_mhc_post(DeepSeekMhcPostLaunch launch) {
  auto status = validate_deepseek_mhc_post_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(cudaPeekAtLastError(),
                       "cudaPeekAtLastError before DeepSeek mHC post");
  if (!status.ok()) return status;
  mhc_post_kernel<<<launch.token_count, 256, 0,
                    reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.layer_output_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.residual_bf16),
      reinterpret_cast<const float*>(launch.post_mix_f32),
      reinterpret_cast<const float*>(launch.residual_mix_f32),
      reinterpret_cast<__nv_bfloat16*>(launch.output_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.hidden_size);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek mHC post launch");
}

Status launch_deepseek_mhc_target_hidden_tap(
    DeepSeekMhcTargetHiddenTapLaunch launch) {
  auto status = validate_deepseek_mhc_target_hidden_tap_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(
      cudaPeekAtLastError(),
      "cudaPeekAtLastError before DeepSeek mHC target hidden tap");
  if (!status.ok()) return status;
  mhc_target_hidden_tap_kernel<<<
      launch.token_count, 256, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.residual_hc_bf16),
      reinterpret_cast<__nv_bfloat16*>(launch.target_hidden_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.hidden_size, launch.target_stage_index);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek mHC target hidden tap launch");
}

}  // namespace pih
