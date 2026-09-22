#include "pih/backend/cuda/deepseek_rms_norm.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

__global__ void deepseek_rms_norm_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, std::uint32_t* error_flag,
    std::uint32_t rows, std::uint32_t hidden_size) {
  __shared__ float warp_sums[8];
  for (std::uint32_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const auto offset = static_cast<std::uint64_t>(row) * hidden_size;
    float sum = 0.0F;
    for (std::uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const auto value = __bfloat162float(input[offset + column]);
      const auto scale = __bfloat162float(weight[column]);
      if (!isfinite(value) || !isfinite(scale)) {
        atomicOr(error_flag, 1U);
      }
      sum += value * value;
    }
    for (std::uint32_t delta = 16; delta != 0; delta >>= 1U) {
      sum += __shfl_down_sync(0xffffffffU, sum, delta);
    }
    if ((threadIdx.x & 31U) == 0) warp_sums[threadIdx.x >> 5U] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
      sum = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.0F;
      for (std::uint32_t delta = 16; delta != 0; delta >>= 1U) {
        sum += __shfl_down_sync(0xffffffffU, sum, delta);
      }
      if (threadIdx.x == 0) {
        warp_sums[0] = rsqrtf(
            sum / static_cast<float>(hidden_size) +
            DeepSeekRmsNormLaunch::kEpsilon);
        if (!isfinite(warp_sums[0])) atomicOr(error_flag, 2U);
      }
    }
    __syncthreads();
    const auto inverse_rms = warp_sums[0];
    for (std::uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const auto value = __bfloat162float(input[offset + column]) *
                         inverse_rms * __bfloat162float(weight[column]);
      if (!isfinite(value)) {
        atomicOr(error_flag, 4U);
      } else {
        output[offset + column] = __float2bfloat16_rn(value);
      }
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" __global__ void deepseek_fused_residual_rms_norm_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* residual,
    const __nv_bfloat16* weight, __nv_bfloat16* residual_output,
    __nv_bfloat16* normalized_output, std::uint32_t* error_flag,
    std::uint32_t rows) {
  __shared__ float warp_sums[8];
  constexpr std::uint32_t hidden_size = 4096;
  for (std::uint32_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const auto offset = static_cast<std::uint64_t>(row) * hidden_size;
    float sum = 0.0F;
    for (std::uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const auto left = __bfloat162float(input[offset + column]);
      const auto right = __bfloat162float(residual[offset + column]);
      const auto scale = __bfloat162float(weight[column]);
      if (!isfinite(left) || !isfinite(right) || !isfinite(scale)) {
        atomicOr(error_flag, 1U);
      }
      const auto combined_bf16 = __float2bfloat16_rn(left + right);
      residual_output[offset + column] = combined_bf16;
      const auto combined = __bfloat162float(combined_bf16);
      sum += combined * combined;
    }
    for (std::uint32_t delta = 16; delta != 0; delta >>= 1U) {
      sum += __shfl_down_sync(0xffffffffU, sum, delta);
    }
    if ((threadIdx.x & 31U) == 0) warp_sums[threadIdx.x >> 5U] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
      sum = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.0F;
      for (std::uint32_t delta = 16; delta != 0; delta >>= 1U) {
        sum += __shfl_down_sync(0xffffffffU, sum, delta);
      }
      if (threadIdx.x == 0) {
        warp_sums[0] = rsqrtf(
            sum / static_cast<float>(hidden_size) +
            DeepSeekFusedResidualRmsNormLaunch::kEpsilon);
        if (!isfinite(warp_sums[0])) atomicOr(error_flag, 2U);
      }
    }
    __syncthreads();
    const auto inverse_rms = warp_sums[0];
    for (std::uint32_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const auto value = __bfloat162float(residual_output[offset + column]) *
                         inverse_rms * __bfloat162float(weight[column]);
      if (!isfinite(value)) {
        atomicOr(error_flag, 4U);
      } else {
        normalized_output[offset + column] = __float2bfloat16_rn(value);
      }
    }
    __syncthreads();
  }
}

namespace {

__global__ void initialize_fused_fixture(__nv_bfloat16* storage,
                                         std::uint64_t elements) {
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    storage[index] = __float2bfloat16_rn(
        (static_cast<int>(index % 31U) - 15) * 0.03125F);
    storage[elements + index] = __float2bfloat16_rn(
        (static_cast<int>(index % 17U) - 8) * 0.015625F);
    storage[2 * elements + index] = __float2bfloat16_rn(
        0.75F + static_cast<float>(index % 13U) * 0.0078125F);
  }
}

__global__ void residual_add_fixture(const __nv_bfloat16* input,
                                     const __nv_bfloat16* residual,
                                     __nv_bfloat16* output,
                                     std::uint64_t elements) {
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    output[index] = __float2bfloat16_rn(
        __bfloat162float(input[index]) + __bfloat162float(residual[index]));
  }
}

__global__ void compare_fused_fixture(const __nv_bfloat16* fused_residual,
                                      const __nv_bfloat16* fused_norm,
                                      const __nv_bfloat16* control_residual,
                                      const __nv_bfloat16* control_norm,
                                      std::uint32_t* mismatch,
                                      std::uint64_t elements) {
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    if (__bfloat16_as_ushort(fused_residual[index]) !=
            __bfloat16_as_ushort(control_residual[index]) ||
        __bfloat16_as_ushort(fused_norm[index]) !=
            __bfloat16_as_ushort(control_norm[index])) {
      atomicOr(mismatch, 1U);
    }
  }
}

}  // namespace

Status launch_deepseek_rms_norm(DeepSeekRmsNormLaunch launch) {
  const auto valid = validate_deepseek_rms_norm_launch(launch);
  if (!valid.ok()) return valid;
  const auto clean = cuda_status(
      cudaPeekAtLastError(), "cudaPeekAtLastError before DeepSeek RMSNorm");
  if (!clean.ok()) return clean;
  deepseek_rms_norm_kernel<<<
      launch.rows, 256, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.input_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.weight_bf16),
      reinterpret_cast<__nv_bfloat16*>(launch.output_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag), launch.rows,
      launch.hidden_size);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek RMSNorm launch");
}

Status launch_deepseek_fused_residual_rms_norm(
    DeepSeekFusedResidualRmsNormLaunch launch) {
  const auto valid = validate_deepseek_fused_residual_rms_norm_launch(launch);
  if (!valid.ok()) return valid;
  const auto clean = cuda_status(
      cudaPeekAtLastError(),
      "cudaPeekAtLastError before DeepSeek fused residual RMSNorm");
  if (!clean.ok()) return clean;
  deepseek_fused_residual_rms_norm_kernel<<<
      launch.rows, 256, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.input_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.residual_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.weight_bf16),
      reinterpret_cast<__nv_bfloat16*>(launch.residual_output_bf16),
      reinterpret_cast<__nv_bfloat16*>(launch.normalized_output_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag), launch.rows);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek fused residual RMSNorm launch");
}

Status cuda_verify_deepseek_fused_residual_rms_norm_fixture() {
  constexpr std::uint32_t kRows = 2;
  constexpr std::uint64_t kElements = 4096ULL * kRows;
  constexpr std::size_t kStorageBytes =
      7ULL * kElements * sizeof(__nv_bfloat16);
  __nv_bfloat16* storage = nullptr;
  std::uint32_t* flags = nullptr;
  cudaStream_t stream = nullptr;
  auto error = cudaMalloc(&storage, kStorageBytes);
  if (error == cudaSuccess) error = cudaMalloc(&flags, 3 * sizeof(*flags));
  if (error == cudaSuccess) {
    error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
  if (error == cudaSuccess) error = cudaMemsetAsync(flags, 0, 3 * sizeof(*flags), stream);
  if (error == cudaSuccess) {
    initialize_fused_fixture<<<32, 256, 0, stream>>>(storage, kElements);
    error = cudaGetLastError();
  }
  auto* input = storage;
  auto* residual = storage == nullptr ? nullptr : storage + kElements;
  auto* weight = storage == nullptr ? nullptr : storage + 2 * kElements;
  auto* fused_residual = storage == nullptr ? nullptr : storage + 3 * kElements;
  auto* fused_norm = storage == nullptr ? nullptr : storage + 4 * kElements;
  auto* control_residual = storage == nullptr ? nullptr : storage + 5 * kElements;
  auto* control_norm = storage == nullptr ? nullptr : storage + 6 * kElements;
  if (error == cudaSuccess) {
    const auto status = launch_deepseek_fused_residual_rms_norm({
        reinterpret_cast<std::uintptr_t>(input),
        reinterpret_cast<std::uintptr_t>(residual),
        reinterpret_cast<std::uintptr_t>(weight),
        reinterpret_cast<std::uintptr_t>(fused_residual),
        reinterpret_cast<std::uintptr_t>(fused_norm),
        reinterpret_cast<std::uintptr_t>(flags),
        reinterpret_cast<std::uintptr_t>(stream), kRows, 4096,
        DeepSeekFusedResidualRmsNormLaunch::kEpsilon});
    if (!status.ok()) error = cudaErrorLaunchFailure;
  }
  if (error == cudaSuccess) {
    residual_add_fixture<<<32, 256, 0, stream>>>(
        input, residual, control_residual, kElements);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    const auto status = launch_deepseek_rms_norm({
        reinterpret_cast<std::uintptr_t>(control_residual),
        reinterpret_cast<std::uintptr_t>(weight),
        reinterpret_cast<std::uintptr_t>(control_norm),
        reinterpret_cast<std::uintptr_t>(flags + 1),
        reinterpret_cast<std::uintptr_t>(stream), kRows, 4096,
        DeepSeekRmsNormLaunch::kEpsilon});
    if (!status.ok()) error = cudaErrorLaunchFailure;
  }
  if (error == cudaSuccess) {
    compare_fused_fixture<<<32, 256, 0, stream>>>(
        fused_residual, fused_norm, control_residual, control_norm,
        flags + 2, kElements);
    error = cudaGetLastError();
  }
  std::uint32_t observed[3]{};
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(observed, flags, sizeof(observed),
                            cudaMemcpyDeviceToHost, stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (flags != nullptr) cudaFree(flags);
  if (storage != nullptr) cudaFree(storage);
  if (error != cudaSuccess) {
    return cuda_status(error, "DeepSeek fused residual RMSNorm fixture");
  }
  if (observed[0] != 0 || observed[1] != 0 || observed[2] != 0) {
    return Status::Internal(
        "DeepSeek fused residual RMSNorm differs from locked control fixture");
  }
  return Status::Ok();
}

}  // namespace pih
