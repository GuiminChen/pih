#include "pih/backend/cuda/deepseek_sparse_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

__device__ constexpr float kNegativeInfinity = -__builtin_huge_valf();

__device__ const __nv_bfloat16* resolve_kv_row(
    const __nv_bfloat16* recent_kv, const __nv_bfloat16* compressed_kv,
    const std::uint32_t* page_slots, std::int32_t index,
    std::int32_t recent_offset, std::int32_t compressed_offset,
    std::uint32_t compressed_slot_count, std::uint32_t logical_page_count,
    std::uint32_t physical_page_count, std::uint32_t* error_flag) {
  if (compressed_kv == nullptr) {
    return recent_kv + static_cast<std::uint64_t>(index) *
                           DeepSeekSparseAttentionLaunch::kHeadDim;
  }
  if (index >= recent_offset &&
      static_cast<std::uint32_t>(index - recent_offset) < 128U) {
    return recent_kv + static_cast<std::uint64_t>(index - recent_offset) *
                           DeepSeekSparseAttentionLaunch::kHeadDim;
  }
  if (index < compressed_offset ||
      static_cast<std::uint32_t>(index - compressed_offset) >=
          compressed_slot_count) {
    atomicOr(error_flag, 16U);
    return nullptr;
  }
  const auto logical = static_cast<std::uint32_t>(index - compressed_offset);
  const auto logical_page = logical / 64U;
  if (logical_page >= logical_page_count) {
    atomicOr(error_flag, 16U);
    return nullptr;
  }
  const auto physical_page = page_slots[logical_page];
  if (physical_page >= physical_page_count) {
    atomicOr(error_flag, 16U);
    return nullptr;
  }
  const auto physical_row = physical_page * 64U + logical % 64U;
  return compressed_kv + static_cast<std::uint64_t>(physical_row) *
                             DeepSeekSparseAttentionLaunch::kHeadDim;
}

__global__ void sparse_attention_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* recent_kv,
    const __nv_bfloat16* compressed_kv, const std::uint32_t* page_slots,
    const float* attention_sink, const std::int32_t* indices,
    __nv_bfloat16* output, std::uint32_t* error_flag,
    std::uint32_t query_count, std::uint32_t head_count,
    std::uint32_t kv_count, std::uint32_t index_count,
    std::int32_t recent_offset, std::int32_t compressed_offset,
    std::uint32_t compressed_slot_count, std::uint32_t logical_page_count,
    std::uint32_t physical_page_count) {
  __shared__ float weights[DeepSeekSparseAttentionLaunch::kTileIndices];
  __shared__ float previous_scale;
  __shared__ float running_max;
  __shared__ float running_sum;
  __shared__ std::uint32_t valid_count;
  const auto query_ordinal = blockIdx.x / head_count;
  const auto head = blockIdx.x % head_count;
  float first_accumulator = 0.0F;
  float second_accumulator = 0.0F;
  if (threadIdx.x == 0) {
    running_max = kNegativeInfinity;
    running_sum = 0.0F;
    valid_count = 0;
  }
  __syncthreads();
  for (std::uint32_t tile = 0; tile < index_count;
       tile += DeepSeekSparseAttentionLaunch::kTileIndices) {
    const auto count = min(DeepSeekSparseAttentionLaunch::kTileIndices,
                           index_count - tile);
    if (threadIdx.x == 0) {
      float tile_max = kNegativeInfinity;
      for (std::uint32_t item = 0; item < count; ++item) {
        const auto index = indices[static_cast<std::uint64_t>(query_ordinal) *
                                       index_count +
                                   tile + item];
        if (index == -1) {
          weights[item] = kNegativeInfinity;
          continue;
        }
        if (index < 0 || static_cast<std::uint32_t>(index) >= kv_count) {
          atomicOr(error_flag, 1U);
          weights[item] = kNegativeInfinity;
          continue;
        }
        const auto* kv = resolve_kv_row(
            recent_kv, compressed_kv, page_slots, index, recent_offset,
            compressed_offset, compressed_slot_count, logical_page_count,
            physical_page_count, error_flag);
        if (kv == nullptr) {
          weights[item] = kNegativeInfinity;
          continue;
        }
        ++valid_count;
        float score = 0.0F;
        for (std::uint32_t column = 0;
             column < DeepSeekSparseAttentionLaunch::kHeadDim; ++column) {
          score += __bfloat162float(
                       query[(static_cast<std::uint64_t>(query_ordinal) *
                                  head_count +
                              head) *
                                 DeepSeekSparseAttentionLaunch::kHeadDim +
                             column]) *
                   __bfloat162float(kv[column]);
        }
        score *= rsqrtf(
            static_cast<float>(DeepSeekSparseAttentionLaunch::kHeadDim));
        if (!isfinite(score)) {
          atomicOr(error_flag, 2U);
          score = kNegativeInfinity;
        }
        weights[item] = score;
        tile_max = fmaxf(tile_max, score);
      }
      const auto next_max = fmaxf(running_max, tile_max);
      previous_scale = isinf(running_max) ? 0.0F
                                           : expf(running_max - next_max);
      running_sum *= previous_scale;
      for (std::uint32_t item = 0; item < count; ++item) {
        if (isinf(weights[item]) && weights[item] < 0.0F) {
          weights[item] = 0.0F;
          continue;
        }
        const auto value = expf(weights[item] - next_max);
        running_sum += value;
        weights[item] =
            __bfloat162float(__float2bfloat16_rn(value));
      }
      running_max = next_max;
    }
    __syncthreads();
    first_accumulator *= previous_scale;
    second_accumulator *= previous_scale;
    const auto first_column = threadIdx.x;
    const auto second_column = threadIdx.x + blockDim.x;
    for (std::uint32_t item = 0; item < count; ++item) {
      const auto index = indices[static_cast<std::uint64_t>(query_ordinal) *
                                     index_count +
                                 tile + item];
      if (index < 0 || static_cast<std::uint32_t>(index) >= kv_count) continue;
      const auto* kv = resolve_kv_row(
          recent_kv, compressed_kv, page_slots, index, recent_offset,
          compressed_offset, compressed_slot_count, logical_page_count,
          physical_page_count, error_flag);
      if (kv == nullptr) continue;
      first_accumulator +=
          weights[item] *
          __bfloat162float(kv[first_column]);
      second_accumulator +=
          weights[item] *
          __bfloat162float(kv[second_column]);
    }
    __syncthreads();
  }
  if (valid_count == 0) {
    if (threadIdx.x == 0) atomicOr(error_flag, 4U);
    return;
  }
  const auto sink = attention_sink[head];
  const auto denominator = running_sum + expf(sink - running_max);
  if (!isfinite(sink) || !isfinite(denominator) || denominator <= 0.0F) {
    if (threadIdx.x == 0) atomicOr(error_flag, 8U);
    return;
  }
  const auto base = (static_cast<std::uint64_t>(query_ordinal) * head_count +
                     head) *
                    DeepSeekSparseAttentionLaunch::kHeadDim;
  output[base + threadIdx.x] =
      __float2bfloat16_rn(first_accumulator / denominator);
  output[base + threadIdx.x + blockDim.x] =
      __float2bfloat16_rn(second_accumulator / denominator);
}

}  // namespace

Status launch_deepseek_sparse_attention(
    DeepSeekSparseAttentionLaunch launch) {
  auto status = validate_deepseek_sparse_attention_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(cudaPeekAtLastError(),
                       "cudaPeekAtLastError before DeepSeek sparse attention");
  if (!status.ok()) return status;
  const auto blocks = launch.query_count * launch.head_count;
  sparse_attention_kernel<<<
      blocks, 256, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.query_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.latent_kv_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.compressed_kv_bf16),
      reinterpret_cast<const std::uint32_t*>(launch.page_slots_u32),
      reinterpret_cast<const float*>(launch.attention_sink_f32),
      reinterpret_cast<const std::int32_t*>(launch.indices_i32),
      reinterpret_cast<__nv_bfloat16*>(launch.output_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.query_count, launch.head_count, launch.kv_count,
      launch.index_count, launch.recent_physical_offset,
      launch.compressed_physical_offset, launch.compressed_slot_count,
      launch.logical_page_count, launch.physical_page_count);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek sparse attention launch");
}

}  // namespace pih
