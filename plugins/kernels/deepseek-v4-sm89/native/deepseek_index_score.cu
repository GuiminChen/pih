#include "pih/backend/cuda/deepseek_index_score.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

__global__ void index_score_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* index_kv,
    const float* head_weight, float* score,
    std::uint32_t* error_flag, std::uint32_t head_count,
    std::uint32_t slot_count, const std::uint32_t* page_slots,
    std::uint32_t slot_base, std::uint32_t physical_page_count) {
  __shared__ float contributions[DeepSeekIndexScoreLaunch::kMaximumHeads];
  const auto query_ordinal = blockIdx.x / slot_count;
  const auto slot = blockIdx.x % slot_count;
  const auto head = threadIdx.x;
  const auto logical_slot = slot_base + slot;
  const auto physical_page =
      page_slots == nullptr ? 0U : page_slots[logical_slot / 64U];
  const bool valid_page =
      page_slots == nullptr || physical_page < physical_page_count;
  if (!valid_page && head == 0) atomicOr(error_flag, 4U);
  float contribution = 0.0F;
  if (head < head_count && valid_page) {
    float dot = 0.0F;
    const auto query_base =
        (static_cast<std::uint64_t>(query_ordinal) * head_count + head) *
        DeepSeekIndexScoreLaunch::kHeadDim;
    const auto physical_slot = page_slots == nullptr
        ? slot
        : static_cast<std::uint64_t>(physical_page) * 64U +
              logical_slot % 64U;
    const auto kv_base = static_cast<std::uint64_t>(physical_slot) *
                         DeepSeekIndexScoreLaunch::kHeadDim;
    for (std::uint32_t column = 0;
         column < DeepSeekIndexScoreLaunch::kHeadDim; ++column) {
      dot += __bfloat162float(query[query_base + column]) *
             __bfloat162float(index_kv[kv_base + column]);
    }
    const auto weight =
        head_weight[static_cast<std::uint64_t>(query_ordinal) * head_count +
                    head];
    contribution = fmaxf(dot, 0.0F) * weight;
    if (!isfinite(contribution)) {
      atomicOr(error_flag, 1U);
      contribution = 0.0F;
    }
  }
  contributions[head] = contribution;
  __syncthreads();
  for (std::uint32_t stride = 32; stride != 0; stride >>= 1U) {
    if (head < stride) contributions[head] += contributions[head + stride];
    __syncthreads();
  }
  if (head == 0) {
    if (!isfinite(contributions[0])) {
      atomicOr(error_flag, 2U);
      score[static_cast<std::uint64_t>(query_ordinal) * slot_count + slot] =
          0.0F;
    } else {
      score[static_cast<std::uint64_t>(query_ordinal) * slot_count + slot] =
          contributions[0];
    }
  }
}

}  // namespace

Status launch_deepseek_index_score(DeepSeekIndexScoreLaunch launch) {
  auto status = validate_deepseek_index_score_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(cudaPeekAtLastError(),
                       "cudaPeekAtLastError before DeepSeek index score");
  if (!status.ok()) return status;
  const auto blocks = launch.query_count * launch.slot_count;
  index_score_kernel<<<
      blocks, DeepSeekIndexScoreLaunch::kMaximumHeads, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.query_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.index_kv_bf16),
      reinterpret_cast<const float*>(launch.head_weight_f32),
      reinterpret_cast<float*>(launch.score_f32),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.head_count, launch.slot_count,
      reinterpret_cast<const std::uint32_t*>(launch.page_slots_u32),
      launch.slot_base, launch.physical_page_count);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek index score launch");
}

}  // namespace pih
