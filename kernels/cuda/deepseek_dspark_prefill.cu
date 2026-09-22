#include "pih/backend/cuda/deepseek_dspark_prefill.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

__global__ void dspark_recent_store_kernel(
    const __nv_bfloat16* source, const std::uint32_t* positions,
    __nv_bfloat16* recent, std::uint32_t* error,
    std::uint32_t token_count, std::uint32_t maximum_position_count) {
  constexpr std::uint32_t kRingRows = 128;
  constexpr std::uint32_t kVectorDimension = 512;
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  std::uint32_t latest = token_count;
  for (std::uint32_t token = 0; token < token_count; ++token) {
    const auto position = positions[token];
    if (position < maximum_position_count &&
        position % kRingRows == row) {
      latest = token;
    }
  }
  if (latest != token_count) {
    for (std::uint32_t column = threadIdx.x;
         column < kVectorDimension; column += blockDim.x) {
      recent[static_cast<std::uint64_t>(row) * kVectorDimension + column] =
          source[static_cast<std::uint64_t>(latest) * kVectorDimension +
                 column];
    }
  }
  if (row == 0 && threadIdx.x == 0) {
    const auto first = positions[0];
    if (first >= maximum_position_count) atomicOr(error, 1U);
    for (std::uint32_t token = 1; token < token_count; ++token) {
      if (first > 0xFFFFFFFFU - token ||
          positions[token] != first + token ||
          positions[token] >= maximum_position_count) {
        atomicOr(error, 1U);
      }
    }
  }
}

}  // namespace

Status launch_deepseek_dspark_recent_store(
    DeepSeekDsparkRecentStoreLaunch launch) {
  auto status = validate_deepseek_dspark_recent_store_launch(launch);
  if (!status.ok()) return status;
  status = cuda_status(
      cudaPeekAtLastError(),
      "cudaPeekAtLastError before DeepSeek DSpark recent store");
  if (!status.ok()) return status;
  dspark_recent_store_kernel<<<
      launch.ring_rows, 256, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.source_bf16),
      reinterpret_cast<const std::uint32_t*>(launch.positions_u32),
      reinterpret_cast<__nv_bfloat16*>(launch.recent_ring_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.token_count, launch.maximum_position_count);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek DSpark recent store launch");
}

}  // namespace pih
