#include "pih/backend/cuda/deepseek_mxfp4_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

constexpr std::uint32_t kMaximumTileValues = 4096;

__device__ float e2m1_magnitude(std::uint8_t bits) {
  constexpr float values[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                               2.0F, 3.0F, 4.0F, 6.0F};
  return values[bits & 0x07U];
}

__global__ void deepseek_mxfp4_decode_tile_kernel(
    const std::uint8_t* packed, const std::uint8_t* scales,
    __nv_bfloat16* output, std::uint32_t* error_flag,
    std::uint64_t logical_k, std::uint64_t tile_begin,
    std::uint32_t tile_values) {
  for (std::uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
       local < tile_values; local += blockDim.x * gridDim.x) {
    const auto k = tile_begin + local;
    if (k >= logical_k) {
      atomicOr(error_flag, 1U);
      continue;
    }
    const auto byte = packed[k / 2U];
    const auto nibble = static_cast<std::uint8_t>(
        (k & 1U) == 0 ? byte & 0x0FU : byte >> 4U);
    const auto scale_bits = scales[k / 32U];
    if (scale_bits == 0xFFU) {
      atomicOr(error_flag, 2U);
      continue;
    }
    float value = e2m1_magnitude(nibble);
    if ((nibble & 0x08U) != 0) value = -value;
    value = ldexpf(value, static_cast<int>(scale_bits) - 127);
    if (!isfinite(value)) {
      atomicOr(error_flag, 4U);
      continue;
    }
    output[local] = __float2bfloat16_rn(value);
  }
}

}  // namespace

Status launch_deepseek_mxfp4_decode_tile(DeepSeekMxfp4Tile tile) {
  if (tile.packed == 0 || tile.scale_bits == 0 || tile.output_bf16 == 0 ||
      tile.error_flag == 0 || tile.stream == 0 || tile.logical_k == 0 ||
      tile.tile_values == 0 || tile.tile_values > kMaximumTileValues ||
      tile.tile_begin >= tile.logical_k ||
      tile.tile_values > tile.logical_k - tile.tile_begin) {
    return Status::InvalidArgument("DeepSeek MXFP4 tile launch is invalid");
  }
  const auto clean_before = cuda_status(
      cudaPeekAtLastError(), "cudaPeekAtLastError before DeepSeek MXFP4 decode");
  if (!clean_before.ok()) return clean_before;
  const auto blocks = (tile.tile_values + 255U) / 256U;
  deepseek_mxfp4_decode_tile_kernel<<<
      blocks, 256, 0, reinterpret_cast<cudaStream_t>(tile.stream)>>>(
      reinterpret_cast<const std::uint8_t*>(tile.packed),
      reinterpret_cast<const std::uint8_t*>(tile.scale_bits),
      reinterpret_cast<__nv_bfloat16*>(tile.output_bf16),
      reinterpret_cast<std::uint32_t*>(tile.error_flag), tile.logical_k,
      tile.tile_begin, tile.tile_values);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek MXFP4 decode tile launch");
}

}  // namespace pih
