#include "pih/backend/cuda/deepseek_route_gather.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

__global__ void route_gather_kernel(
    const __nv_bfloat16* source, const std::uint32_t* token_indices,
    __nv_bfloat16* destination, std::uint32_t* error_flag,
    std::uint32_t route_count, std::uint32_t packed_token_count) {
  const auto elements = static_cast<std::uint64_t>(route_count) *
                        DeepSeekRouteGatherLaunch::kHiddenSize;
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const auto route = static_cast<std::uint32_t>(
        index / DeepSeekRouteGatherLaunch::kHiddenSize);
    const auto column = static_cast<std::uint32_t>(
        index % DeepSeekRouteGatherLaunch::kHiddenSize);
    const auto token = token_indices[route];
    if (token >= packed_token_count) {
      atomicOr(error_flag, 1U);
      continue;
    }
    destination[index] =
        source[static_cast<std::uint64_t>(token) *
                   DeepSeekRouteGatherLaunch::kHiddenSize +
               column];
  }
}

}  // namespace

Status launch_deepseek_route_gather(DeepSeekRouteGatherLaunch launch) {
  const auto valid = validate_deepseek_route_gather_launch(launch);
  if (!valid.ok()) return valid;
  const auto clean = cuda_status(
      cudaPeekAtLastError(), "cudaPeekAtLastError before DeepSeek gather");
  if (!clean.ok()) return clean;
  const auto elements = static_cast<std::uint64_t>(launch.route_count) *
                        DeepSeekRouteGatherLaunch::kHiddenSize;
  const auto blocks = static_cast<std::uint32_t>((elements + 255U) / 256U);
  route_gather_kernel<<<
      blocks, 256, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.source_hidden_bf16),
      reinterpret_cast<const std::uint32_t*>(launch.token_indices_u32),
      reinterpret_cast<__nv_bfloat16*>(launch.route_hidden_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag), launch.route_count,
      launch.packed_token_count);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek route gather launch");
}

}  // namespace pih
