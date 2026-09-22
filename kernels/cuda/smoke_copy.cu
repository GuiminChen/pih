#include "pih/backend/cuda/smoke_copy.h"

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {
namespace {

__global__ void smoke_copy_kernel(const std::uint8_t* source,
                                  std::uint8_t* destination,
                                  std::uint64_t bytes) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index < bytes) {
    destination[index] = source[index];
  }
}

}  // namespace

Status cuda_smoke_copy(std::uint64_t bytes) {
  constexpr std::uint64_t kMaximumBytes = 64ULL * 1024 * 1024;
  if (bytes == 0 || bytes > kMaximumBytes) {
    return Status::InvalidArgument("smoke copy bytes must be in [1, 64 MiB]");
  }
  std::vector<std::uint8_t> input(static_cast<std::size_t>(bytes));
  std::vector<std::uint8_t> output(static_cast<std::size_t>(bytes));
  for (std::uint64_t i = 0; i < bytes; ++i) {
    input[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i * 131U + 17U);
  }

  std::uint8_t* source = nullptr;
  std::uint8_t* destination = nullptr;
  auto error = cudaMalloc(&source, static_cast<std::size_t>(bytes));
  if (error != cudaSuccess) return cuda_status(error, "smoke cudaMalloc source");
  error = cudaMalloc(&destination, static_cast<std::size_t>(bytes));
  if (error != cudaSuccess) {
    cudaFree(source);
    return cuda_status(error, "smoke cudaMalloc destination");
  }
  error = cudaMemcpy(source, input.data(), static_cast<std::size_t>(bytes),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    constexpr unsigned kThreads = 256;
    const auto blocks = static_cast<unsigned>((bytes + kThreads - 1) / kThreads);
    smoke_copy_kernel<<<blocks, kThreads>>>(source, destination, bytes);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(output.data(), destination, static_cast<std::size_t>(bytes),
                       cudaMemcpyDeviceToHost);
  }
  cudaFree(destination);
  cudaFree(source);
  if (error != cudaSuccess) return cuda_status(error, "smoke copy execution");
  if (output != input) return Status::Internal("smoke copy data mismatch");
  return Status::Ok();
}

}  // namespace pih
