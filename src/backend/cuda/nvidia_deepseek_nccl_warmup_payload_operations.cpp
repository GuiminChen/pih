#include "pih/backend/cuda/nvidia_deepseek_nccl_warmup_payload_operations.h"

#include <algorithm>
#include <limits>
#include <vector>

#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_status.h"

namespace pih {

Status NvidiaDeepSeekNcclWarmupPayloadOperations::prepare(
    DeepSeekNcclRole role, void* device_buffer, std::uint64_t bytes,
    DriverStreamHandle stream, std::uint64_t pattern_identity) {
  if (device_buffer == nullptr || bytes == 0 || stream == 0 ||
      bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument("DeepSeek NCCL CUDA warm-up preparation is invalid");
  }
  const auto byte_pattern = role == DeepSeekNcclRole::kSend
      ? static_cast<int>((pattern_identity % 255U) + 1U) : 0;
  return cuda_status(cudaMemsetAsync(
      device_buffer, byte_pattern, static_cast<std::size_t>(bytes),
      reinterpret_cast<cudaStream_t>(stream)),
      "cudaMemsetAsync DeepSeek NCCL warm-up payload");
}

Result<Sha256Digest> NvidiaDeepSeekNcclWarmupPayloadOperations::digest(
    const void* device_buffer, std::uint64_t bytes) {
  if (device_buffer == nullptr || bytes == 0 ||
      bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument("DeepSeek NCCL CUDA warm-up digest is invalid");
  }
  constexpr std::size_t kChunkBytes = 4U * 1024U * 1024U;
  std::vector<std::byte> host(
      static_cast<std::size_t>(std::min<std::uint64_t>(bytes, kChunkBytes)));
  Sha256 hash;
  std::uint64_t offset = 0;
  while (offset < bytes) {
    const auto count = static_cast<std::size_t>(
        std::min<std::uint64_t>(host.size(), bytes - offset));
    const auto* source = static_cast<const std::byte*>(device_buffer) + offset;
    auto status = cuda_status(cudaMemcpy(host.data(), source, count,
                                         cudaMemcpyDeviceToHost),
                              "cudaMemcpy DeepSeek NCCL warm-up digest D2H");
    if (!status.ok()) return status;
    status = hash.update(std::span<const std::byte>(host.data(), count));
    if (!status.ok()) return status;
    offset += count;
  }
  return hash.finalize();
}

}  // namespace pih
