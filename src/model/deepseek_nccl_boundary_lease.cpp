#include "pih/model/deepseek_nccl_boundary_lease.h"

#include <cstddef>
#include <limits>

namespace pih {

Result<DeepSeekNcclBoundaryLease> DeepSeekNcclBoundaryLease::Create(
    const DeepSeekNcclP2pManifest& manifest, Buffer& buffer,
    std::uint64_t owner_id, std::uintptr_t context_identity,
    std::uintptr_t boundary_stream) {
  if (owner_id == 0 || context_identity == 0 || boundary_stream == 0 ||
      context_identity != manifest.context_identity ||
      owner_id != manifest.buffer_owner_id ||
      buffer.generation() != manifest.buffer_generation ||
      buffer.device().type() != DeviceType::kCuda ||
      buffer.device().index() != static_cast<std::int32_t>(manifest.local_global_rank) ||
      buffer.size_bytes() != manifest.buffer_capacity_bytes ||
      manifest.token_count == 0) {
    return Status::InvalidArgument("DeepSeek NCCL boundary buffer identity is invalid");
  }
  const std::uint64_t bytes = std::uint64_t{manifest.token_count} * 32768U;
  if (manifest.buffer_offset_bytes > buffer.size_bytes() ||
      bytes > buffer.size_bytes() - manifest.buffer_offset_bytes ||
      reinterpret_cast<std::uintptr_t>(buffer.data()) >
          std::numeric_limits<std::uintptr_t>::max() - manifest.buffer_offset_bytes) {
    return Status::InvalidArgument("DeepSeek NCCL boundary buffer span is invalid");
  }
  auto* data = static_cast<std::byte*>(buffer.data()) + manifest.buffer_offset_bytes;
  if (reinterpret_cast<std::uintptr_t>(data) % 256U != 0) {
    return Status::InvalidArgument("DeepSeek NCCL boundary buffer is not 256-byte aligned");
  }
  return DeepSeekNcclBoundaryLease(manifest.role, data, bytes, boundary_stream);
}

Status DeepSeekNcclBoundaryLease::bind(
    DeepSeekNcclP2pBindingTarget& target) const {
  return target.bind_p2p(role_, data_, bytes_, stream_);
}

}  // namespace pih
