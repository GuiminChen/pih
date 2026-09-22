#include "token_input_upload.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <new>

namespace pih::deepseek_v41 {
Status ValidateTokenInputUpload(const TokenInputUploadLaunch& x, std::uint32_t tokens) {
  if (!x.stream || !tokens || tokens > 4096) return Status::InvalidArgument("Token upload stream or count invalid");
  const std::array regions{x.device_ids, x.device_hashes[0], x.device_hashes[1], x.host_ids, x.host_hashes[0], x.host_hashes[1]};
  int device = -1;
  auto error = cudaGetDevice(&device); if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
  for (unsigned i = 0; i < regions.size(); ++i) {
    const auto r = regions[i]; const auto bytes = tokens * (i % 3 == 0 ? 4ULL : 24ULL * 4);
    if (!r.address || r.address % 4 || r.bytes != bytes || r.bytes > std::numeric_limits<std::uintptr_t>::max() - r.address)
      return Status::InvalidArgument("Token upload buffer extent or alignment invalid");
    for (unsigned j = 0; j < i; ++j)
      if (r.address < regions[j].address + regions[j].bytes && regions[j].address < r.address + r.bytes)
        return Status::InvalidArgument("Token upload buffers must be disjoint");
    cudaPointerAttributes attributes{};
    error = cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(r.address));
    if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
    if ((i < 3 && (attributes.type != cudaMemoryTypeDevice || attributes.device != device)) ||
        (i >= 3 && attributes.type != cudaMemoryTypeHost))
      return Status::FailedPrecondition("Token upload requires current-device destinations and pinned host sources");
  }
  return Status::Ok();
}
Status UploadTokenInputs(EngramHashState& hashes, std::uint32_t start,
    std::span<const std::uint32_t> tokens, const TokenInputUploadLaunch& x) {
  if (tokens.empty() || tokens.size() > 4096 || hashes.position() != start)
    return Status::InvalidArgument("Token upload requires contiguous admitted hash input");
  const auto valid = ValidateTokenInputUpload(x, static_cast<unsigned>(tokens.size())); if (!valid.ok()) return valid;
  // Snapshot before writing any caller-owned pinned buffers, including when the
  // input span is itself a view into one of those buffers.
  std::array<std::uint32_t, 4096> snapshot;
  std::copy(tokens.begin(), tokens.end(), snapshot.begin());
  try {
    auto generated = hashes.Append(start, {snapshot.data(), tokens.size()}); if (!generated.ok()) return generated.status();
    std::memcpy(reinterpret_cast<void*>(x.host_ids.address), snapshot.data(), tokens.size_bytes());
    for (unsigned layer = 0; layer < 2; ++layer) {
      auto* destination = reinterpret_cast<std::uint32_t*>(x.host_hashes[layer].address);
      for (std::size_t token = 0; token < tokens.size(); ++token)
        std::memcpy(destination + token * 24, (*generated)[token].ids[layer].data(), 24 * 4);
    }
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Token hash staging allocation failed"); }
  const std::array sources{x.host_ids, x.host_hashes[0], x.host_hashes[1]};
  const std::array destinations{x.device_ids, x.device_hashes[0], x.device_hashes[1]};
  for (unsigned i = 0; i < sources.size(); ++i) {
    const auto error = cudaMemcpyAsync(reinterpret_cast<void*>(destinations[i].address),
        reinterpret_cast<const void*>(sources[i].address), sources[i].bytes, cudaMemcpyHostToDevice,
        reinterpret_cast<cudaStream_t>(x.stream));
    if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
  }
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
