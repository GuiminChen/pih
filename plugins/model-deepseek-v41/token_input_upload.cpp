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
Status UploadPackedTokenInputs(const PackedRequest& plan,
    std::span<EngramHashState* const> states, const TokenInputUploadLaunch& x) {
  const auto valid = ValidateTokenInputUpload(x, static_cast<unsigned>(plan.tokens().size()));
  if (!valid.ok()) return valid;
  for (const auto& member : plan.spans()) {
    if (member.cache_slot >= states.size() || !states[member.cache_slot] || member.finish ||
        states[member.cache_slot]->position() != member.position)
      return Status::InvalidArgument("Packed token upload hash position or slot invalid");
    for (const auto& other : plan.spans())
      if (other.cache_slot != member.cache_slot && other.cache_slot < states.size() &&
          states[other.cache_slot] == states[member.cache_slot])
        return Status::InvalidArgument("Packed Engram states alias across sequences");
  }
  try {
    std::vector<EngramTokenHashes> generated(plan.tokens().size());
    for (const auto& member : plan.spans()) {
      auto rows = states[member.cache_slot]->Append(member.position,
          plan.tokens().subspan(member.token_offset, member.token_count));
      if (!rows.ok()) return rows.status();
      std::copy(rows->begin(), rows->end(), generated.begin() + member.token_offset);
    }
    std::memcpy(reinterpret_cast<void*>(x.host_ids.address), plan.tokens().data(), plan.tokens().size_bytes());
    for (unsigned layer = 0; layer < 2; ++layer) {
      auto* destination = reinterpret_cast<std::uint32_t*>(x.host_hashes[layer].address);
      for (std::size_t token = 0; token < generated.size(); ++token)
        std::memcpy(destination + token * 24, generated[token].ids[layer].data(), 24 * 4);
    }
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Packed token hash staging allocation failed"); }
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
Status UploadPackedHeadRows(const PackedRequest& plan, EngramDeviceRegion host,
    EngramDeviceRegion device, std::uintptr_t stream) {
  const auto bytes = plan.spans().size() * 4ULL;
  if (!stream || !bytes || !host.address || host.address % 4 || host.bytes != bytes ||
      !device.address || device.address % 4 || device.bytes != bytes ||
      bytes > std::numeric_limits<std::uintptr_t>::max() - host.address ||
      bytes > std::numeric_limits<std::uintptr_t>::max() - device.address ||
      (host.address < device.address + bytes && device.address < host.address + bytes))
    return Status::InvalidArgument("Packed head row table extent invalid");
  int current_device = -1;
  auto error = cudaGetDevice(&current_device); if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
  cudaPointerAttributes host_attributes{}, device_attributes{};
  error = cudaPointerGetAttributes(&host_attributes, reinterpret_cast<void*>(host.address));
  if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
  error = cudaPointerGetAttributes(&device_attributes, reinterpret_cast<void*>(device.address));
  if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
  if (host_attributes.type != cudaMemoryTypeHost || device_attributes.type != cudaMemoryTypeDevice ||
      device_attributes.device != current_device)
    return Status::FailedPrecondition("Packed head row table requires pinned host and current-device storage");
  auto rows = plan.LastTokenRows(); if (!rows.ok()) return rows.status();
  std::memcpy(reinterpret_cast<void*>(host.address), rows->data(), bytes);
  error = cudaMemcpyAsync(reinterpret_cast<void*>(device.address), reinterpret_cast<const void*>(host.address),
      bytes, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream));
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
}  // namespace pih::deepseek_v41
