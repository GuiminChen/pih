#include "pih/io/canonical_extent_writer.h"

#include <algorithm>
#include <limits>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Status write_all(CanonicalSequentialSink& sink, Sha256& digest,
                 std::span<const std::byte> bytes) {
  while (!bytes.empty()) {
    auto written = sink.write(bytes);
    if (!written.ok()) return written.status();
    if (*written == 0 || *written > bytes.size()) {
      return Status::Internal("canonical sink made invalid write progress");
    }
    Status status = digest.update(bytes.first(*written));
    if (!status.ok()) return status;
    bytes = bytes.subspan(*written);
  }
  return Status::Ok();
}

Status write_zeros(CanonicalSequentialSink& sink, Sha256& digest,
                   std::span<std::byte> workspace, std::uint64_t count) {
  std::fill(workspace.begin(), workspace.end(), std::byte{0});
  while (count != 0) {
    const auto chunk = static_cast<std::size_t>(
        std::min<std::uint64_t>(count, workspace.size()));
    Status status = write_all(sink, digest, workspace.first(chunk));
    if (!status.ok()) return status;
    count -= chunk;
  }
  return Status::Ok();
}

}  // namespace

Result<CanonicalExtentWriteReceipt> write_canonical_extents(
    std::span<const std::byte> metadata, std::uint64_t metadata_region_bytes,
    std::span<const CanonicalExtentWritePlan> extents,
    CanonicalPayloadReader& reader, CanonicalSequentialSink& sink,
    std::size_t maximum_workspace_bytes) {
  if (maximum_workspace_bytes == 0 ||
      maximum_workspace_bytes > kMaximumCanonicalWriterWorkspaceBytes ||
      metadata.size() > metadata_region_bytes) {
    return Status::InvalidArgument("canonical writer bounds are invalid");
  }
  if (metadata_region_bytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted("canonical metadata exceeds address space");
  }
  std::uint64_t expected_offset = metadata_region_bytes;
  std::string_view previous_identity;
  for (const auto& extent : extents) {
    if (extent.identity.empty() ||
        (!previous_identity.empty() && previous_identity >= extent.identity) ||
        extent.file_offset != expected_offset ||
        extent.logical_bytes == 0 || extent.logical_bytes > extent.extent_bytes) {
      return Status::InvalidArgument("canonical extent plan is not canonical");
    }
    auto next = checked_add_u64(expected_offset, extent.extent_bytes);
    if (!next.ok()) return next.status();
    expected_offset = *next;
    previous_identity = extent.identity;
  }

  std::vector<std::byte> workspace(maximum_workspace_bytes, std::byte{0});
  Sha256 file_digest;
  Status status = write_all(sink, file_digest, metadata);
  if (status.ok()) {
    status = write_zeros(sink, file_digest, workspace,
                         metadata_region_bytes - metadata.size());
  }
  for (const auto& extent : extents) {
    std::uint64_t logical_offset = 0;
    while (status.ok() && logical_offset < extent.logical_bytes) {
      const auto request = static_cast<std::size_t>(std::min<std::uint64_t>(
          extent.logical_bytes - logical_offset, workspace.size()));
      auto received = reader.read(extent.identity, logical_offset,
                                  std::span(workspace).first(request));
      if (!received.ok()) {
        status = received.status();
        break;
      }
      if (*received == 0 || *received > request) {
        status = Status::FailedPrecondition(
            "canonical payload reader made invalid progress");
        break;
      }
      status = write_all(sink, file_digest,
                         std::span<const std::byte>(workspace).first(*received));
      logical_offset += *received;
    }
    if (status.ok()) {
      status = write_zeros(sink, file_digest, workspace,
                           extent.extent_bytes - extent.logical_bytes);
    }
  }
  if (!status.ok()) return status;
  status = sink.sync();
  if (!status.ok()) return status;
  auto digest = file_digest.finalize();
  if (!digest.ok()) return digest.status();
  return CanonicalExtentWriteReceipt{expected_offset, *digest, extents.size(),
                                     maximum_workspace_bytes};
}

}  // namespace pih
