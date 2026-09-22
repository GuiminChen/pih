#include "pih/io/canonical_extent_verifier.h"

#include <algorithm>
#include <limits>
#include <string_view>

#include "pih/core/checked_math.h"

namespace pih {

Result<CanonicalExtentVerificationReceipt> verify_canonical_extent_file(
    std::span<const std::byte> file, std::uint64_t metadata_region_bytes,
    std::span<const CanonicalExtentVerificationPlan> extents) {
  if (metadata_region_bytes > file.size()) {
    return Status::FailedPrecondition("canonical file metadata is truncated");
  }
  std::uint64_t expected_offset = metadata_region_bytes;
  std::string_view previous_identity;
  for (const auto& extent : extents) {
    if (extent.identity.empty() ||
        (!previous_identity.empty() && previous_identity >= extent.identity) ||
        extent.file_offset != expected_offset || extent.logical_bytes == 0 ||
        extent.logical_bytes > extent.extent_bytes ||
        extent.logical_sha256 == Sha256Digest{}) {
      return Status::InvalidArgument("canonical verification plan is invalid");
    }
    auto next = checked_add_u64(expected_offset, extent.extent_bytes);
    if (!next.ok()) return next.status();
    if (*next > file.size() || extent.file_offset >
                                 static_cast<std::uint64_t>(
                                     std::numeric_limits<std::size_t>::max())) {
      return Status::FailedPrecondition("canonical extent is truncated");
    }
    const auto begin = static_cast<std::size_t>(extent.file_offset);
    const auto logical = static_cast<std::size_t>(extent.logical_bytes);
    const auto padded = static_cast<std::size_t>(extent.extent_bytes);
    auto observed = sha256(file.subspan(begin, logical));
    if (!observed.ok()) return observed.status();
    if (*observed != extent.logical_sha256) {
      return Status::FailedPrecondition("canonical payload checksum mismatch");
    }
    if (!std::all_of(file.begin() + begin + logical,
                     file.begin() + begin + padded,
                     [](std::byte value) { return value == std::byte{0}; })) {
      return Status::FailedPrecondition("canonical extent padding is not zero");
    }
    expected_offset = *next;
    previous_identity = extent.identity;
  }
  if (expected_offset != file.size()) {
    return Status::FailedPrecondition("canonical file has trailing bytes");
  }
  auto digest = sha256(file);
  if (!digest.ok()) return digest.status();
  return CanonicalExtentVerificationReceipt{
      static_cast<std::uint64_t>(file.size()), extents.size(), *digest};
}

}  // namespace pih
