#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

struct CanonicalExtentVerificationPlan final {
  std::string identity;
  std::uint64_t file_offset;
  std::uint64_t logical_bytes;
  std::uint64_t extent_bytes;
  Sha256Digest logical_sha256;
};

struct CanonicalExtentVerificationReceipt final {
  std::uint64_t file_bytes;
  std::size_t payload_record_count;
  Sha256Digest file_sha256;
};

Result<CanonicalExtentVerificationReceipt> verify_canonical_extent_file(
    std::span<const std::byte> file, std::uint64_t metadata_region_bytes,
    std::span<const CanonicalExtentVerificationPlan> extents);

}  // namespace pih
