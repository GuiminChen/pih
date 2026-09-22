#pragma once
#include <cstdint>
#include <span>
#include <string_view>
#include "pih/core/sha256.h"

namespace pih::offline_deepseek {
// Borrowed descriptors must stay open for the entire call. The caller admits
// source provenance, tensor geometry and layout; this layer verifies byte ranges.
struct CopyRange final {
  int source_fd;
  std::uint64_t begin;
  std::uint64_t bytes;
  Sha256Digest payload_sha256;
};
struct CopyReceipt final {
  std::uint64_t file_bytes;
  Sha256Digest file_sha256;
};
// Creates a new member beneath an already-open staging directory. Never replaces
// a member. Failure retains the partial file for explicit caller cleanup; only
// success yields a receipt. No rename/publication or immutable admission occurs.
Result<CopyReceipt> CopyIdentityShard(
    int staging_directory_fd, std::string_view member_name,
    std::span<const std::byte> header_prefix, std::span<const CopyRange> ranges);
// Bounded metadata only (128 MiB maximum). Verifies input and readback against
// the expected digest; exclusive creation, fsync, no overwrite or publication.
Result<CopyReceipt> WriteMetadataMember(int staging_directory_fd,
    std::string_view member_name, std::string_view contents,
    const Sha256Digest& expected_sha256);
}  // namespace pih::offline_deepseek
