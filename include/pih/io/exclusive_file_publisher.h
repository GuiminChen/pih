#pragma once

#include <cstdint>
#include <filesystem>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

struct ExclusiveFilePublicationReceipt final {
  std::filesystem::path published_path;
  std::uint64_t file_bytes;
  Sha256Digest file_sha256;
};

Result<ExclusiveFilePublicationReceipt> publish_verified_file_exclusive(
    const std::filesystem::path& staging_path,
    const std::filesystem::path& published_path,
    std::uint64_t expected_file_bytes,
    const Sha256Digest& expected_file_sha256);

}  // namespace pih
