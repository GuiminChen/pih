#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

struct CanonicalExtentWritePlan final {
  std::string identity;
  std::uint64_t file_offset;
  std::uint64_t logical_bytes;
  std::uint64_t extent_bytes;
};

class CanonicalPayloadReader {
 public:
  virtual ~CanonicalPayloadReader() = default;
  virtual Result<std::size_t> read(std::string_view identity,
                                   std::uint64_t logical_offset,
                                   std::span<std::byte> output) = 0;
};

class CanonicalSequentialSink {
 public:
  virtual ~CanonicalSequentialSink() = default;
  virtual Result<std::size_t> write(std::span<const std::byte> input) = 0;
  virtual Status sync() = 0;
};

struct CanonicalExtentWriteReceipt final {
  std::uint64_t file_bytes;
  Sha256Digest file_sha256;
  std::size_t payload_record_count;
  std::size_t maximum_workspace_bytes;
};

inline constexpr std::size_t kMaximumCanonicalWriterWorkspaceBytes =
    1024U * 1024U;

Result<CanonicalExtentWriteReceipt> write_canonical_extents(
    std::span<const std::byte> metadata, std::uint64_t metadata_region_bytes,
    std::span<const CanonicalExtentWritePlan> extents,
    CanonicalPayloadReader& reader, CanonicalSequentialSink& sink,
    std::size_t maximum_workspace_bytes);

}  // namespace pih
