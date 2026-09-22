#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "pih/core/sha256.h"

namespace pih {

class CanonicalHashBuilder final {
 public:
  static Result<CanonicalHashBuilder> Create(std::string_view domain,
                                              std::uint32_t field_count);
  Status add_u32(std::uint16_t field_id, std::uint32_t value);
  Status add_u64(std::uint16_t field_id, std::uint64_t value);
  Status add_bytes(std::uint16_t field_id,
                   std::span<const std::byte> value);
  Status add_hash(std::uint16_t field_id, const Sha256Digest& value);
  Status add_ascii_utf8(std::uint16_t field_id, std::string_view value);
  Result<Sha256Digest> finalize();

 private:
  Status add_header(std::uint16_t field_id, std::uint8_t type,
                    std::uint64_t length);
  Status add_payload(std::span<const std::byte> payload);

  Sha256 hash_;
  std::uint32_t expected_fields_ = 0;
  std::uint32_t added_fields_ = 0;
  std::uint16_t last_field_id_ = 0;
  bool failed_ = false;
  bool finalized_ = false;
};

}  // namespace pih
