#include "pih/core/canonical_hash.h"

#include <array>

namespace pih { namespace {
template <typename T>
std::array<std::byte, sizeof(T)> little_endian(T value) {
  std::array<std::byte, sizeof(T)> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<std::byte>(value >> (index * 8U));
  }
  return result;
}
bool valid_domain(std::string_view domain) {
  if (domain.empty() || domain.size() > 127 ||
      !domain.starts_with("pih:")) return false;
  for (unsigned char value : domain) {
    if (value < 0x20 || value > 0x7e) return false;
  }
  return true;
}
}  // namespace pih::<anonymous>

Result<CanonicalHashBuilder> CanonicalHashBuilder::Create(
    std::string_view domain, std::uint32_t field_count) {
  if (!valid_domain(domain)) {
    return Status::InvalidArgument("canonical hash domain is invalid");
  }
  CanonicalHashBuilder value;
  auto status = value.hash_.update(std::as_bytes(std::span(domain)));
  const std::byte nul{0};
  if (status.ok()) status = value.hash_.update({&nul, 1});
  const auto count = little_endian(field_count);
  if (status.ok()) status = value.hash_.update(count);
  if (!status.ok()) return status;
  value.expected_fields_ = field_count;
  return value;
}

Status CanonicalHashBuilder::add_header(std::uint16_t field_id,
                                        std::uint8_t type,
                                        std::uint64_t length) {
  if (failed_ || finalized_ || field_id == 0 ||
      field_id <= last_field_id_ || added_fields_ >= expected_fields_) {
    failed_ = true;
    return Status::FailedPrecondition("canonical hash field order is invalid");
  }
  const auto id = little_endian(field_id);
  const std::byte typed = static_cast<std::byte>(type);
  const auto size = little_endian(length);
  auto status = hash_.update(id);
  if (status.ok()) status = hash_.update({&typed, 1});
  if (status.ok()) status = hash_.update(size);
  if (!status.ok()) { failed_ = true; return status; }
  last_field_id_ = field_id;
  ++added_fields_;
  return Status::Ok();
}

Status CanonicalHashBuilder::add_payload(
    std::span<const std::byte> payload) {
  auto status = hash_.update(payload);
  if (!status.ok()) failed_ = true;
  return status;
}

Status CanonicalHashBuilder::add_u32(std::uint16_t id, std::uint32_t value) {
  auto status = add_header(id, 1, 4);
  if (!status.ok()) return status;
  const auto bytes = little_endian(value);
  return add_payload(bytes);
}
Status CanonicalHashBuilder::add_u64(std::uint16_t id, std::uint64_t value) {
  auto status = add_header(id, 2, 8);
  if (!status.ok()) return status;
  const auto bytes = little_endian(value);
  return add_payload(bytes);
}
Status CanonicalHashBuilder::add_bytes(
    std::uint16_t id, std::span<const std::byte> value) {
  auto status = add_header(id, 3, value.size());
  return status.ok() ? add_payload(value) : status;
}
Status CanonicalHashBuilder::add_hash(std::uint16_t id,
                                      const Sha256Digest& value) {
  bool zero = true;
  for (const auto byte : value.bytes) {
    if (byte != std::byte{0}) { zero = false; break; }
  }
  if (zero) {
    failed_ = true;
    return Status::InvalidArgument(
        "zero hash is not a canonical content digest");
  }
  auto status = add_header(id, 4, value.bytes.size());
  return status.ok() ? add_payload(value.bytes) : status;
}
Status CanonicalHashBuilder::add_ascii_utf8(std::uint16_t id,
                                             std::string_view value) {
  for (unsigned char byte : value) {
    if (byte == 0 || byte > 0x7f) {
      failed_ = true;
      return Status::InvalidArgument(
          "canonical UTF-8 requires the NFC implementation");
    }
  }
  auto status = add_header(id, 5, value.size());
  return status.ok()
      ? add_payload(std::as_bytes(std::span(value))) : status;
}
Result<Sha256Digest> CanonicalHashBuilder::finalize() {
  if (failed_ || finalized_ || added_fields_ != expected_fields_) {
    failed_ = true;
    return Status::FailedPrecondition("canonical hash object is incomplete");
  }
  finalized_ = true;
  return hash_.finalize();
}

}  // namespace pih
