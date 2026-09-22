#include "pih/model/qwen3_int4_artifact_metadata.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>

namespace pih {
namespace {

constexpr std::array<std::byte, 8> kMagic{
    std::byte{'X'}, std::byte{'I'}, std::byte{'N'}, std::byte{'G'},
    std::byte{'W'}, std::byte{'4'}, std::byte{'A'}, std::byte{'1'}};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeaderBytes = 160;
constexpr std::size_t kChecksumOffset = 120;

bool nonzero(const Sha256Digest& digest) { return digest != Sha256Digest{}; }

void append_u16(std::vector<std::byte>& output, std::uint16_t value) {
  output.push_back(static_cast<std::byte>(value));
  output.push_back(static_cast<std::byte>(value >> 8U));
}
void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
  for (std::size_t i = 0; i < 4; ++i) output.push_back(static_cast<std::byte>(value >> (i * 8U)));
}
void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i) output.push_back(static_cast<std::byte>(value >> (i * 8U)));
}
void append_string(std::vector<std::byte>& output, std::string_view value) {
  output.insert(output.end(), reinterpret_cast<const std::byte*>(value.data()),
                reinterpret_cast<const std::byte*>(value.data() + value.size()));
}

class Cursor final {
 public:
  explicit Cursor(std::span<const std::byte> input) : input_(input) {}
  Result<std::uint8_t> u8() {
    if (position_ == input_.size()) return Status::InvalidArgument("metadata truncated");
    return std::to_integer<std::uint8_t>(input_[position_++]);
  }
  Result<std::uint16_t> u16() {
    auto bytes = take(2); if (!bytes.ok()) return bytes.status();
    return std::to_integer<std::uint16_t>((*bytes)[0]) |
           (std::to_integer<std::uint16_t>((*bytes)[1]) << 8U);
  }
  Result<std::uint32_t> u32() {
    auto bytes = take(4); if (!bytes.ok()) return bytes.status();
    std::uint32_t value = 0; for (std::size_t i=0;i<4;++i) value |= std::to_integer<std::uint32_t>((*bytes)[i]) << (i*8U); return value;
  }
  Result<std::uint64_t> u64() {
    auto bytes = take(8); if (!bytes.ok()) return bytes.status();
    std::uint64_t value = 0; for (std::size_t i=0;i<8;++i) value |= std::to_integer<std::uint64_t>((*bytes)[i]) << (i*8U); return value;
  }
  Result<std::span<const std::byte>> take(std::size_t count) {
    if (count > input_.size() - position_) return Status::InvalidArgument("metadata truncated");
    auto result = input_.subspan(position_, count); position_ += count; return result;
  }
  Result<std::string> string(std::size_t count) {
    auto bytes = take(count); if (!bytes.ok()) return bytes.status();
    return std::string(reinterpret_cast<const char*>(bytes->data()), bytes->size());
  }
  std::size_t position() const { return position_; }
 private:
  std::span<const std::byte> input_; std::size_t position_ = 0;
};

Result<Sha256Digest> read_digest(Cursor& cursor) {
  auto bytes = cursor.take(32); if (!bytes.ok()) return bytes.status();
  Sha256Digest digest; std::copy(bytes->begin(), bytes->end(), digest.bytes.begin()); return digest;
}

bool same_layout(const QwenInt4ArtifactRecordPlan& a,
                 const QwenInt4ArtifactRecordPlan& b) {
  return a.identity == b.identity && a.kind == b.kind &&
         a.source_name == b.source_name && a.alias_owner == b.alias_owner &&
         a.logical_bytes == b.logical_bytes && a.file_offset == b.file_offset &&
         a.extent_bytes == b.extent_bytes;
}

}  // namespace

Result<QwenInt4ArtifactMetadata> QwenInt4ArtifactMetadata::Create(
    const QwenInt4ArtifactLayout& layout, QwenInt4ArtifactRoots roots,
    std::vector<QwenInt4PayloadDigest> payload_digests) {
  if (!nonzero(roots.source_artifact) || !nonzero(roots.source_binding) ||
      !nonzero(roots.disposition)) {
    return Status::InvalidArgument("Qwen INT4 artifact root is empty");
  }
  std::sort(payload_digests.begin(), payload_digests.end(), [](const auto& a, const auto& b) { return a.identity < b.identity; });
  std::map<std::string, Sha256Digest> digests;
  for (const auto& item : payload_digests) {
    if (!nonzero(item.sha256) || !digests.emplace(item.identity, item.sha256).second)
      return Status::InvalidArgument("Qwen INT4 payload digest is invalid");
  }
  std::vector<QwenInt4ArtifactMetadataRecord> records;
  records.reserve(layout.records().size());
  std::size_t consumed = 0;
  for (const auto& record : layout.records()) {
    Sha256Digest digest{};
    if (record.kind != QwenInt4ArtifactRecordKind::kAlias) {
      const auto found = digests.find(record.identity);
      if (found == digests.end()) return Status::InvalidArgument("Qwen INT4 payload digest is missing");
      digest = found->second; ++consumed;
    }
    records.push_back({record, digest});
  }
  if (consumed != digests.size() || consumed != layout.payload_record_count())
    return Status::InvalidArgument("Qwen INT4 payload digest coverage mismatch");
  QwenInt4ArtifactMetadata result(roots, std::move(records), 0);
  auto encoded = result.serialize_fixed_region();
  if (!encoded.ok()) return encoded.status();
  Cursor cursor(*encoded);
  auto prefix = cursor.take(12);
  if (!prefix.ok()) return prefix.status();
  auto size = cursor.u32();
  if (!size.ok()) return size.status();
  result.encoded_bytes_ = *size;
  return result;
}

Result<std::vector<std::byte>> QwenInt4ArtifactMetadata::serialize_fixed_region() const {
  std::vector<std::byte> output;
  output.reserve(QwenInt4ArtifactLayout::kMetadataBytes);
  output.insert(output.end(), kMagic.begin(), kMagic.end());
  append_u32(output, kVersion); append_u32(output, 0);
  append_u32(output, static_cast<std::uint32_t>(records_.size())); append_u32(output, 0);
  for (const auto& root : {roots_.source_artifact, roots_.source_binding, roots_.disposition}) output.insert(output.end(), root.bytes.begin(), root.bytes.end());
  output.resize(output.size() + 32, std::byte{0});
  append_u16(output, static_cast<std::uint16_t>(kFormatId.size()));
  append_u16(output, static_cast<std::uint16_t>(kLayoutId.size())); append_u32(output, 0);
  append_string(output, kFormatId); append_string(output, kLayoutId);
  for (const auto& record : records_) {
    const auto& layout = record.layout;
    if (layout.identity.size() > 65535 || layout.source_name.size() > 65535 || layout.alias_owner.size() > 65535)
      return Status::ResourceExhausted("Qwen INT4 metadata string is too long");
    output.push_back(static_cast<std::byte>(layout.kind)); output.push_back(std::byte{0});
    append_u16(output, static_cast<std::uint16_t>(layout.identity.size()));
    append_u16(output, static_cast<std::uint16_t>(layout.source_name.size()));
    append_u16(output, static_cast<std::uint16_t>(layout.alias_owner.size()));
    append_u64(output, layout.logical_bytes); append_u64(output, layout.file_offset); append_u64(output, layout.extent_bytes);
    output.insert(output.end(), record.payload_sha256.bytes.begin(), record.payload_sha256.bytes.end());
    append_string(output, layout.identity); append_string(output, layout.source_name); append_string(output, layout.alias_owner);
  }
  if (output.size() > QwenInt4ArtifactLayout::kMetadataBytes || output.size() > std::numeric_limits<std::uint32_t>::max())
    return Status::ResourceExhausted("Qwen INT4 metadata exceeds fixed region");
  const auto encoded = static_cast<std::uint32_t>(output.size());
  for (std::size_t i=0;i<4;++i) output[12+i] = static_cast<std::byte>(encoded >> (i*8U));
  auto checksum_input = output;
  auto checksum = sha256(checksum_input); if (!checksum.ok()) return checksum.status();
  std::copy(checksum->bytes.begin(), checksum->bytes.end(), output.begin() + kChecksumOffset);
  output.resize(QwenInt4ArtifactLayout::kMetadataBytes, std::byte{0});
  return output;
}

Result<QwenInt4ArtifactMetadata> QwenInt4ArtifactMetadata::ParseAndVerify(
    std::span<const std::byte> fixed_region,
    const QwenInt4ArtifactLayout& expected_layout) {
  if (fixed_region.size() != QwenInt4ArtifactLayout::kMetadataBytes)
    return Status::InvalidArgument("Qwen INT4 metadata region size mismatch");
  Cursor header(fixed_region);
  auto magic = header.take(8); if (!magic.ok() || !std::equal(magic->begin(), magic->end(), kMagic.begin())) return Status::InvalidArgument("Qwen INT4 metadata magic mismatch");
  auto version = header.u32(); auto encoded = header.u32(); auto count = header.u32(); auto reserved = header.u32();
  if (!version.ok() || !encoded.ok() || !count.ok() || !reserved.ok() || *version != kVersion || *reserved != 0 || *encoded < kHeaderBytes || *encoded > fixed_region.size() || *count != expected_layout.records().size())
    return Status::InvalidArgument("Qwen INT4 metadata header mismatch");
  for (std::size_t i=*encoded;i<fixed_region.size();++i) if (fixed_region[i] != std::byte{0}) return Status::InvalidArgument("Qwen INT4 metadata tail is not zero");
  auto checksum_wire = std::vector<std::byte>(fixed_region.begin(), fixed_region.begin()+*encoded);
  Sha256Digest stored; std::copy(checksum_wire.begin()+kChecksumOffset, checksum_wire.begin()+kChecksumOffset+32, stored.bytes.begin());
  std::fill(checksum_wire.begin()+kChecksumOffset, checksum_wire.begin()+kChecksumOffset+32, std::byte{0});
  auto computed = sha256(checksum_wire); if (!computed.ok() || *computed != stored) return Status::FailedPrecondition("Qwen INT4 metadata checksum mismatch");
  Cursor cursor(fixed_region.first(*encoded));
  auto fixed_header = cursor.take(24);
  if (!fixed_header.ok()) return fixed_header.status();
  auto source = read_digest(cursor); auto binding = read_digest(cursor); auto disposition = read_digest(cursor); auto ignored_checksum = read_digest(cursor);
  auto format_len = cursor.u16(); auto layout_len = cursor.u16(); auto reserved2 = cursor.u32();
  if (!source.ok() || !binding.ok() || !disposition.ok() || !ignored_checksum.ok() || !format_len.ok() || !layout_len.ok() || !reserved2.ok() || *reserved2 != 0) return Status::InvalidArgument("Qwen INT4 metadata roots truncated");
  auto format = cursor.string(*format_len); auto layout_id = cursor.string(*layout_len);
  if (!format.ok() || !layout_id.ok() || *format != kFormatId || *layout_id != kLayoutId) return Status::InvalidArgument("Qwen INT4 metadata format mismatch");
  std::vector<QwenInt4ArtifactMetadataRecord> records; records.reserve(*count);
  for (std::size_t index=0; index<*count; ++index) {
    auto kind=cursor.u8(); auto zero=cursor.u8(); auto il=cursor.u16(); auto sl=cursor.u16(); auto al=cursor.u16(); auto logical=cursor.u64(); auto offset=cursor.u64(); auto extent=cursor.u64(); auto digest=read_digest(cursor);
    if (!kind.ok()||!zero.ok()||!il.ok()||!sl.ok()||!al.ok()||!logical.ok()||!offset.ok()||!extent.ok()||!digest.ok()||*zero!=0) return Status::InvalidArgument("Qwen INT4 metadata record truncated");
    auto identity=cursor.string(*il); auto source_name=cursor.string(*sl); auto alias=cursor.string(*al);
    if (!identity.ok()||!source_name.ok()||!alias.ok()) return Status::InvalidArgument("Qwen INT4 metadata record string truncated");
    QwenInt4ArtifactRecordPlan parsed{*identity, static_cast<QwenInt4ArtifactRecordKind>(*kind), *source_name, *alias, *logical, *offset, *extent};
    if (!same_layout(parsed, expected_layout.records()[index]) || (parsed.kind == QwenInt4ArtifactRecordKind::kAlias ? nonzero(*digest) : !nonzero(*digest))) return Status::FailedPrecondition("Qwen INT4 metadata record mismatch");
    records.push_back({std::move(parsed), *digest});
  }
  if (cursor.position() != *encoded) return Status::InvalidArgument("Qwen INT4 metadata has trailing encoded bytes");
  QwenInt4ArtifactRoots roots{*source,*binding,*disposition};
  if (!nonzero(roots.source_artifact)||!nonzero(roots.source_binding)||!nonzero(roots.disposition)) return Status::FailedPrecondition("Qwen INT4 metadata root is empty");
  return QwenInt4ArtifactMetadata(roots, std::move(records), *encoded);
}

}  // namespace pih
