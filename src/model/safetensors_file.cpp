#include "pih/model/safetensors_file.h"

#include <algorithm>
#include <array>
#include <limits>
#include <fstream>
#include <vector>
#include <utility>

namespace pih {

Result<SafetensorsFile> SafetensorsFile::Open(
    const std::filesystem::path& path, std::uint64_t maximum_file_bytes) {
  auto mapping = MappedFile::OpenReadOnly(path, maximum_file_bytes);
  if (!mapping.ok()) return mapping.status();
  if (mapping->size_bytes() >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted("Safetensors mapping exceeds address space");
  }
  const auto prefix = std::span<const std::byte>(
      mapping->data(), static_cast<std::size_t>(mapping->size_bytes()));
  auto header = SafetensorsHeader::ParsePrefix(prefix, mapping->size_bytes());
  if (!header.ok()) return header.status();
  return SafetensorsFile(std::move(mapping).value(), std::move(header).value());
}

Result<SafetensorsFile> SafetensorsFile::Borrow(std::span<const std::byte> bytes) {
  if (bytes.empty()) return Status::InvalidArgument("Safetensors borrowed bytes are empty");
  auto header = SafetensorsHeader::ParsePrefix(bytes, bytes.size());
  if (!header.ok()) return header.status();
  return SafetensorsFile(bytes, std::move(*header));
}

Result<ImmutableTensorBytes> SafetensorsFile::tensor(std::string_view name) const {
  const auto* record = header_.tensor(name);
  if (record == nullptr) {
    return Status::InvalidArgument("Safetensors tensor does not exist");
  }
  if (record->file_end < record->file_begin || record->file_end > bytes_.size())
    return Status::InvalidArgument("Safetensors tensor range exceeds file");
  return ImmutableTensorBytes{record->dtype, record->shape,
      bytes_.subspan(static_cast<std::size_t>(record->file_begin),
                     static_cast<std::size_t>(record->file_end - record->file_begin))};
}

Result<SafetensorsHeaderFileReceipt> load_safetensors_header_file(
    const std::filesystem::path& path, std::uint64_t maximum_file_bytes) {
  if (path.empty() || maximum_file_bytes < 8)
    return Status::InvalidArgument("Safetensors header file request is invalid");
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return Status::Unavailable("cannot open Safetensors header file");
  const auto end = stream.tellg();
  if (end < 8 || static_cast<std::uint64_t>(end) > maximum_file_bytes)
    return Status::ResourceExhausted("Safetensors file size is outside budget");
  const auto file_bytes = static_cast<std::uint64_t>(end);
  stream.seekg(0);
  std::array<std::byte, 8> length_prefix{};
  if (!stream.read(reinterpret_cast<char*>(length_prefix.data()),
                   static_cast<std::streamsize>(length_prefix.size()))) {
    return Status::Unavailable("cannot read Safetensors length prefix");
  }
  std::uint64_t header_bytes = 0;
  for (std::size_t index = 0; index < length_prefix.size(); ++index) {
    header_bytes |= static_cast<std::uint64_t>(
        std::to_integer<unsigned char>(length_prefix[index])) << (index * 8U);
  }
  if (header_bytes == 0 || header_bytes > SafetensorsHeader::kMaxHeaderBytes ||
      header_bytes > file_bytes - 8) {
    return Status::ResourceExhausted(
        "Safetensors header length is outside budget");
  }
  std::vector<std::byte> prefix(static_cast<std::size_t>(header_bytes + 8));
  std::copy(length_prefix.begin(), length_prefix.end(), prefix.begin());
  if (!stream.read(reinterpret_cast<char*>(prefix.data() + 8),
                   static_cast<std::streamsize>(header_bytes))) {
    return Status::Unavailable("cannot read complete Safetensors header");
  }
  auto header = SafetensorsHeader::ParsePrefix(prefix, file_bytes);
  if (!header.ok()) return header.status();
  auto digest = sha256(prefix);
  if (!digest.ok()) return digest.status();
  return SafetensorsHeaderFileReceipt{
      std::move(*header), file_bytes, header_bytes + 8, *digest};
}

Result<SafetensorsHeaderFileReceipt> load_safetensors_header_descriptor(
    const ControllerFileLease& lease) {
  const auto file_bytes = lease.identity().file_bytes;
  if (file_bytes < 8) {
    return Status::InvalidArgument(
        "Safetensors descriptor is shorter than length prefix");
  }
  std::array<std::byte, 8> length_prefix{};
  auto read = lease.read_exact(0, length_prefix);
  if (!read.ok()) return read;
  std::uint64_t header_bytes = 0;
  for (std::size_t index = 0; index < length_prefix.size(); ++index) {
    header_bytes |= static_cast<std::uint64_t>(
        std::to_integer<unsigned char>(length_prefix[index])) << (index * 8U);
  }
  if (header_bytes == 0 || header_bytes > SafetensorsHeader::kMaxHeaderBytes ||
      header_bytes > file_bytes - 8) {
    return Status::ResourceExhausted(
        "Safetensors descriptor header length is outside budget");
  }
  std::vector<std::byte> prefix(static_cast<std::size_t>(header_bytes + 8));
  std::copy(length_prefix.begin(), length_prefix.end(), prefix.begin());
  read = lease.read_exact(8, std::span<std::byte>(prefix).subspan(8));
  if (!read.ok()) return read;
  auto header = SafetensorsHeader::ParsePrefix(prefix, file_bytes);
  if (!header.ok()) return header.status();
  auto digest = sha256(prefix);
  if (!digest.ok()) return digest.status();
  return SafetensorsHeaderFileReceipt{
      std::move(*header), file_bytes, header_bytes + 8, *digest};
}

}  // namespace pih
