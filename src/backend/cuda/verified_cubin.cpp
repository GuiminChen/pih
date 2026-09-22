#include "pih/backend/cuda/verified_cubin.h"

#include <algorithm>
#include <limits>

#include "pih/io/mapped_file.h"

namespace pih {
namespace {

constexpr std::size_t kElf64HeaderBytes = 64;

bool has_cuda_elf64_header(std::span<const std::byte> bytes) noexcept {
  if (bytes.size() < kElf64HeaderBytes) return false;
  return bytes[0] == std::byte{0x7f} && bytes[1] == std::byte{'E'} &&
         bytes[2] == std::byte{'L'} && bytes[3] == std::byte{'F'} &&
         bytes[4] == std::byte{2} && bytes[5] == std::byte{1} &&
         bytes[6] == std::byte{1} && bytes[18] == std::byte{0xbe} &&
         bytes[19] == std::byte{0} && bytes[20] == std::byte{1} &&
         bytes[21] == std::byte{0} && bytes[22] == std::byte{0} &&
         bytes[23] == std::byte{0};
}

}  // namespace

Result<VerifiedCubin> VerifiedCubin::Load(
    const std::filesystem::path& path, std::uint64_t maximum_bytes,
    const Sha256Digest& expected_digest) {
  if (maximum_bytes < kElf64HeaderBytes) {
    return Status::InvalidArgument("cubin byte budget is smaller than ELF64 header");
  }
  auto mapped = MappedFile::OpenReadOnly(path, maximum_bytes);
  if (!mapped.ok()) return mapped.status();
  if (mapped->size_bytes() < kElf64HeaderBytes) {
    return Status::InvalidArgument("cubin is smaller than ELF64 header");
  }
  if (mapped->size_bytes() >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted("cubin exceeds address space");
  }

  std::vector<std::byte> owned(static_cast<std::size_t>(mapped->size_bytes()));
  std::copy_n(mapped->data(), owned.size(), owned.data());
  const std::span<const std::byte> bytes = owned;
  if (!has_cuda_elf64_header(bytes)) {
    return Status::InvalidArgument("cubin is not CUDA ELF64 little-endian code");
  }
  auto digest = sha256(bytes);
  if (!digest.ok()) return digest.status();
  if (digest.value() != expected_digest) {
    return Status::InvalidArgument("cubin SHA-256 does not match manifest");
  }
  return VerifiedCubin(std::move(owned), digest.value());
}

Result<VerifiedCubin> VerifiedCubin::Load(
    const std::filesystem::path& path, std::uint64_t maximum_bytes,
    std::string_view expected_sha256) {
  auto digest = Sha256Digest::ParseHex(expected_sha256);
  if (!digest.ok()) return digest.status();
  return Load(path, maximum_bytes, digest.value());
}

}  // namespace pih
