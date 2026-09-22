#include "pih/io/exclusive_file_publisher.h"

#include <system_error>

#include "pih/io/mapped_file.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace pih {

Result<ExclusiveFilePublicationReceipt> publish_verified_file_exclusive(
    const std::filesystem::path& staging_path,
    const std::filesystem::path& published_path,
    std::uint64_t expected_file_bytes,
    const Sha256Digest& expected_file_sha256) {
  if (expected_file_bytes == 0 || expected_file_sha256 == Sha256Digest{} ||
      staging_path.empty() || published_path.empty() ||
      staging_path == published_path ||
      staging_path.parent_path() != published_path.parent_path()) {
    return Status::InvalidArgument("exclusive publication contract is invalid");
  }
  Sha256Digest observed_digest;
  {
    auto staged = MappedFile::OpenReadOnly(staging_path, expected_file_bytes);
    if (!staged.ok()) return staged.status();
    if (staged->size_bytes() != expected_file_bytes) {
      return Status::FailedPrecondition("staged publication length mismatch");
    }
    auto observed = sha256(std::span<const std::byte>(
        staged->data(), static_cast<std::size_t>(staged->size_bytes())));
    if (!observed.ok()) return observed.status();
    if (*observed != expected_file_sha256) {
      return Status::FailedPrecondition("staged publication checksum mismatch");
    }
    observed_digest = *observed;
  }

#ifdef _WIN32
  if (!MoveFileExW(staging_path.c_str(), published_path.c_str(),
                   MOVEFILE_WRITE_THROUGH)) {
    return Status::FailedPrecondition("exclusive publication rename failed");
  }
#else
  if (::link(staging_path.c_str(), published_path.c_str()) != 0) {
    return Status::FailedPrecondition("exclusive publication link failed");
  }
  if (::unlink(staging_path.c_str()) != 0) {
    return Status::Internal("exclusive publication staging unlink failed");
  }
  const int directory = ::open(published_path.parent_path().c_str(),
                               O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0 || ::fsync(directory) != 0) {
    if (directory >= 0) ::close(directory);
    return Status::Internal("exclusive publication directory fsync failed");
  }
  ::close(directory);
#endif
  return ExclusiveFilePublicationReceipt{published_path, expected_file_bytes,
                                         observed_digest};
}

}  // namespace pih
