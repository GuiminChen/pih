#include "identity_copy.h"
#include <algorithm>
#include <cerrno>
#include <map>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pih::offline_deepseek {
namespace {
constexpr std::uint64_t kMaximumShardBytes = 512ULL << 30;
bool Same(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode &&
      a.st_size == b.st_size && a.st_mtim.tv_sec == b.st_mtim.tv_sec &&
      a.st_mtim.tv_nsec == b.st_mtim.tv_nsec && a.st_ctim.tv_sec == b.st_ctim.tv_sec &&
      a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
Status Read(int fd, std::uint64_t offset, std::span<std::byte> bytes) {
  while (!bytes.empty()) {
    const auto count = ::pread(fd, bytes.data(), bytes.size(), static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return Status::Unavailable("identity source read failed");
    offset += static_cast<std::size_t>(count);
    bytes = bytes.subspan(static_cast<std::size_t>(count));
  }
  return Status::Ok();
}
Status Write(int fd, std::span<const std::byte> bytes) {
  while (!bytes.empty()) {
    const auto count = ::write(fd, bytes.data(), bytes.size());
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return Status::Unavailable("identity output write failed; partial member retained");
    bytes = bytes.subspan(static_cast<std::size_t>(count));
  }
  return Status::Ok();
}
struct OwnedFd final {
  int value;
  ~OwnedFd() { if (value >= 0) ::close(value); }
};
}  // namespace

Result<CopyReceipt> CopyIdentityShard(
    int staging_directory_fd, std::string_view member_name,
    std::span<const std::byte> header_prefix, std::span<const CopyRange> ranges) {
  if (staging_directory_fd < 0 || member_name.empty() || member_name.size() > 255 ||
      member_name == "." || member_name == ".." || member_name.find_first_of("/\\") != member_name.npos ||
      member_name.find('\0') != member_name.npos || header_prefix.size() < 8 ||
      header_prefix.size() > (16ULL << 20) + 8 || ranges.empty() || ranges.size() > 72'317)
    return Status::InvalidArgument("identity shard input is outside its bounds");
  struct stat directory{};
  if (::fstat(staging_directory_fd, &directory) != 0 || !S_ISDIR(directory.st_mode))
    return Status::InvalidArgument("identity staging descriptor is not a directory");
  std::map<int, struct stat> sources;
  std::uint64_t total = header_prefix.size();
  for (const auto& range : ranges) {
    if (range.source_fd < 0 || range.bytes == 0 || range.payload_sha256 == Sha256Digest{} ||
        range.bytes > kMaximumShardBytes - total)
      return Status::InvalidArgument("identity payload range is invalid");
    if (!sources.contains(range.source_fd)) {
      struct stat source{};
      if (::fstat(range.source_fd, &source) != 0 || !S_ISREG(source.st_mode) || source.st_size < 0)
        return Status::InvalidArgument("identity source descriptor is not a regular file");
      if (sources.size() >= 1024)
        return Status::ResourceExhausted("identity source descriptor limit exceeded");
      sources.emplace(range.source_fd, source);
    }
    const auto size = static_cast<std::uint64_t>(sources.at(range.source_fd).st_size);
    if (range.begin > size || range.bytes > size - range.begin)
      return Status::InvalidArgument("identity payload extends beyond its source");
    total += range.bytes;
  }
  // Allocate workspace before creating a file, so allocation failure leaves no member.
  std::vector<std::byte> buffer(1U << 20);
  const std::string name(member_name);
  OwnedFd output{::openat(staging_directory_fd, name.c_str(),
      O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600)};
  if (output.value < 0) return Status::FailedPrecondition("identity output must be a new staging member");
  struct stat created{};
  if (::fstat(output.value, &created) != 0)
    return Status::Unavailable("cannot inspect identity output; partial member retained");
  Sha256 object;
  auto status = Write(output.value, header_prefix);
  if (!status.ok()) return status;
  status = object.update(header_prefix);
  if (!status.ok()) return status;
  for (const auto& range : ranges) {
    Sha256 payload;
    for (std::uint64_t offset = 0; offset < range.bytes;) {
      auto chunk = std::span(buffer).first(static_cast<std::size_t>(
          std::min<std::uint64_t>(buffer.size(), range.bytes - offset)));
      status = Read(range.source_fd, range.begin + offset, chunk);
      if (!status.ok()) return status;
      status = payload.update(chunk);
      if (!status.ok()) return status;
      status = object.update(chunk);
      if (!status.ok()) return status;
      status = Write(output.value, chunk);
      if (!status.ok()) return status;
      offset += chunk.size();
    }
    auto digest = payload.finalize();
    if (!digest.ok()) return digest.status();
    if (*digest != range.payload_sha256)
      return Status::FailedPrecondition("identity payload hash mismatch; partial member retained");
  }
  auto expected = object.finalize();
  if (!expected.ok()) return expected.status();
  if (::fsync(output.value) != 0)
    return Status::Unavailable("identity output sync failed; partial member retained");
  struct stat synced{};
  if (::fstat(output.value, &synced) != 0 || synced.st_size < 0 ||
      static_cast<std::uint64_t>(synced.st_size) != total)
    return Status::FailedPrecondition("identity output length differs before readback");
  Sha256 reread;
  for (std::uint64_t offset = 0; offset < total;) {
    auto chunk = std::span(buffer).first(static_cast<std::size_t>(
        std::min<std::uint64_t>(buffer.size(), total - offset)));
    status = Read(output.value, offset, chunk);
    if (!status.ok()) return status;
    status = reread.update(chunk);
    if (!status.ok()) return status;
    offset += chunk.size();
  }
  auto observed = reread.finalize();
  if (!observed.ok()) return observed.status();
  if (*observed != *expected)
    return Status::FailedPrecondition("identity output readback mismatch; partial member retained");
  for (const auto& [fd, initial] : sources) {
    struct stat current{};
    if (::fstat(fd, &current) != 0 || !Same(initial, current))
      return Status::FailedPrecondition("identity source changed during copy; partial member retained");
  }
  struct stat output_state{}, named{};
  if (::fstat(output.value, &output_state) != 0 ||
      ::fstatat(staging_directory_fd, name.c_str(), &named, AT_SYMLINK_NOFOLLOW) != 0 ||
      !S_ISREG(output_state.st_mode) || output_state.st_dev != created.st_dev || output_state.st_ino != created.st_ino ||
      !Same(synced, output_state) || !Same(output_state, named))
    return Status::FailedPrecondition("identity output changed or was replaced; partial member retained");
  if (::fsync(staging_directory_fd) != 0)
    return Status::Unavailable("identity staging directory sync failed; member retained");
  return CopyReceipt{total, *expected};
}
Result<CopyReceipt> WriteMetadataMember(int staging_directory_fd,
    std::string_view member_name, std::string_view contents,
    const Sha256Digest& expected_sha256) {
  if (member_name.empty() || member_name.size() > 255 || member_name == "." ||
      member_name == ".." || member_name.find_first_of("/\\") != member_name.npos ||
      member_name.find('\0') != member_name.npos || contents.empty() ||
      contents.size() > (128ULL << 20) || expected_sha256 == Sha256Digest{})
    return Status::InvalidArgument("metadata member outside bounds");
  struct stat directory{};
  if (::fstat(staging_directory_fd, &directory) != 0 || !S_ISDIR(directory.st_mode))
    return Status::InvalidArgument("metadata staging descriptor is not a directory");
  const auto bytes = std::as_bytes(std::span(contents));
  auto digest = sha256(bytes);
  if (!digest.ok()) return digest.status();
  if (*digest != expected_sha256)
    return Status::InvalidArgument("metadata input digest differs before write");
  std::vector<std::byte> buffer(1U << 20);
  const std::string name(member_name);
  OwnedFd output{::openat(staging_directory_fd, name.c_str(),
      O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600)};
  if (output.value < 0)
    return Status::FailedPrecondition("metadata output must be a new staging member");
  auto status = Write(output.value, bytes);
  if (!status.ok()) return status;
  if (::fsync(output.value) != 0)
    return Status::Unavailable("metadata sync failed; partial member retained");
  struct stat synced{};
  if (::fstat(output.value, &synced) != 0 || !S_ISREG(synced.st_mode) ||
      synced.st_size < 0 || static_cast<std::uint64_t>(synced.st_size) != contents.size())
    return Status::FailedPrecondition("metadata output size differs; member retained");
  Sha256 reread;
  for (std::uint64_t offset = 0; offset < contents.size();) {
    auto chunk = std::span(buffer).first(static_cast<std::size_t>(
        std::min<std::uint64_t>(buffer.size(), contents.size() - offset)));
    status = Read(output.value, offset, chunk);
    if (!status.ok()) return status;
    status = reread.update(chunk);
    if (!status.ok()) return status;
    offset += chunk.size();
  }
  auto observed = reread.finalize();
  if (!observed.ok()) return observed.status();
  struct stat current{}, named{};
  if (*observed != expected_sha256 || ::fstat(output.value, &current) != 0 ||
      ::fstatat(staging_directory_fd, name.c_str(), &named, AT_SYMLINK_NOFOLLOW) != 0 ||
      !Same(synced, current) || !Same(current, named))
    return Status::FailedPrecondition("metadata readback or identity changed; member retained");
  if (::fsync(staging_directory_fd) != 0)
    return Status::Unavailable("metadata directory sync failed; member retained");
  return CopyReceipt{static_cast<std::uint64_t>(contents.size()), *observed};
}
}  // namespace pih::offline_deepseek
