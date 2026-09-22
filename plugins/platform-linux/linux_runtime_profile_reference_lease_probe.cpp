#include "pih/platform/linux/linux_runtime_profile_reference_lease_probe.h"

#include <array>
#include <algorithm>
#include <cerrno>
#include <memory>
#include <limits>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#if __has_include(<linux/fsverity.h>)
#include <linux/fsverity.h>
#define PIH_HAS_LINUX_FSVERITY_HEADER 1
#else
#define PIH_HAS_LINUX_FSVERITY_HEADER 0
#endif

namespace pih {
namespace {

Status failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

class OwnedFd final : public RuntimeProfileReferenceLeaseOwner {
 public:
  explicit OwnedFd(int fd) noexcept : fd_(fd) {}
  ~OwnedFd() override {
    if (fd_ >= 0) ::close(fd_);
  }
 private:
  int fd_ = -1;
};

Status require_fsverity(int fd) {
#if PIH_HAS_LINUX_FSVERITY_HEADER
  std::vector<std::byte> storage(sizeof(struct fsverity_digest) + 64);
  auto* digest = reinterpret_cast<struct fsverity_digest*>(storage.data());
  digest->digest_size = 64;
  if (::ioctl(fd, FS_IOC_MEASURE_VERITY, digest) != 0) {
    if (errno == ENODATA || errno == ENOTTY || errno == EOPNOTSUPP) {
      return Status::FailedPrecondition(
          "runtime reference lease is not protected by fs-verity");
    }
    return failure("FS_IOC_MEASURE_VERITY runtime reference lease");
  }
  if (digest->digest_algorithm != FS_VERITY_HASH_ALG_SHA256 ||
      digest->digest_size != 32) {
    return Status::FailedPrecondition(
        "runtime reference fs-verity algorithm is not SHA-256");
  }
  return Status::Ok();
#else
  (void)fd;
  return Status::FailedPrecondition(
      "Linux fs-verity headers are unavailable in this build");
#endif
}

Result<Sha256Digest> hash_fd(int fd, std::uint64_t exact_bytes) {
  Sha256 hash;
  std::array<std::byte, 64 * 1024> buffer{};
  std::uint64_t consumed = 0;
  while (consumed < exact_bytes) {
    const auto remaining = exact_bytes - consumed;
    const auto requested = static_cast<std::size_t>(
        std::min<std::uint64_t>(remaining, buffer.size()));
    const auto count = ::pread(fd, buffer.data(), requested,
                               static_cast<off_t>(consumed));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) return failure("pread runtime reference lease");
    if (count == 0) {
      return Status::FailedPrecondition(
          "runtime reference lease was truncated");
    }
    auto status = hash.update(
        std::span<const std::byte>(buffer).first(static_cast<std::size_t>(count)));
    if (!status.ok()) return status;
    consumed += static_cast<std::uint64_t>(count);
  }
  return hash.finalize();
}

}  // namespace

LinuxRuntimeProfileReferenceLeaseProbe::LinuxRuntimeProfileReferenceLeaseProbe(
    std::array<std::int32_t, 7> inherited_fds) noexcept
    : inherited_fds_(inherited_fds) {}

Result<LinuxRuntimeProfileReferenceLeaseProbe>
LinuxRuntimeProfileReferenceLeaseProbe::Create(
    std::array<std::int32_t, 7> inherited_fds) {
  for (std::size_t left = 0; left < inherited_fds.size(); ++left) {
    if (inherited_fds[left] < 0) {
      return Status::InvalidArgument(
          "runtime reference inherited fd is invalid");
    }
    for (std::size_t right = left + 1; right < inherited_fds.size(); ++right) {
      if (inherited_fds[left] == inherited_fds[right]) {
        return Status::InvalidArgument(
            "runtime reference inherited fd is duplicated");
      }
    }
  }
  return LinuxRuntimeProfileReferenceLeaseProbe(inherited_fds);
}

Result<RuntimeProfileReferenceLeaseObservation>
LinuxRuntimeProfileReferenceLeaseProbe::observe(
    const RuntimeProfileReferenceDescriptor& descriptor) {
  const auto role = static_cast<std::uint8_t>(descriptor.role);
  if (role == 0 || role > inherited_fds_.size()) {
    return Status::InvalidArgument("runtime reference role is invalid");
  }
  const int fd = inherited_fds_[role - 1];
  const int flags = ::fcntl(fd, F_GETFL);
  if (flags < 0) return failure("F_GETFL runtime reference lease");
  struct stat identity {};
  if (::fstat(fd, &identity) != 0) {
    return failure("fstat runtime reference lease");
  }
  if ((flags & O_ACCMODE) != O_RDONLY || !S_ISREG(identity.st_mode) ||
      identity.st_size < 0 ||
      static_cast<std::uint64_t>(identity.st_size) != descriptor.exact_bytes) {
    return Status::FailedPrecondition(
        "runtime reference lease fd identity is invalid");
  }
  auto immutable = require_fsverity(fd);
  if (!immutable.ok()) return immutable;
  auto content_root = hash_fd(fd, descriptor.exact_bytes);
  if (!content_root.ok()) return content_root.status();
  const int owned_fd = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
  if (owned_fd < 0) return failure("duplicate runtime reference lease fd");
  return RuntimeProfileReferenceLeaseObservation{
      descriptor.role, descriptor.exact_bytes, *content_root, true, true,
      std::make_shared<OwnedFd>(owned_fd)};
}

Result<std::vector<std::byte>>
LinuxRuntimeProfileReferenceLeaseProbe::read_bounded_object(
    const RuntimeProfileReferenceDescriptor& descriptor,
    std::uint64_t maximum_bytes) const {
  const auto role = static_cast<std::uint8_t>(descriptor.role);
  if (role == 0 || role > inherited_fds_.size() ||
      descriptor.exact_bytes == 0 || descriptor.exact_bytes > maximum_bytes ||
      descriptor.exact_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::InvalidArgument(
        "runtime reference object exceeds its read bound");
  }
  const int fd = inherited_fds_[role - 1];
  std::vector<std::byte> bytes(
      static_cast<std::size_t>(descriptor.exact_bytes));
  std::size_t consumed = 0;
  while (consumed < bytes.size()) {
    const auto count = ::pread(fd, bytes.data() + consumed,
                               bytes.size() - consumed,
                               static_cast<off_t>(consumed));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) return failure("pread bounded runtime reference object");
    if (count == 0) {
      return Status::FailedPrecondition(
          "runtime reference object was truncated");
    }
    consumed += static_cast<std::size_t>(count);
  }
  return bytes;
}

}  // namespace pih
