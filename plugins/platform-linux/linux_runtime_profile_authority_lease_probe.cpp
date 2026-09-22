#include "pih/platform/linux/linux_runtime_profile_authority_lease_probe.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <memory>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#if __has_include(<linux/fsverity.h>)
#include <linux/fsverity.h>
#define PIH_HAS_LINUX_AUTHORITY_FSVERITY_HEADER 1
#else
#define PIH_HAS_LINUX_AUTHORITY_FSVERITY_HEADER 0
#endif

namespace pih {
namespace {

Status authority_failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

class OwnedAuthorityFd final : public RuntimeProfileAuthorityLeaseOwner {
 public:
  explicit OwnedAuthorityFd(int fd) noexcept : fd_(fd) {}
  ~OwnedAuthorityFd() override {
    if (fd_ >= 0) ::close(fd_);
  }

 private:
  int fd_ = -1;
};

Status require_authority_fsverity(int fd) {
#if PIH_HAS_LINUX_AUTHORITY_FSVERITY_HEADER
  std::vector<std::byte> storage(sizeof(struct fsverity_digest) + 64);
  auto* digest = reinterpret_cast<struct fsverity_digest*>(storage.data());
  digest->digest_size = 64;
  if (::ioctl(fd, FS_IOC_MEASURE_VERITY, digest) != 0) {
    if (errno == ENODATA || errno == ENOTTY || errno == EOPNOTSUPP) {
      return Status::FailedPrecondition(
          "runtime authority lease is not protected by fs-verity");
    }
    return authority_failure("FS_IOC_MEASURE_VERITY runtime authority lease");
  }
  if (digest->digest_algorithm != FS_VERITY_HASH_ALG_SHA256 ||
      digest->digest_size != 32) {
    return Status::FailedPrecondition(
        "runtime authority fs-verity algorithm is not SHA-256");
  }
  return Status::Ok();
#else
  (void)fd;
  return Status::FailedPrecondition(
      "Linux fs-verity headers are unavailable in this build");
#endif
}

Result<Sha256Digest> hash_authority_fd(int fd, std::uint64_t exact_bytes) {
  Sha256 hash;
  std::array<std::byte, 64 * 1024> buffer{};
  std::uint64_t consumed = 0;
  while (consumed < exact_bytes) {
    const auto requested = static_cast<std::size_t>(
        std::min<std::uint64_t>(exact_bytes - consumed, buffer.size()));
    const auto count =
        ::pread(fd, buffer.data(), requested, static_cast<off_t>(consumed));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) return authority_failure("pread runtime authority lease");
    if (count == 0) {
      return Status::FailedPrecondition("runtime authority lease was truncated");
    }
    auto status = hash.update(std::span<const std::byte>(buffer).first(
        static_cast<std::size_t>(count)));
    if (!status.ok()) return status;
    consumed += static_cast<std::uint64_t>(count);
  }
  return hash.finalize();
}

}  // namespace

LinuxRuntimeProfileAuthorityLeaseProbe::LinuxRuntimeProfileAuthorityLeaseProbe(
    std::array<std::int32_t, 5> inherited_fds) noexcept
    : inherited_fds_(inherited_fds) {}

Result<LinuxRuntimeProfileAuthorityLeaseProbe>
LinuxRuntimeProfileAuthorityLeaseProbe::Create(
    std::array<std::int32_t, 5> inherited_fds) {
  for (std::size_t left = 0; left < inherited_fds.size(); ++left) {
    if (inherited_fds[left] < 0) {
      return Status::InvalidArgument("runtime authority inherited fd is invalid");
    }
    for (std::size_t right = left + 1; right < inherited_fds.size(); ++right) {
      if (inherited_fds[left] == inherited_fds[right]) {
        return Status::InvalidArgument(
            "runtime authority inherited fd is duplicated");
      }
    }
  }
  return LinuxRuntimeProfileAuthorityLeaseProbe(inherited_fds);
}

Result<RuntimeProfileAuthorityLeaseObservation>
LinuxRuntimeProfileAuthorityLeaseProbe::observe(
    const RuntimeProfileAuthorityDescriptor& descriptor) {
  const auto role = static_cast<std::uint8_t>(descriptor.role);
  if (role == 0 || role > inherited_fds_.size()) {
    return Status::InvalidArgument("runtime authority role is invalid");
  }
  const int fd = inherited_fds_[role - 1];
  const int flags = ::fcntl(fd, F_GETFL);
  if (flags < 0) return authority_failure("F_GETFL runtime authority lease");
  struct stat identity {};
  if (::fstat(fd, &identity) != 0) {
    return authority_failure("fstat runtime authority lease");
  }
  if ((flags & O_ACCMODE) != O_RDONLY || !S_ISREG(identity.st_mode) ||
      identity.st_size < 0 ||
      static_cast<std::uint64_t>(identity.st_size) != descriptor.exact_bytes) {
    return Status::FailedPrecondition(
        "runtime authority lease fd identity is invalid");
  }
  auto immutable = require_authority_fsverity(fd);
  if (!immutable.ok()) return immutable;
  auto root = hash_authority_fd(fd, descriptor.exact_bytes);
  if (!root.ok()) return root.status();
  const int owned_fd = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
  if (owned_fd < 0) {
    return authority_failure("duplicate runtime authority lease fd");
  }
  return RuntimeProfileAuthorityLeaseObservation{
      descriptor.role, descriptor.exact_bytes, *root, true, true,
      std::make_shared<OwnedAuthorityFd>(owned_fd)};
}

}  // namespace pih
