#include "pih/platform/linux/linux_engine_allocation_lease_probe.h"

#include <array>
#include <cerrno>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pih {
namespace {

Status failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

Result<bool> query_write_lock(int fd, bool independent_view) {
#ifndef F_OFD_GETLK
  (void)fd;
  (void)independent_view;
  return Status::FailedPrecondition("Linux OFD locks are unavailable");
#else
  struct flock lock {};
  lock.l_type = F_WRLCK;
  lock.l_whence = SEEK_SET;
  lock.l_start = 0;
  lock.l_len = 0;
  if (::fcntl(fd, F_OFD_GETLK, &lock) != 0)
    return failure("F_OFD_GETLK engine allocation lease");
  if (!independent_view) return lock.l_type == F_UNLCK;
  return lock.l_type == F_WRLCK && lock.l_pid == -1;
#endif
}

}  // namespace

Result<EngineAllocationLeaseObservation>
LinuxEngineAllocationLeaseProbe::Observe(std::int32_t inherited_lease_fd) {
  if (inherited_lease_fd < 0)
    return Status::InvalidArgument("engine allocation lease fd is invalid");
  struct stat inherited_identity {};
  if (::fstat(inherited_lease_fd, &inherited_identity) != 0)
    return failure("fstat inherited engine allocation lease");
  if (!S_ISREG(inherited_identity.st_mode) ||
      inherited_identity.st_size !=
          static_cast<off_t>(Sha256Digest{}.bytes.size()))
    return Status::FailedPrecondition(
        "engine allocation lease target identity is invalid");

  Sha256Digest token{};
  std::size_t consumed = 0;
  while (consumed < token.bytes.size()) {
    const auto count = ::pread(inherited_lease_fd, token.bytes.data() + consumed,
                               token.bytes.size() - consumed,
                               static_cast<off_t>(consumed));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) return failure("pread engine allocation lease token");
    if (count == 0)
      return Status::FailedPrecondition(
          "engine allocation lease token was truncated");
    consumed += static_cast<std::size_t>(count);
  }

  auto inherited_has_no_conflict = query_write_lock(inherited_lease_fd, false);
  if (!inherited_has_no_conflict.ok())
    return inherited_has_no_conflict.status();

  const auto path = "/proc/self/fd/" + std::to_string(inherited_lease_fd);
  const int independent_fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
  if (independent_fd < 0)
    return failure("open independent engine allocation lease view");
  struct stat independent_identity {};
  if (::fstat(independent_fd, &independent_identity) != 0) {
    const auto status = failure("fstat independent engine allocation lease");
    ::close(independent_fd);
    return status;
  }
  const bool same_identity =
      inherited_identity.st_dev == independent_identity.st_dev &&
      inherited_identity.st_ino == independent_identity.st_ino;
  auto independent_sees_lock = query_write_lock(independent_fd, true);
  const int close_result = ::close(independent_fd);
  if (!independent_sees_lock.ok()) return independent_sees_lock.status();
  if (close_result != 0)
    return failure("close independent engine allocation lease view");

  return EngineAllocationLeaseObservation{
      token, static_cast<std::uint64_t>(inherited_identity.st_dev),
      static_cast<std::uint64_t>(inherited_identity.st_ino), true,
      same_identity && *inherited_has_no_conflict && *independent_sees_lock};
}

}  // namespace pih
