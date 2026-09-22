#include "pih/platform/linux/linux_cgroup_v2_generation_probe.h"

#include <array>
#include <cerrno>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "pih/model/cgroup_v2_events.h"

namespace pih {
namespace {

Status failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

}  // namespace

Result<LinuxCgroupV2GenerationProbe> LinuxCgroupV2GenerationProbe::Create(
    std::int32_t generation_cgroup_directory_fd) {
  if (generation_cgroup_directory_fd < 0)
    return Status::InvalidArgument("generation cgroup directory fd is invalid");
  struct stat identity {};
  if (::fstat(generation_cgroup_directory_fd, &identity) != 0)
    return failure("fstat generation cgroup directory");
  if (!S_ISDIR(identity.st_mode))
    return Status::InvalidArgument("generation cgroup fd is not a directory");
  return LinuxCgroupV2GenerationProbe(generation_cgroup_directory_fd);
}

Result<bool> LinuxCgroupV2GenerationProbe::domain_empty() const {
  const int fd = ::openat(directory_fd_, "cgroup.events",
                          O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) return failure("openat generation cgroup.events");
  std::array<char, 4097> bytes{};
  std::size_t used = 0;
  for (;;) {
    const auto count = ::read(fd, bytes.data() + used, bytes.size() - used);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) {
      const auto status = failure("read generation cgroup.events");
      ::close(fd);
      return status;
    }
    if (count == 0) break;
    used += static_cast<std::size_t>(count);
    if (used == bytes.size()) {
      ::close(fd);
      return Status::ResourceExhausted(
          "generation cgroup.events exceeds bounded payload");
    }
  }
  if (::close(fd) != 0) return failure("close generation cgroup.events");
  auto populated = parse_cgroup_v2_populated(
      std::string_view(bytes.data(), used));
  if (!populated.ok()) return populated.status();
  return !*populated;
}

}  // namespace pih
