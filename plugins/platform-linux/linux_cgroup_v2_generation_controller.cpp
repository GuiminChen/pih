#include "pih/platform/linux/linux_cgroup_v2_generation_controller.h"

#include <cerrno>
#include <string>

#include <fcntl.h>
#include <unistd.h>

namespace pih {
namespace {

Status failure(const char* operation) {
  return Status::Unavailable(std::string(operation) + " failed with errno " +
                             std::to_string(errno));
}

}  // namespace

Result<LinuxCgroupV2GenerationController>
LinuxCgroupV2GenerationController::Create(
    std::int32_t generation_cgroup_directory_fd) {
  if (generation_cgroup_directory_fd < 0) {
    return Status::InvalidArgument("generation cgroup directory fd is invalid");
  }
  const int owned_fd =
      ::fcntl(generation_cgroup_directory_fd, F_DUPFD_CLOEXEC, 0);
  if (owned_fd < 0) return failure("duplicate generation cgroup directory");
  auto probe = LinuxCgroupV2GenerationProbe::Create(
      owned_fd);
  if (!probe.ok()) {
    (void)::close(owned_fd);
    return probe.status();
  }
  return LinuxCgroupV2GenerationController(
      owned_fd, std::move(*probe));
}

LinuxCgroupV2GenerationController::LinuxCgroupV2GenerationController(
    LinuxCgroupV2GenerationController&& other) noexcept
    : directory_fd_(other.directory_fd_), probe_(std::move(other.probe_)) {
  other.directory_fd_ = -1;
}

LinuxCgroupV2GenerationController&
LinuxCgroupV2GenerationController::operator=(
    LinuxCgroupV2GenerationController&& other) noexcept {
  if (this == &other) return *this;
  close_owned();
  directory_fd_ = other.directory_fd_;
  probe_ = std::move(other.probe_);
  other.directory_fd_ = -1;
  return *this;
}

LinuxCgroupV2GenerationController::~LinuxCgroupV2GenerationController() {
  close_owned();
}

void LinuxCgroupV2GenerationController::close_owned() noexcept {
  if (directory_fd_ >= 0) (void)::close(directory_fd_);
  directory_fd_ = -1;
}

Status LinuxCgroupV2GenerationController::force_kill_domain() {
  const int fd = ::openat(directory_fd_, "cgroup.kill",
                          O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) return failure("openat generation cgroup.kill");
  constexpr char command = '1';
  ssize_t count = 0;
  do {
    count = ::write(fd, &command, 1);
  } while (count < 0 && errno == EINTR);
  if (count != 1) {
    const auto status = count < 0
                            ? failure("write generation cgroup.kill")
                            : Status::Unavailable(
                                  "generation cgroup.kill write was partial");
    ::close(fd);
    return status;
  }
  // A complete cgroup.kill write is the mutation commit point. A close error
  // cannot make that mutation retry-safe, so it must not replace success.
  (void)::close(fd);
  return Status::Ok();
}

Result<bool> LinuxCgroupV2GenerationController::domain_empty() {
  return probe_.domain_empty();
}

}  // namespace pih
