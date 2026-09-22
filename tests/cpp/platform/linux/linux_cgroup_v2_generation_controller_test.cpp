#include "pih/platform/linux/linux_cgroup_v2_generation_controller.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pih {
namespace {

class TemporaryCgroupDirectory final {
 public:
  TemporaryCgroupDirectory() {
    std::array<char, 64> pattern{};
    const std::string value = "/tmp/pih-cgroup-controller-XXXXXX";
    std::copy(value.begin(), value.end(), pattern.begin());
    char* created = ::mkdtemp(pattern.data());
    if (created == nullptr) return;
    path_ = created;
    fd_ = ::open(path_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    write_file("cgroup.kill", "");
    write_file("cgroup.events", "populated 1\nfrozen 0\n");
  }
  ~TemporaryCgroupDirectory() {
    if (fd_ >= 0) {
      (void)::unlinkat(fd_, "cgroup.kill", 0);
      (void)::unlinkat(fd_, "cgroup.events", 0);
      (void)::close(fd_);
    }
    if (!path_.empty()) (void)::rmdir(path_.c_str());
  }
  int fd() const { return fd_; }
  void write_events(const char* value) { write_file("cgroup.events", value); }
  std::string read_kill() const {
    const int fd = ::openat(fd_, "cgroup.kill", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return {};
    std::array<char, 8> bytes{};
    const auto count = ::read(fd, bytes.data(), bytes.size());
    (void)::close(fd);
    return count > 0 ? std::string(bytes.data(), static_cast<std::size_t>(count))
                     : std::string{};
  }

 private:
  void write_file(const char* name, const char* value) {
    if (fd_ < 0) return;
    const int fd = ::openat(fd_, name,
                            O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) return;
    const std::string_view bytes(value);
    if (!bytes.empty()) (void)::write(fd, bytes.data(), bytes.size());
    (void)::close(fd);
  }

  std::string path_;
  int fd_ = -1;
};

TEST(LinuxCgroupV2GenerationControllerTest,
     OwnsDirectoryAfterCallerClosesOriginalDescriptor) {
  TemporaryCgroupDirectory directory;
  ASSERT_GE(directory.fd(), 0);
  const int caller_fd = ::fcntl(directory.fd(), F_DUPFD_CLOEXEC, 0);
  ASSERT_GE(caller_fd, 0);
  auto value = LinuxCgroupV2GenerationController::Create(caller_fd);
  ASSERT_TRUE(value.ok());
  auto controller = std::move(*value);
  ASSERT_EQ(::close(caller_fd), 0);

  ASSERT_TRUE(controller.force_kill_domain().ok());
  EXPECT_EQ(directory.read_kill(), "1");
  auto empty = controller.domain_empty();
  ASSERT_TRUE(empty.ok());
  EXPECT_FALSE(*empty);

  directory.write_events("populated 0\nfrozen 0\n");
  empty = controller.domain_empty();
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(*empty);
}

}  // namespace
}  // namespace pih
