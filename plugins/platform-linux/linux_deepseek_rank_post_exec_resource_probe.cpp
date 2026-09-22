#include "pih/platform/linux/linux_deepseek_rank_post_exec_resource_probe.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pih {
namespace {

constexpr std::size_t kMaximumDecimalBytes = 64;
constexpr std::uint64_t kMaximumProcEntries = 1U << 20U;
constexpr std::uint64_t kMaximumMapsBytes = 128U << 20U;

class OwnedFd final {
 public:
  explicit OwnedFd(int value) noexcept : value_(value) {}
  ~OwnedFd() {
    if (value_ >= 0) (void)::close(value_);
  }
  OwnedFd(const OwnedFd&) = delete;
  OwnedFd& operator=(const OwnedFd&) = delete;
  [[nodiscard]] int get() const noexcept { return value_; }
  [[nodiscard]] explicit operator bool() const noexcept { return value_ >= 0; }

 private:
  int value_ = -1;
};

class OwnedDirectory final {
 public:
  explicit OwnedDirectory(DIR* value) noexcept : value_(value) {}
  ~OwnedDirectory() {
    if (value_ != nullptr) (void)::closedir(value_);
  }
  OwnedDirectory(const OwnedDirectory&) = delete;
  OwnedDirectory& operator=(const OwnedDirectory&) = delete;
  [[nodiscard]] DIR* get() const noexcept { return value_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return value_ != nullptr;
  }

 private:
  DIR* value_ = nullptr;
};

Status system_failure(std::string_view operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

Result<std::uint64_t> decimal(std::string_view text,
                              std::string_view label) {
  while (!text.empty() &&
         (text.back() == '\n' || text.back() == '\r' ||
          text.back() == ' ' || text.back() == '\t')) {
    text.remove_suffix(1);
  }
  if (text.empty()) {
    return Status::InvalidArgument(std::string(label) + " is empty");
  }
  std::uint64_t value = 0;
  for (const auto byte : text) {
    if (byte < '0' || byte > '9') {
      return Status::InvalidArgument(std::string(label) +
                                     " is not decimal");
    }
    const auto digit = static_cast<std::uint64_t>(byte - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
      return Status::ResourceExhausted(std::string(label) +
                                       " overflows uint64");
    }
    value = value * 10U + digit;
  }
  return value;
}

Result<std::uint64_t> read_decimal_file(const char* path,
                                        std::string_view label) {
  OwnedFd fd(::open(path, O_RDONLY | O_CLOEXEC));
  if (!fd) return system_failure(label);
  std::array<char, kMaximumDecimalBytes + 1> bytes{};
  std::size_t used = 0;
  while (used < bytes.size()) {
    const auto count = ::read(fd.get(), bytes.data() + used,
                              bytes.size() - used);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) return system_failure(label);
    if (count == 0) break;
    used += static_cast<std::size_t>(count);
  }
  if (used == bytes.size()) {
    return Status::ResourceExhausted(std::string(label) +
                                     " exceeds its byte bound");
  }
  return decimal(std::string_view(bytes.data(), used), label);
}

bool numeric_name(std::string_view value) noexcept {
  return !value.empty() &&
         std::all_of(value.begin(), value.end(), [](unsigned char byte) {
           return byte >= '0' && byte <= '9';
         });
}

Result<std::uint64_t> count_tasks() {
  OwnedDirectory directory(::opendir("/proc/self/task"));
  if (!directory) return system_failure("open /proc/self/task");
  std::set<std::uint64_t> tasks;
  std::uint64_t entries = 0;
  while (true) {
    errno = 0;
    const auto* entry = ::readdir(directory.get());
    if (entry == nullptr) {
      if (errno != 0) return system_failure("enumerate /proc/self/task");
      break;
    }
    if (++entries > kMaximumProcEntries) {
      return Status::ResourceExhausted(
          "Linux task inventory exceeds its bound");
    }
    const std::string_view name(entry->d_name);
    if (!numeric_name(name)) continue;
    auto task = decimal(name, "Linux task identity");
    if (!task.ok()) return task.status();
    struct stat metadata {};
    if (::fstatat(::dirfd(directory.get()), entry->d_name, &metadata,
                  AT_SYMLINK_NOFOLLOW) != 0) {
      if (errno == ENOENT) continue;
      return system_failure("inspect Linux task");
    }
    if (!S_ISDIR(metadata.st_mode) || !tasks.insert(*task).second) {
      return Status::FailedPrecondition(
          "Linux task inventory is not canonical");
    }
  }
  if (tasks.empty()) {
    return Status::FailedPrecondition("Linux task inventory is empty");
  }
  return static_cast<std::uint64_t>(tasks.size());
}

Result<std::uint64_t> count_open_fds() {
  OwnedDirectory directory(::opendir("/proc/self/fd"));
  if (!directory) return system_failure("open /proc/self/fd");
  const int census_fd = ::dirfd(directory.get());
  std::set<int> descriptors;
  std::uint64_t entries = 0;
  while (true) {
    errno = 0;
    const auto* entry = ::readdir(directory.get());
    if (entry == nullptr) {
      if (errno != 0) return system_failure("enumerate /proc/self/fd");
      break;
    }
    if (++entries > kMaximumProcEntries) {
      return Status::ResourceExhausted(
          "Linux descriptor inventory exceeds its bound");
    }
    const std::string_view name(entry->d_name);
    if (!numeric_name(name)) continue;
    auto descriptor = decimal(name, "Linux descriptor identity");
    if (!descriptor.ok() || *descriptor > static_cast<std::uint64_t>(INT_MAX)) {
      return Status::InvalidArgument(
          "Linux descriptor inventory is malformed");
    }
    const int value = static_cast<int>(*descriptor);
    if (value == census_fd) continue;
    errno = 0;
    if (::fcntl(value, F_GETFD) >= 0) {
      if (!descriptors.insert(value).second) {
        return Status::FailedPrecondition(
            "Linux descriptor inventory is not canonical");
      }
    } else if (errno != EBADF) {
      return system_failure("inspect Linux descriptor");
    }
  }
  return static_cast<std::uint64_t>(descriptors.size());
}

Result<std::uint64_t> count_vmas() {
  OwnedFd fd(::open("/proc/self/maps", O_RDONLY | O_CLOEXEC));
  if (!fd) return system_failure("open /proc/self/maps");
  std::array<char, 16U * 1024U> bytes{};
  std::uint64_t total_bytes = 0;
  std::uint64_t lines = 0;
  bool saw_data = false;
  bool ended_with_newline = false;
  while (true) {
    const auto count = ::read(fd.get(), bytes.data(), bytes.size());
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) return system_failure("read /proc/self/maps");
    if (count == 0) break;
    saw_data = true;
    const auto size = static_cast<std::size_t>(count);
    if (total_bytes > kMaximumMapsBytes - size) {
      return Status::ResourceExhausted(
          "Linux VMA inventory exceeds its byte bound");
    }
    total_bytes += size;
    for (std::size_t index = 0; index < size; ++index) {
      if (bytes[index] == '\n') ++lines;
    }
    ended_with_newline = bytes[size - 1] == '\n';
    if (lines > kMaximumProcEntries) {
      return Status::ResourceExhausted(
          "Linux VMA inventory exceeds its entry bound");
    }
  }
  if (!saw_data || !ended_with_newline || lines == 0) {
    return Status::FailedPrecondition(
        "Linux VMA inventory is incomplete");
  }
  return lines;
}

Result<std::uint64_t> finite_rlimit(rlim_t value,
                                    std::string_view label) {
  if (value == RLIM_INFINITY) {
    return Status::FailedPrecondition(std::string(label) +
                                      " must be finite");
  }
  if constexpr (sizeof(rlim_t) > sizeof(std::uint64_t)) {
    if (value > static_cast<rlim_t>(
                    std::numeric_limits<std::uint64_t>::max())) {
      return Status::ResourceExhausted(std::string(label) +
                                       " exceeds uint64");
    }
  }
  return static_cast<std::uint64_t>(value);
}

}  // namespace

Result<DeepSeekRankPostExecResourceSnapshot>
LinuxDeepSeekRankPostExecResourceProbe::sample() {
  struct rlimit nofile {};
  if (::getrlimit(RLIMIT_NOFILE, &nofile) != 0) {
    return system_failure("getrlimit RLIMIT_NOFILE");
  }
  auto nofile_soft = finite_rlimit(nofile.rlim_cur, "RLIMIT_NOFILE soft");
  if (!nofile_soft.ok()) return nofile_soft.status();
  auto nofile_hard = finite_rlimit(nofile.rlim_max, "RLIMIT_NOFILE hard");
  if (!nofile_hard.ok()) return nofile_hard.status();
  auto nr_open = read_decimal_file("/proc/sys/fs/nr_open", "fs.nr_open");
  if (!nr_open.ok()) return nr_open.status();
  auto map_limit = read_decimal_file(
      "/proc/sys/vm/max_map_count", "vm.max_map_count");
  if (!map_limit.ok()) return map_limit.status();
  auto tasks = count_tasks();
  if (!tasks.ok()) return tasks.status();
  auto descriptors = count_open_fds();
  if (!descriptors.ok()) return descriptors.status();
  auto maps = count_vmas();
  if (!maps.ok()) return maps.status();
  auto inflight = scm_rights_probe_->sample_inflight_fd_count();
  if (!inflight.ok()) return inflight.status();
  const int dumpable = ::prctl(PR_GET_DUMPABLE);
  if (dumpable < 0) return system_failure("PR_GET_DUMPABLE");
  return DeepSeekRankPostExecResourceSnapshot{
      static_cast<std::uint64_t>(::getpid()), *tasks, *descriptors,
      *inflight, *maps, *nofile_soft, *nofile_hard, *nr_open,
      *map_limit, dumpable == 0};
}

}  // namespace pih
