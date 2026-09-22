#include "pih/platform/linux/linux_deepseek_rank_artifact_prefault_operations.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <linux/magic.h>
#include <linux/openat2.h>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

constexpr std::size_t kMaximumControlBytes = 1U << 20U;
constexpr std::size_t kMincoreChunkPages = 65536;

Status failure(std::string_view operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

class OwnedFd final {
 public:
  explicit OwnedFd(int value = -1) noexcept : value_(value) {}
  ~OwnedFd() {
    if (value_ >= 0) (void)::close(value_);
  }
  OwnedFd(const OwnedFd&) = delete;
  OwnedFd& operator=(const OwnedFd&) = delete;
  OwnedFd(OwnedFd&& other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  OwnedFd& operator=(OwnedFd&& other) noexcept {
    if (this != &other) {
      if (value_ >= 0) (void)::close(value_);
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  [[nodiscard]] int get() const noexcept { return value_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return value_ >= 0;
  }

 private:
  int value_ = -1;
};

Result<std::string> read_bounded_fd(int fd, std::string_view label) {
  std::string result;
  result.reserve(4096);
  std::array<char, 4096> buffer{};
  while (true) {
    const auto count = ::read(fd, buffer.data(), buffer.size());
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) return failure(label);
    if (count == 0) break;
    if (result.size() > kMaximumControlBytes -
                            static_cast<std::size_t>(count)) {
      return Status::ResourceExhausted(std::string(label) +
                                       " exceeds its byte bound");
    }
    result.append(buffer.data(), static_cast<std::size_t>(count));
  }
  if (result.empty()) {
    return Status::FailedPrecondition(std::string(label) + " is empty");
  }
  return result;
}

Result<std::string> read_bounded_path(const char* path,
                                      std::string_view label) {
  OwnedFd fd(::open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (!fd) return failure(label);
  return read_bounded_fd(fd.get(), label);
}

Result<std::string> read_bounded_at(int directory_fd, const char* name,
                                    std::string_view label) {
  OwnedFd fd(::openat(directory_fd, name,
                      O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (!fd) return failure(label);
  return read_bounded_fd(fd.get(), label);
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
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) /
                    10U) {
      return Status::ResourceExhausted(std::string(label) +
                                       " overflows uint64");
    }
    value = value * 10U + digit;
  }
  return value;
}

Result<std::string> current_cgroup_path() {
  auto control = read_bounded_path("/proc/self/cgroup",
                                   "read /proc/self/cgroup");
  if (!control.ok()) return control.status();
  std::string path;
  std::size_t cursor = 0;
  std::uint32_t matches = 0;
  while (cursor < control->size()) {
    const auto end = control->find('\n', cursor);
    const auto length = end == std::string::npos
                            ? control->size() - cursor
                            : end - cursor;
    const std::string_view line(control->data() + cursor, length);
    if (line.starts_with("0::")) {
      ++matches;
      path.assign(line.substr(3));
    }
    if (end == std::string::npos) break;
    cursor = end + 1U;
  }
  if (matches != 1 || path.empty() || path.front() != '/' ||
      (path.size() > 1 && path.back() == '/') ||
      path.find('\0') != std::string::npos) {
    return Status::FailedPrecondition(
        "Linux worker does not have one canonical cgroup-v2 membership");
  }
  std::size_t begin = 1;
  while (begin < path.size()) {
    const auto end = path.find('/', begin);
    const auto component = std::string_view(path).substr(
        begin, end == std::string::npos ? path.size() - begin
                                        : end - begin);
    if (component.empty() || component == "." || component == "..") {
      return Status::FailedPrecondition(
          "Linux worker cgroup-v2 membership path is not canonical");
    }
    if (end == std::string::npos) break;
    begin = end + 1U;
  }
  return path;
}

Result<OwnedFd> open_current_cgroup(std::string_view path) {
  OwnedFd root(::open("/sys/fs/cgroup", O_PATH | O_DIRECTORY | O_CLOEXEC));
  if (!root) return failure("open cgroup-v2 root");
  struct statfs filesystem {};
  if (::fstatfs(root.get(), &filesystem) != 0) {
    return failure("fstatfs cgroup-v2 root");
  }
  if (static_cast<unsigned long>(filesystem.f_type) !=
      static_cast<unsigned long>(CGROUP2_SUPER_MAGIC)) {
    return Status::FailedPrecondition(
        "Linux cgroup root is not cgroup v2");
  }
  if (path == "/") {
    const auto duplicate = ::fcntl(root.get(), F_DUPFD_CLOEXEC, 0);
    if (duplicate < 0) return failure("duplicate cgroup-v2 root");
    return OwnedFd(duplicate);
  }
  std::string relative(path.substr(1));
  struct open_how how {};
  how.flags = O_PATH | O_DIRECTORY | O_CLOEXEC;
  how.resolve = RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS |
                RESOLVE_NO_SYMLINKS;
  const auto descriptor = static_cast<int>(::syscall(
      SYS_openat2, root.get(), relative.c_str(), &how, sizeof(how)));
  if (descriptor < 0) return failure("open current cgroup-v2 directory");
  return OwnedFd(descriptor);
}

Result<std::uint64_t> status_kib(std::string_view status,
                                 std::string_view key) {
  std::size_t cursor = 0;
  while (cursor < status.size()) {
    const auto end = status.find('\n', cursor);
    const auto length = end == std::string_view::npos
                            ? status.size() - cursor
                            : end - cursor;
    auto line = status.substr(cursor, length);
    if (line.starts_with(key)) {
      line.remove_prefix(key.size());
      while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) {
        line.remove_prefix(1);
      }
      if (!line.ends_with(" kB")) {
        return Status::InvalidArgument(
            "Linux status memory value has an unknown unit");
      }
      line.remove_suffix(3);
      auto value = decimal(line, key);
      if (!value.ok()) return value.status();
      return checked_mul_u64(*value, 1024);
    }
    if (end == std::string_view::npos) break;
    cursor = end + 1U;
  }
  return Status::FailedPrecondition(std::string(key) +
                                    " is absent from Linux status");
}

struct MemoryStat final {
  std::uint64_t file = 0;
  std::uint64_t anon = 0;
  std::uint64_t kernel = 0;
};

Result<MemoryStat> parse_memory_stat(std::string_view input) {
  MemoryStat result;
  bool saw_file = false;
  bool saw_anon = false;
  bool saw_kernel = false;
  std::size_t cursor = 0;
  while (cursor < input.size()) {
    const auto end = input.find('\n', cursor);
    const auto length = end == std::string_view::npos
                            ? input.size() - cursor
                            : end - cursor;
    const auto line = input.substr(cursor, length);
    const auto separator = line.find(' ');
    if (separator != std::string_view::npos) {
      const auto name = line.substr(0, separator);
      auto value = decimal(line.substr(separator + 1U), name);
      if (!value.ok()) return value.status();
      if (name == "file") {
        if (saw_file) return Status::InvalidArgument("duplicate cgroup file");
        saw_file = true;
        result.file = *value;
      } else if (name == "anon") {
        if (saw_anon) return Status::InvalidArgument("duplicate cgroup anon");
        saw_anon = true;
        result.anon = *value;
      } else if (name == "kernel") {
        if (saw_kernel) {
          return Status::InvalidArgument("duplicate cgroup kernel");
        }
        saw_kernel = true;
        result.kernel = *value;
      }
    }
    if (end == std::string_view::npos) break;
    cursor = end + 1U;
  }
  if (!saw_file || !saw_anon || !saw_kernel) {
    return Status::FailedPrecondition(
        "Linux cgroup memory.stat lacks required owners");
  }
  return result;
}

Result<std::uint64_t> monotonic_now_ns() {
  struct timespec value {};
  if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return failure("clock_gettime CLOCK_MONOTONIC");
  }
  if (value.tv_sec < 0 || value.tv_nsec < 0 ||
      value.tv_nsec >= 1000000000L) {
    return Status::FailedPrecondition("Linux monotonic clock is invalid");
  }
  auto seconds = checked_mul_u64(
      static_cast<std::uint64_t>(value.tv_sec), UINT64_C(1000000000));
  if (!seconds.ok()) return seconds.status();
  return checked_add_u64(*seconds,
                         static_cast<std::uint64_t>(value.tv_nsec));
}

Result<std::uint64_t> resident_pages(const std::byte* address,
                                     std::size_t bytes,
                                     std::size_t page_bytes) {
  const auto pages = 1U + (bytes - 1U) / page_bytes;
  std::vector<unsigned char> vector(pages);
  if (::mincore(const_cast<std::byte*>(address), bytes, vector.data()) != 0) {
    return failure("mincore selected artifact pages");
  }
  return static_cast<std::uint64_t>(std::count_if(
      vector.begin(), vector.end(),
      [](unsigned char value) { return (value & 1U) != 0; }));
}

}  // namespace

struct LinuxDeepSeekRankArtifactPrefaultOperations::Impl final {
  OwnedFd cgroup_directory;
  std::string cgroup_path;
};

LinuxDeepSeekRankArtifactPrefaultOperations::
    LinuxDeepSeekRankArtifactPrefaultOperations(
        std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

LinuxDeepSeekRankArtifactPrefaultOperations::
    ~LinuxDeepSeekRankArtifactPrefaultOperations() = default;

Result<std::unique_ptr<LinuxDeepSeekRankArtifactPrefaultOperations>>
LinuxDeepSeekRankArtifactPrefaultOperations::Create() {
  auto path = current_cgroup_path();
  if (!path.ok()) return path.status();
  auto directory = open_current_cgroup(*path);
  if (!directory.ok()) return directory.status();
  auto impl = std::make_unique<Impl>();
  impl->cgroup_directory = std::move(*directory);
  impl->cgroup_path = std::move(*path);
  return std::unique_ptr<LinuxDeepSeekRankArtifactPrefaultOperations>(
      new LinuxDeepSeekRankArtifactPrefaultOperations(std::move(impl)));
}

Result<DeepSeekRankArtifactPrefaultResourceSnapshot>
LinuxDeepSeekRankArtifactPrefaultOperations::sample_resources() {
  if (impl_ == nullptr || !impl_->cgroup_directory) {
    return Status::FailedPrecondition(
        "Linux artifact prefault operations are not initialized");
  }
  auto current_path = current_cgroup_path();
  if (!current_path.ok()) return current_path.status();
  if (*current_path != impl_->cgroup_path) {
    return Status::FailedPrecondition(
        "Linux worker cgroup membership changed during prefault");
  }
  auto now = monotonic_now_ns();
  if (!now.ok()) return now.status();
  struct rusage usage {};
  if (::getrusage(RUSAGE_SELF, &usage) != 0) {
    return failure("getrusage artifact prefault");
  }
  if (usage.ru_majflt < 0) {
    return Status::FailedPrecondition(
        "Linux major-fault counter is negative");
  }
  auto status = read_bounded_path("/proc/self/status",
                                  "read /proc/self/status");
  if (!status.ok()) return status.status();
  auto vmpte = status_kib(*status, "VmPTE:");
  if (!vmpte.ok()) return vmpte.status();
  auto current = read_bounded_at(impl_->cgroup_directory.get(),
                                 "memory.current", "read memory.current");
  if (!current.ok()) return current.status();
  auto current_bytes = decimal(*current, "memory.current");
  if (!current_bytes.ok()) return current_bytes.status();
  auto stat = read_bounded_at(impl_->cgroup_directory.get(),
                              "memory.stat", "read memory.stat");
  if (!stat.ok()) return stat.status();
  auto memory = parse_memory_stat(*stat);
  if (!memory.ok()) return memory.status();
  return DeepSeekRankArtifactPrefaultResourceSnapshot{
      *now, static_cast<std::uint64_t>(usage.ru_majflt), *vmpte,
      *current_bytes, memory->file, memory->anon, memory->kernel};
}

Result<DeepSeekRankArtifactPrefaultRangeObservation>
LinuxDeepSeekRankArtifactPrefaultOperations::prefault_range(
    std::span<const std::byte> mapping, std::uint64_t page_bytes) {
  if (mapping.empty() ||
      page_bytes != DeepSeekRankArtifactPrefaultTransaction::kPageBytes ||
      (reinterpret_cast<std::uintptr_t>(mapping.data()) % page_bytes) != 0) {
    return Status::InvalidArgument(
        "Linux artifact prefault range is not 4 KiB page canonical");
  }
  const auto page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0 || static_cast<std::uint64_t>(page_size) != page_bytes) {
    return Status::FailedPrecondition(
        "Linux artifact prefault base page differs from profile");
  }
  const auto maximum_chunk_bytes =
      kMincoreChunkPages * static_cast<std::size_t>(page_bytes);
  std::uint64_t resident_before_pages = 0;
  std::uint64_t resident_after_pages = 0;
  std::uint64_t touched_pages = 0;
  for (std::size_t offset = 0; offset < mapping.size();) {
    const auto bytes =
        std::min(maximum_chunk_bytes, mapping.size() - offset);
    const auto* address = mapping.data() + offset;
    auto before = resident_pages(address, bytes,
                                 static_cast<std::size_t>(page_bytes));
    if (!before.ok()) return before.status();
    auto total = checked_add_u64(resident_before_pages, *before);
    if (!total.ok()) return total.status();
    resident_before_pages = *total;
    if (::madvise(const_cast<std::byte*>(address), bytes,
                  MADV_WILLNEED) != 0) {
      return failure("madvise selected artifact pages");
    }
    volatile std::uint8_t sink = 0;
    std::uint64_t chunk_touches = 0;
    for (std::size_t page_offset = 0; page_offset < bytes;
         page_offset += static_cast<std::size_t>(page_bytes)) {
      sink = static_cast<std::uint8_t>(
          sink ^ std::to_integer<std::uint8_t>(address[page_offset]));
      ++chunk_touches;
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    (void)sink;
    auto after = resident_pages(address, bytes,
                                static_cast<std::size_t>(page_bytes));
    if (!after.ok()) return after.status();
    if (*after != chunk_touches) {
      return Status::FailedPrecondition(
          "Linux selected artifact range is not fully resident");
    }
    total = checked_add_u64(resident_after_pages, *after);
    if (!total.ok()) return total.status();
    resident_after_pages = *total;
    total = checked_add_u64(touched_pages, chunk_touches);
    if (!total.ok()) return total.status();
    touched_pages = *total;
    offset += bytes;
  }
  auto resident_before_bytes =
      checked_mul_u64(resident_before_pages, page_bytes);
  if (!resident_before_bytes.ok()) return resident_before_bytes.status();
  auto resident_after_bytes =
      checked_mul_u64(resident_after_pages, page_bytes);
  if (!resident_after_bytes.ok()) return resident_after_bytes.status();
  return DeepSeekRankArtifactPrefaultRangeObservation{
      *resident_before_bytes, *resident_after_bytes, touched_pages};
}

}  // namespace pih
