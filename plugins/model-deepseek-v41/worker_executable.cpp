#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "worker_executable.h"
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include <new>

namespace pih::deepseek_v41 {
namespace {
struct Fd {
  int value;
  ~Fd() { if (value >= 0) ::close(value); }
};
constexpr int kSeals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
bool Same(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
      a.st_mode == b.st_mode && a.st_uid == b.st_uid && a.st_gid == b.st_gid &&
      a.st_nlink == b.st_nlink && a.st_mtim.tv_sec == b.st_mtim.tv_sec &&
      a.st_mtim.tv_nsec == b.st_mtim.tv_nsec && a.st_ctim.tv_sec == b.st_ctim.tv_sec &&
      a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
}
WorkerExecutable::~WorkerExecutable() { if (fd_ >= 0) ::close(fd_); }
Result<int> WorkerExecutable::Descriptor() const {
  struct stat state{};
  const auto seals = ::fcntl(fd_, F_GET_SEALS);
  if (fd_ < 0 || seals < 0 || (seals & kSeals) != kSeals || ::fstat(fd_, &state) ||
      !S_ISREG(state.st_mode) || state.st_size < 64 ||
      static_cast<std::uint64_t>(state.st_size) != bytes_ || (state.st_mode & 07777) != 0500)
    return Status::FailedPrecondition("Worker executable snapshot is not sealed and executable");
  return fd_;
}
Result<std::unique_ptr<WorkerExecutable>> WorkerExecutable::Open(const std::filesystem::path& path,
    const Sha256Digest& expected, std::uint64_t budget, Clock::time_point deadline) {
  if (!path.is_absolute() || expected == Sha256Digest{} || budget < 64 || budget > (512ULL << 20) || Clock::now() >= deadline)
    return Status::InvalidArgument("Worker executable path, digest, budget or deadline invalid");
  try {
    Fd directory{::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)};
    if (directory.value < 0) return Status::Unavailable("Cannot open executable filesystem root");
    for (const auto& part : path.parent_path().relative_path()) {
      if (Clock::now() >= deadline) return Status::DeadlineExceeded("Worker executable path admission expired");
      const auto name = part.string();
      if (name.empty() || name == "." || name == ".." || name.find('\0') != name.npos)
        return Status::InvalidArgument("Worker executable path is not canonical");
      Fd next{::openat(directory.value, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)};
      if (next.value < 0) return Status::FailedPrecondition("Worker executable parent cannot be opened without links");
      std::swap(directory.value, next.value);
    }
    const auto name = path.filename().string();
    if (name.empty() || name == "." || name == ".." || name.find('\0') != name.npos)
      return Status::InvalidArgument("Worker executable member name invalid");
    Fd source{::openat(directory.value, name.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)};
    struct stat before{}, after{};
    if (source.value < 0 || ::fstat(source.value, &before) || !S_ISREG(before.st_mode) ||
        before.st_nlink != 1 || (before.st_mode & (0222 | S_ISUID | S_ISGID)) || !(before.st_mode & 0111) ||
        before.st_size < 64 || static_cast<std::uint64_t>(before.st_size) > budget)
      return Status::FailedPrecondition("Worker executable must be bounded, read-only, single-link ELF storage");
    Fd snapshot{static_cast<int>(::syscall(SYS_memfd_create, "pih-v41-worker", MFD_CLOEXEC | MFD_ALLOW_SEALING | MFD_EXEC))};
    if (snapshot.value < 0) return Status::Unavailable("Executable memfd unavailable; no mutable-path fallback");
    std::vector<std::byte> chunk(1U << 20);
    Sha256 hash;
    std::uint64_t offset = 0;
    while (offset < static_cast<std::uint64_t>(before.st_size)) {
      if (Clock::now() >= deadline) return Status::DeadlineExceeded("Worker executable admission expired");
      const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(chunk.size(), before.st_size - offset));
      std::size_t read = 0;
      while (read < count) {
        if (Clock::now() >= deadline) return Status::DeadlineExceeded("Worker executable read expired");
        const auto n = ::pread(source.value, chunk.data() + read, count - read, offset + read);
        if (n < 0 && errno == EINTR) {
          if (Clock::now() >= deadline) return Status::DeadlineExceeded("Worker executable read expired");
          continue;
        }
        if (n <= 0) return Status::Unavailable("Worker executable snapshot read failed");
        read += static_cast<std::size_t>(n);
      }
      if (!offset) {
        const auto* h = reinterpret_cast<const unsigned char*>(chunk.data());
        if (h[0] != 127 || h[1] != 'E' || h[2] != 'L' || h[3] != 'F' ||
            h[4] != 2 || h[5] != 1 || h[6] != 1 || (h[16] != 2 && h[16] != 3) ||
            h[17] || h[18] != 62 || h[19])
          return Status::FailedPrecondition("Worker snapshot requires x86-64 little-endian executable ELF");
      }
      const auto updated = hash.update({chunk.data(), count}); if (!updated.ok()) return updated;
      std::size_t written = 0;
      while (written < count) {
        if (Clock::now() >= deadline) return Status::DeadlineExceeded("Worker executable copy expired");
        const auto n = ::pwrite(snapshot.value, chunk.data() + written, count - written, offset + written);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return Status::Unavailable("Worker executable snapshot write failed");
        written += static_cast<std::size_t>(n);
      }
      offset += count;
    }
    if (::fstat(source.value, &after) || !Same(before, after))
      return Status::FailedPrecondition("Worker executable changed during admission");
    const auto digest = hash.finalize(); if (!digest.ok()) return digest.status();
    if (*digest != expected) return Status::FailedPrecondition("Worker executable digest differs from deployment");
    if (::fchmod(snapshot.value, 0500) || ::fcntl(snapshot.value, F_ADD_SEALS, kSeals))
      return Status::Unavailable("Cannot seal executable snapshot");
    auto owner = std::unique_ptr<WorkerExecutable>(new WorkerExecutable);
    owner->bytes_ = offset; owner->fd_ = std::exchange(snapshot.value, -1);
    const auto ready = owner->Descriptor(); if (!ready.ok()) return ready.status();
    if (Clock::now() >= deadline) return Status::DeadlineExceeded("Worker executable admission completed late");
    return owner;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("Worker executable admission allocation failed");
  }
}
}  // namespace pih::deepseek_v41
