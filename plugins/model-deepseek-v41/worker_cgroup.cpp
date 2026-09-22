#include "worker_cgroup.h"
#include <array>
#include <fcntl.h>
#include <linux/magic.h>
#include <new>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <unistd.h>

namespace pih::deepseek_v41 {
namespace {
struct Fd { int value; ~Fd() { if (value >= 0) ::close(value); } };
Result<std::string> Read(int root, const char* name) {
  Fd fd{::openat(root, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)};
  if (fd.value < 0) return Status::Unavailable("Cannot open worker cgroup control for reading");
  std::array<char, 4097> buffer{};
  const auto count = ::read(fd.value, buffer.data(), buffer.size());
  if (count <= 0 || count == static_cast<ssize_t>(buffer.size()))
    return Status::Unavailable("Worker cgroup control read failed or exceeds bound");
  std::string text(buffer.data(), static_cast<std::size_t>(count));
  while (!text.empty() && (text.back() == '\n' || text.back() == ' ' || text.back() == '\t')) text.pop_back();
  return text;
}
Status Write(int root, const char* name, const std::string& value) {
  Fd fd{::openat(root, name, O_WRONLY | O_CLOEXEC | O_NOFOLLOW)};
  if (fd.value < 0) return Status::Unavailable("Cannot open worker cgroup control for writing");
  if (::write(fd.value, value.data(), value.size()) != static_cast<ssize_t>(value.size()))
    return Status::Unavailable("Worker cgroup control write failed or partial; retain group");
  return Status::Ok();
}
Status Set(int root, const char* name, const std::string& value) {
  auto status = Write(root, name, value); if (!status.ok()) return status;
  auto actual = Read(root, name); if (!actual.ok()) return actual.status();
  if (*actual != value) return Status::FailedPrecondition("Worker cgroup limit readback mismatch");
  return Status::Ok();
}
}
WorkerCgroup::~WorkerCgroup() {
  if (directory_ >= 0) ::close(directory_);
  if (parent_ >= 0) ::close(parent_);
}
Status WorkerCgroup::Create(int parent, const std::string& leaf, WorkerCgroupLimits limits) {
  if (parent_ >= 0 || created_) return Status::FailedPrecondition("Worker cgroup owner is single-use");
  const auto page = ::sysconf(_SC_PAGESIZE);
  if (parent < 0 || leaf.size() < 16 || leaf.size() > 96 || !limits.memory_bytes || page <= 0 ||
      limits.memory_bytes % static_cast<std::uint64_t>(page) || !limits.pids || limits.pids > 65536 ||
      limits.cpu_period_us < 1000 || limits.cpu_period_us > 1000000 ||
      limits.cpu_quota_us < 1000 || limits.cpu_quota_us > 1000000000)
    return Status::InvalidArgument("Worker cgroup leaf or limits invalid");
  for (const char c : leaf) if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-'))
    return Status::InvalidArgument("Worker cgroup name must be a single canonical leaf");
  try {
    leaf_ = leaf;
    parent_ = ::fcntl(parent, F_DUPFD_CLOEXEC, 0);
    struct stat info{}; struct statfs fs{};
    if (parent_ < 0 || ::fstat(parent_, &info) || !S_ISDIR(info.st_mode) ||
        ::fstatfs(parent_, &fs) || fs.f_type != CGROUP2_SUPER_MAGIC)
      return Status::FailedPrecondition("Worker parent is not an admitted cgroup-v2 directory");
    if (::mkdirat(parent_, leaf_.c_str(), 0700))
      return Status::Unavailable("Cannot create fresh worker cgroup; existing groups are never adopted");
    created_ = true;
    directory_ = ::openat(parent_, leaf_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (directory_ < 0) return Status::Unavailable("Cannot retain newly created worker cgroup");
    auto empty = Empty(); if (!empty.ok()) return empty.status();
    if (!*empty) return Status::FailedPrecondition("New worker cgroup unexpectedly populated");
    auto status = Set(directory_, "memory.max", std::to_string(limits.memory_bytes)); if (!status.ok()) return status;
    status = Set(directory_, "memory.swap.max", "0"); if (!status.ok()) return status;
    status = Set(directory_, "memory.oom.group", "1"); if (!status.ok()) return status;
    status = Set(directory_, "pids.max", std::to_string(limits.pids)); if (!status.ok()) return status;
    status = Set(directory_, "cpu.max", std::to_string(limits.cpu_quota_us) + " " + std::to_string(limits.cpu_period_us));
    if (!status.ok()) return status;
    auto type = Read(directory_, "cgroup.type"); if (!type.ok()) return type.status();
    if (*type != "domain") return Status::FailedPrecondition("Worker cgroup is not a domain");
    ready_ = true; return Status::Ok();
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Worker cgroup configuration allocation failed; retain group"); }
}
Status WorkerCgroup::SameMember() const {
  if (!created_ || removed_ || directory_ < 0 || parent_ < 0)
    return Status::FailedPrecondition("Worker cgroup has no retained owned member");
  struct stat owned{}, named{};
  if (::fstat(directory_, &owned) || ::fstatat(parent_, leaf_.c_str(), &named, AT_SYMLINK_NOFOLLOW) ||
      !S_ISDIR(named.st_mode) || owned.st_dev != named.st_dev || owned.st_ino != named.st_ino)
    return Status::FailedPrecondition("Worker cgroup name no longer identifies the owned directory");
  return Status::Ok();
}
Result<int> WorkerCgroup::ReadyDescriptor() const {
  if (!ready_ || killed_) return Status::FailedPrecondition("Worker cgroup limits are not admitted ready");
  auto status = SameMember(); if (!status.ok()) return status;
  return directory_;
}
Result<bool> WorkerCgroup::Empty() const {
  auto status = SameMember(); if (!status.ok()) return status;
  auto events = Read(directory_, "cgroup.events"); if (!events.ok()) return events.status();
  std::string_view remaining = *events;
  bool seen = false, empty = false;
  while (!remaining.empty()) {
    const auto end = remaining.find('\n'); const auto line = remaining.substr(0, end);
    if (line.starts_with("populated ")) {
      if (seen || (line != "populated 0" && line != "populated 1"))
        return Status::FailedPrecondition("Worker cgroup populated observation invalid");
      seen = true; empty = line == "populated 0";
    }
    if (end == std::string_view::npos) break;
    remaining.remove_prefix(end + 1);
  }
  if (!seen) return Status::FailedPrecondition("Worker cgroup events omit populated state");
  return empty;
}
Status WorkerCgroup::Kill() {
  if (killed_) return Status::Ok();
  auto status = SameMember(); if (!status.ok()) return status;
  ready_ = false;
  status = Write(directory_, "cgroup.kill", "1"); if (!status.ok()) return status;
  killed_ = true; return Status::Ok();
}
Status WorkerCgroup::Remove() {
  if (removed_) return Status::Ok();
  auto empty = Empty(); if (!empty.ok()) return empty.status();
  if (!*empty) return Status::FailedPrecondition("Worker cgroup is still populated");
  auto same = SameMember(); if (!same.ok()) return same;
  if (::unlinkat(parent_, leaf_.c_str(), AT_REMOVEDIR))
    return Status::Unavailable("Cannot remove owned empty worker cgroup; nested groups are not recursively removed");
  ready_ = false; removed_ = true; return Status::Ok();
}
}  // namespace pih::deepseek_v41
