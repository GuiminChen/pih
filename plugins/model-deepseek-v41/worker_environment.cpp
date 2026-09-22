#include "worker_environment.h"
#include <filesystem>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <new>
#include <utility>

namespace pih::deepseek_v41 {
namespace {
struct Fd { int value; ~Fd() { if (value >= 0) ::close(value); } };
Status Directory(const std::string& value) {
  if (value.empty() || value.size() > 1024 || value.front() != '/' || value.back() == '/' ||
      value.find_first_of(":;$\r\n") != value.npos || value.find('\0') != value.npos || value.find("//") != value.npos)
    return Status::InvalidArgument("Worker library directory must be canonical and absolute");
  const std::filesystem::path path(value);
  Fd directory{::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)};
  if (directory.value < 0) return Status::Unavailable("Cannot open library filesystem root");
  for (const auto& part : path.relative_path()) {
    const auto name = part.string();
    if (name.empty() || name == "." || name == "..")
      return Status::InvalidArgument("Worker library directory contains relative components");
    Fd next{::openat(directory.value, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)};
    struct stat state{};
    if (next.value < 0 || ::fstat(next.value, &state) || !S_ISDIR(state.st_mode) ||
        (state.st_mode & 0022) || (state.st_uid != 0 && state.st_uid != ::geteuid()))
      return Status::FailedPrecondition("Worker library path has links, untrusted ownership or shared write permissions");
    std::swap(directory.value, next.value);
  }
  return Status::Ok();
}
}
Result<std::unique_ptr<WorkerEnvironment>> WorkerEnvironment::Create(std::span<const std::string> directories) {
  if (directories.size() > 16) return Status::InvalidArgument("Worker library search list exceeds 16 directories");
  try {
    std::string search;
    for (std::size_t i = 0; i < directories.size(); ++i) {
      for (std::size_t j = 0; j < i; ++j)
        if (directories[i] == directories[j]) return Status::InvalidArgument("Duplicate worker library directory");
      const auto valid = Directory(directories[i]); if (!valid.ok()) return valid;
      if (i) search += ':';
      search += directories[i];
      if (search.size() > 4096) return Status::ResourceExhausted("Worker library search path exceeds byte bound");
    }
    auto owner = std::unique_ptr<WorkerEnvironment>(new WorkerEnvironment);
    owner->entries_ = {"LANG=C", "LC_ALL=C", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "NCCL_CONF_FILE=/dev/null"};
    if (!search.empty()) owner->entries_.push_back("LD_LIBRARY_PATH=" + search);
    return owner;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("Worker environment admission allocation failed");
  }
}
}  // namespace pih::deepseek_v41
