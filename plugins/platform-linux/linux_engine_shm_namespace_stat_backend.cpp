#include "pih/platform/linux/linux_engine_shm_namespace_stat_backend.h"

#include <cerrno>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>

namespace pih {

Result<EngineShmNamespaceStat>
LinuxEngineShmNamespaceStatBackend::stat_at(
    std::int32_t directory_descriptor, std::string_view basename) {
  if (directory_descriptor < 0 || basename.empty() ||
      basename.find('\0') != std::string_view::npos) {
    return Status::InvalidArgument(
        "Linux SHM namespace stat arguments are invalid");
  }
  const std::string name(basename);
  struct stat value {};
  int result = 0;
  do {
    result = ::fstatat(directory_descriptor, name.c_str(), &value,
                       AT_SYMLINK_NOFOLLOW);
  } while (result != 0 && errno == EINTR);
  if (result != 0) {
    const int error = errno;
    if (error == ENOENT) return EngineShmNamespaceStat{};
    return Status::Unavailable(
        "Linux SHM namespace fstatat failed with errno " +
        std::to_string(error));
  }
  if (value.st_size < 0) {
    return Status::FailedPrecondition(
        "Linux SHM namespace object size is negative");
  }
  return EngineShmNamespaceStat{
      true, S_ISREG(value.st_mode) != 0,
      static_cast<std::uint64_t>(value.st_dev),
      static_cast<std::uint64_t>(value.st_ino),
      static_cast<std::uint64_t>(value.st_size)};
}

}  // namespace pih
