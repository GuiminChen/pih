#include "pih/platform/linux/linux_engine_artifact_descriptor_stat_backend.h"

#include <cerrno>
#include <string>

#include <sys/stat.h>
#include <unistd.h>

namespace pih {

Result<EngineArtifactDescriptorStat>
LinuxEngineArtifactDescriptorStatBackend::stat(std::int32_t descriptor) {
  if (descriptor < 0) {
    return Status::InvalidArgument(
        "Linux artifact descriptor is negative");
  }
  struct stat value {};
  int result = 0;
  do {
    result = ::fstat(descriptor, &value);
  } while (result != 0 && errno == EINTR);
  if (result != 0) {
    const int error = errno;
    if (error == EBADF) return EngineArtifactDescriptorStat{};
    return Status::Unavailable(
        "Linux artifact descriptor fstat failed with errno " +
        std::to_string(error));
  }
  if (value.st_size < 0) {
    return Status::FailedPrecondition(
        "Linux artifact descriptor size is negative");
  }
  return EngineArtifactDescriptorStat{
      true, S_ISREG(value.st_mode) != 0,
      static_cast<std::uint64_t>(value.st_dev),
      static_cast<std::uint64_t>(value.st_ino),
      static_cast<std::uint64_t>(value.st_size)};
}

}  // namespace pih
