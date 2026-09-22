#include "pih/platform/linux/linux_engine_listener_kernel_backend.h"

#include <cerrno>
#include <optional>
#include <string>

#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pih {
namespace {

Result<std::optional<struct stat>> descriptor_stat(std::int32_t descriptor) {
  struct stat value {};
  int result = 0;
  do {
    result = ::fstat(descriptor, &value);
  } while (result != 0 && errno == EINTR);
  if (result == 0) return std::optional<struct stat>{value};
  const int error = errno;
  if (error == EBADF) return std::optional<struct stat>{};
  return Status::Unavailable(
      "Linux listener fstat failed with errno " + std::to_string(error));
}

}  // namespace

Result<EngineListenerKernelState>
LinuxEngineListenerKernelBackend::inspect(std::int32_t descriptor) {
  if (descriptor < 0) {
    return Status::InvalidArgument("Linux listener descriptor is negative");
  }
  auto before = descriptor_stat(descriptor);
  if (!before.ok()) return before.status();
  if (!before->has_value()) return EngineListenerKernelState{};
  if (!S_ISSOCK(before->value().st_mode)) {
    return EngineListenerKernelState{
        true, false, false,
        static_cast<std::uint64_t>(before->value().st_dev),
        static_cast<std::uint64_t>(before->value().st_ino)};
  }

  int accepting = 0;
  socklen_t length = sizeof(accepting);
  int result = 0;
  do {
    result = ::getsockopt(descriptor, SOL_SOCKET, SO_ACCEPTCONN, &accepting,
                          &length);
  } while (result != 0 && errno == EINTR);
  if (result != 0) {
    const int error = errno;
    if (error == EBADF) return EngineListenerKernelState{};
    if (error == ENOTSOCK) {
      return Status::FailedPrecondition(
          "Linux listener descriptor changed type during observation");
    }
    return Status::Unavailable(
        "Linux listener getsockopt failed with errno " +
        std::to_string(error));
  }
  if (length != sizeof(accepting)) {
    return Status::FailedPrecondition(
        "Linux listener SO_ACCEPTCONN length drifted");
  }

  auto after = descriptor_stat(descriptor);
  if (!after.ok()) return after.status();
  if (!after->has_value()) return EngineListenerKernelState{};
  if (before->value().st_dev != after->value().st_dev ||
      before->value().st_ino != after->value().st_ino ||
      before->value().st_mode != after->value().st_mode) {
    return Status::FailedPrecondition(
        "Linux listener descriptor changed during observation");
  }
  return EngineListenerKernelState{
      true, true, accepting != 0,
      static_cast<std::uint64_t>(after->value().st_dev),
      static_cast<std::uint64_t>(after->value().st_ino)};
}

}  // namespace pih
