#include "pih/platform/linux/linux_engine_inet_diag_namespace_handle_probe.h"

#include <cerrno>
#include <string>

#include <fcntl.h>
#include <linux/magic.h>
#include <linux/netlink.h>
#include <sys/socket.h>
#include <sys/vfs.h>

namespace pih {
namespace {

Result<bool> descriptor_open(std::int32_t descriptor) {
  int result = 0;
  do {
    result = ::fcntl(descriptor, F_GETFD);
  } while (result < 0 && errno == EINTR);
  if (result >= 0) return true;
  const int error = errno;
  if (error == EBADF) return false;
  return Status::Unavailable("Linux descriptor probe failed with errno " +
                             std::to_string(error));
}

}  // namespace

Result<EngineInetDiagNamespaceHandleObservation>
LinuxEngineInetDiagNamespaceHandleProbe::inspect(
    std::int32_t namespace_descriptor,
    std::int32_t netlink_descriptor) {
  auto namespace_open = descriptor_open(namespace_descriptor);
  if (!namespace_open.ok()) return namespace_open.status();
  auto netlink_open = descriptor_open(netlink_descriptor);
  if (!netlink_open.ok()) return netlink_open.status();

  bool network_namespace = false;
  if (*namespace_open) {
    struct statfs filesystem {};
    int result = 0;
    do {
      result = ::fstatfs(namespace_descriptor, &filesystem);
    } while (result != 0 && errno == EINTR);
    if (result != 0) {
      return Status::Unavailable("Linux namespace fstatfs failed with errno " +
                                 std::to_string(errno));
    }
    network_namespace = filesystem.f_type == NSFS_MAGIC;
  }

  bool sock_diag = false;
  if (*netlink_open) {
    int domain = 0;
    int protocol = 0;
    socklen_t length = sizeof(int);
    if (::getsockopt(netlink_descriptor, SOL_SOCKET, SO_DOMAIN, &domain,
                     &length) != 0 || length != sizeof(int)) {
      return Status::Unavailable("Linux netlink domain probe failed");
    }
    length = sizeof(int);
    if (::getsockopt(netlink_descriptor, SOL_SOCKET, SO_PROTOCOL, &protocol,
                     &length) != 0 || length != sizeof(int)) {
      return Status::Unavailable("Linux netlink protocol probe failed");
    }
    sock_diag = domain == AF_NETLINK && protocol == NETLINK_SOCK_DIAG;
  }
  return EngineInetDiagNamespaceHandleObservation{
      *namespace_open, network_namespace, *netlink_open, sock_diag};
}

}  // namespace pih
