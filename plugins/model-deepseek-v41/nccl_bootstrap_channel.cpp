#include "nccl_bootstrap_channel.h"
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>

namespace pih::deepseek_v41 {
Result<std::array<std::byte, 144>> NcclBootstrapChannel::Encode(std::span<const std::byte> id) {
  if (id.size() != 128 || std::all_of(id.begin(), id.end(), [](auto b) { return b == std::byte{}; }))
    return Status::InvalidArgument("NCCL bootstrap ID must contain 128 nonempty bytes");
  std::array<std::byte, 144> packet{};
  std::memcpy(packet.data(), "PIHNCCL1", 8);
  packet[8] = std::byte{1};
  std::copy(id.begin(), id.end(), packet.begin() + 16);
  return packet;
}
Result<NcclBootstrapChannel::Id> NcclBootstrapChannel::Decode(std::span<const std::byte> packet) {
  if (packet.size() != 144 || std::memcmp(packet.data(), "PIHNCCL1", 8) || packet[8] != std::byte{1} ||
      !std::all_of(packet.begin() + 9, packet.begin() + 16, [](auto b) { return b == std::byte{}; }))
    return Status::FailedPrecondition("NCCL bootstrap packet framing invalid");
  Id id{};
  std::copy(packet.begin() + 16, packet.end(), id.begin());
  if (std::all_of(id.begin(), id.end(), [](auto b) { return b == std::byte{}; }))
    return Status::FailedPrecondition("NCCL bootstrap packet contains empty identity");
  return id;
}
Status NcclBootstrapChannel::ValidateSocket(int fd) {
  int type = 0;
  socklen_t type_bytes = sizeof(type);
  sockaddr_un peer{};
  socklen_t peer_bytes = sizeof(peer);
  const auto flags = ::fcntl(fd, F_GETFL);
  if (fd < 0 || flags < 0 || !(flags & O_NONBLOCK) ||
      ::getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &type_bytes) || type != SOCK_SEQPACKET ||
      ::getpeername(fd, reinterpret_cast<sockaddr*>(&peer), &peer_bytes) || peer.sun_family != AF_UNIX)
    return Status::FailedPrecondition("NCCL bootstrap requires private connected nonblocking UNIX packet socket");
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
