#include "nccl_bootstrap_channel.h"
#include <nccl.h>
#include <charconv>
#include <climits>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <poll.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <unistd.h>

#if NCCL_VERSION_CODE != 23102
#error "NCCL bootstrap helper requires 2.31.2"
#endif
static_assert(NCCL_UNIQUE_ID_BYTES == 128);
namespace {
using Channel = pih::deepseek_v41::NcclBootstrapChannel;
[[noreturn]] void Fail() {
  // Never log the bootstrap identifier or emit it through stdout/stderr.
  std::fputs("native NCCL bootstrap helper failed\n", stderr);
  std::_Exit(1);
}
void Wait(short events, Channel::Clock::time_point deadline) {
  for (;;) {
    if (Channel::Clock::now() >= deadline) Fail();
    pollfd fd{3, events, 0};
    const auto result = ::poll(&fd, 1, 10);
    if (result < 0 && errno == EINTR) continue;
    if (result < 0 || (fd.revents & (POLLERR | POLLHUP | POLLNVAL))) Fail();
    if (result > 0 && (fd.revents & events)) return;
  }
}
}
int main(int argc, char** argv) {
  try {
    // Only the admitted supervisor may enter this helper. FD 3 is the private
    // inherited socket; argv[1] is its absolute monotonic lifetime deadline.
    if (argc != 2 || !Channel::ValidateSocket(3).ok()) Fail();
    std::int64_t ns = 0;
    const auto length = std::strlen(argv[1]);
    if (!length || length > 19 || argv[1][0] == '0') Fail();
    const auto parsed = std::from_chars(argv[1], argv[1] + length, ns);
    if (parsed.ec != std::errc{} || parsed.ptr != argv[1] + length || ns <= 0) Fail();
    const auto deadline = Channel::Clock::time_point(std::chrono::duration_cast<Channel::Clock::duration>(std::chrono::nanoseconds(ns)));
    if (Channel::Clock::now() >= deadline) Fail();
    ucred peer{}; socklen_t peer_bytes = sizeof(peer);
    int signal = 0;
    if (::getsockopt(3, SOL_SOCKET, SO_PEERCRED, &peer, &peer_bytes) || peer_bytes != sizeof(peer) ||
        peer.pid != ::getppid() || peer.uid != ::geteuid() || peer.pid <= 1 ||
        ::prctl(PR_GET_PDEATHSIG, &signal) || signal != SIGKILL ||
        ::prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) != 1) Fail();
    int version = 0;
    if (ncclGetVersion(&version) != ncclSuccess || version != NCCL_VERSION_CODE) Fail();
    ncclUniqueId id{};
    if (ncclGetUniqueId(&id) != ncclSuccess) Fail();
    auto packet = Channel::Encode(std::as_bytes(std::span(id.internal)));
    volatile char* secret = id.internal;
    for (std::size_t i = 0; i < sizeof(id.internal); ++i) secret[i] = 0;
    if (!packet.ok()) Fail();
    for (;;) {
      Wait(POLLOUT, deadline);
      const auto sent = ::send(3, packet->data(), packet->size(), MSG_DONTWAIT | MSG_NOSIGNAL);
      if (sent < 0 && (errno == EINTR || errno == EAGAIN)) continue;
      if (sent != static_cast<ssize_t>(packet->size())) Fail();
      break;
    }
    volatile std::byte* encoded = packet->data();
    for (std::size_t i = 0; i < packet->size(); ++i) encoded[i] = std::byte{};
    // Returning here would kill NCCL's bootstrap listener thread. The parent
    // holds us until rank startup/retirement permits an explicit process kill.
    Wait(POLLIN, deadline);
    Fail(); // Any input, EOF, disconnect or deadline is terminal, never a new ID.
  } catch (...) { Fail(); }
}
