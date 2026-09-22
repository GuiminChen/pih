#include "pih/platform/linux/linux_engine_termination_signal_source.h"

#include <cerrno>
#include <limits>
#include <string>
#include <utility>

#include <pthread.h>
#include <signal.h>
#include <sys/signalfd.h>
#include <unistd.h>

namespace pih {
namespace {

Status failure(const char* operation, int error = errno) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(error));
}

}  // namespace

Result<LinuxEngineTerminationSignalSource>
LinuxEngineTerminationSignalSource::Create(std::uint64_t engine_generation) {
  if (engine_generation == 0)
    return Status::InvalidArgument(
        "engine termination signal generation is invalid");
  sigset_t mask{};
  if (::sigemptyset(&mask) != 0 || ::sigaddset(&mask, SIGTERM) != 0 ||
      ::sigaddset(&mask, SIGINT) != 0)
    return failure("prepare engine termination signal mask");
  sigset_t previous_mask{};
  const int mask_status =
      ::pthread_sigmask(SIG_BLOCK, &mask, &previous_mask);
  if (mask_status != 0)
    return failure("block engine termination signals", mask_status);
  const int fd = ::signalfd(-1, &mask, SFD_CLOEXEC | SFD_NONBLOCK);
  if (fd < 0) {
    const auto status = failure("create engine termination signalfd");
    (void)::pthread_sigmask(SIG_SETMASK, &previous_mask, nullptr);
    return status;
  }
  return LinuxEngineTerminationSignalSource(engine_generation, fd);
}

LinuxEngineTerminationSignalSource::~LinuxEngineTerminationSignalSource() {
  reset();
}

LinuxEngineTerminationSignalSource::LinuxEngineTerminationSignalSource(
    LinuxEngineTerminationSignalSource&& other) noexcept
    : generation_(std::exchange(other.generation_, 0)),
      next_event_identity_(std::exchange(other.next_event_identity_, 1)),
      fd_(std::exchange(other.fd_, -1)) {}

LinuxEngineTerminationSignalSource&
LinuxEngineTerminationSignalSource::operator=(
    LinuxEngineTerminationSignalSource&& other) noexcept {
  if (this != &other) {
    reset();
    generation_ = std::exchange(other.generation_, 0);
    next_event_identity_ = std::exchange(other.next_event_identity_, 1);
    fd_ = std::exchange(other.fd_, -1);
  }
  return *this;
}

void LinuxEngineTerminationSignalSource::reset() noexcept {
  if (fd_ >= 0) (void)::close(fd_);
  fd_ = -1;
}

Result<std::optional<EngineTerminationEvent>>
LinuxEngineTerminationSignalSource::poll() {
  if (fd_ < 0 || generation_ == 0)
    return Status::FailedPrecondition(
        "engine termination signal source is closed");
  if (next_event_identity_ == std::numeric_limits<std::uint64_t>::max())
    return Status::ResourceExhausted(
        "engine termination signal identity exhausted");
  signalfd_siginfo info{};
  const auto count = ::read(fd_, &info, sizeof(info));
  if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
    return std::optional<EngineTerminationEvent>{};
  if (count < 0) return failure("read engine termination signalfd");
  if (count != static_cast<ssize_t>(sizeof(info)))
    return Status::Internal(
        "engine termination signalfd record was truncated");
  EngineTerminationEventKind kind;
  if (info.ssi_signo == SIGTERM)
    kind = EngineTerminationEventKind::kSigterm;
  else if (info.ssi_signo == SIGINT)
    kind = EngineTerminationEventKind::kSigint;
  else
    return Status::FailedPrecondition(
        "engine termination signalfd produced an unexpected signal");
  return std::optional<EngineTerminationEvent>{EngineTerminationEvent{
      generation_, next_event_identity_++, kind}};
}

}  // namespace pih
