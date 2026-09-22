#include "pih/platform/linux/linux_deepseek_rank_process_driver.h"
#include "pih/model/deepseek_rank_control_codec.h"

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <poll.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

namespace pih {
namespace {

void close_fd(int fd) noexcept { if (fd >= 0) (void)::close(fd); }

int open_pidfd(pid_t pid) noexcept {
  return static_cast<int>(::syscall(SYS_pidfd_open, pid, 0));
}

int signal_pidfd(int pidfd, int signal) noexcept {
  return static_cast<int>(::syscall(SYS_pidfd_send_signal, pidfd, signal,
                                    nullptr, 0));
}

Status system_failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

bool pidfd_names_process(int fd, std::uint64_t expected) {
  std::ifstream input("/proc/self/fdinfo/" + std::to_string(fd));
  std::string label;
  std::uint64_t value = 0;
  while (input >> label >> value) {
    if (label == "Pid:") return value == expected;
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return false;
}

}  // namespace

Result<LinuxDeepSeekRankProcessDriver>
LinuxDeepSeekRankProcessDriver::Create(
    std::string worker_executable, std::vector<std::string> fixed_arguments,
    std::uint64_t expected_controller_process_identity,
    int inherited_controller_pidfd, bool expected_dspark_enabled,
    std::uint64_t maximum_metadata_reassembly_bytes) {
  if (worker_executable.empty() || worker_executable.front() != '/' ||
      expected_controller_process_identity == 0 ||
      expected_controller_process_identity != static_cast<std::uint64_t>(::getpid()) ||
      inherited_controller_pidfd < 0 ||
      maximum_metadata_reassembly_bytes == 0 ||
      maximum_metadata_reassembly_bytes >
          kDeepSeekRankArtifactMetadataBlobMaximumBytes ||
      ::fcntl(inherited_controller_pidfd, F_GETFD) < 0 ||
      !pidfd_names_process(inherited_controller_pidfd,
                           expected_controller_process_identity) ||
      ::access(worker_executable.c_str(), X_OK) != 0) {
    return Status::InvalidArgument("Linux DeepSeek rank process driver identity is invalid");
  }
  for (const auto& argument : fixed_arguments) {
    if (argument.empty() || argument.rfind("--pih-", 0) == 0)
      return Status::InvalidArgument("Linux DeepSeek fixed worker argument is empty");
  }
  return LinuxDeepSeekRankProcessDriver(
      std::move(worker_executable), std::move(fixed_arguments),
      expected_controller_process_identity, inherited_controller_pidfd,
      expected_dspark_enabled, maximum_metadata_reassembly_bytes);
}

LinuxDeepSeekRankProcessDriver::LinuxDeepSeekRankProcessDriver(
    std::string executable, std::vector<std::string> arguments,
    std::uint64_t controller, int controller_pidfd,
    bool expected_dspark_enabled,
    std::uint64_t maximum_metadata_reassembly_bytes) noexcept
    : executable_(std::move(executable)), arguments_(std::move(arguments)),
      controller_(controller), controller_pidfd_(controller_pidfd),
      expected_dspark_enabled_(expected_dspark_enabled),
      maximum_metadata_reassembly_bytes_(
          maximum_metadata_reassembly_bytes) {}

LinuxDeepSeekRankProcessDriver::~LinuxDeepSeekRankProcessDriver() { reset(); }

LinuxDeepSeekRankProcessDriver::LinuxDeepSeekRankProcessDriver(
    LinuxDeepSeekRankProcessDriver&& other) noexcept
    : executable_(std::move(other.executable_)),
      arguments_(std::move(other.arguments_)), controller_(other.controller_),
      controller_pidfd_(other.controller_pidfd_),
      expected_dspark_enabled_(other.expected_dspark_enabled_),
      maximum_metadata_reassembly_bytes_(
          other.maximum_metadata_reassembly_bytes_),
      processes_(std::move(other.processes_)) {
  other.processes_.clear();
}

LinuxDeepSeekRankProcessDriver& LinuxDeepSeekRankProcessDriver::operator=(
    LinuxDeepSeekRankProcessDriver&& other) noexcept {
  if (this != &other) {
    reset(); executable_ = std::move(other.executable_);
    arguments_ = std::move(other.arguments_); controller_ = other.controller_;
    controller_pidfd_ = other.controller_pidfd_;
    expected_dspark_enabled_ = other.expected_dspark_enabled_;
    maximum_metadata_reassembly_bytes_ =
        other.maximum_metadata_reassembly_bytes_;
    processes_ = std::move(other.processes_); other.processes_.clear();
  }
  return *this;
}

void LinuxDeepSeekRankProcessDriver::reset() noexcept {
  for (auto& process : processes_) {
    if (process.pidfd >= 0) (void)signal_pidfd(process.pidfd, SIGKILL);
  }
  for (auto& process : processes_) {
    if (process.pid > 0) (void)::waitpid(process.pid, nullptr, 0);
    close_fd(process.control); close_fd(process.pidfd);
  }
  processes_.clear();
}

LinuxDeepSeekRankProcessDriver::OwnedProcess*
LinuxDeepSeekRankProcessDriver::find(
    const DeepSeekRankProcessHandle& handle) noexcept {
  for (auto& process : processes_) {
    if (handle.process_identity == static_cast<std::uint64_t>(process.pid) &&
        handle.pidfd_identity == static_cast<std::uint64_t>(process.pidfd) + 1U &&
        handle.control_identity == static_cast<std::uint64_t>(process.control) + 1U)
      return &process;
  }
  return nullptr;
}

Result<DeepSeekRankProcessHandle> LinuxDeepSeekRankProcessDriver::spawn(
    const DeepSeekRankProcessManifest& manifest) {
  int control[2]{-1, -1};
  int errors[2]{-1, -1};
  if (::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, control) != 0)
    return system_failure("socketpair");
  const int pass_credentials = 1;
  if (::setsockopt(control[0], SOL_SOCKET, SO_PASSCRED, &pass_credentials,
                   sizeof(pass_credentials)) != 0 ||
      ::setsockopt(control[1], SOL_SOCKET, SO_PASSCRED, &pass_credentials,
                   sizeof(pass_credentials)) != 0) {
    close_fd(control[0]);
    close_fd(control[1]);
    return system_failure("setsockopt SO_PASSCRED");
  }
  if (::pipe2(errors, O_CLOEXEC) != 0) {
    close_fd(control[0]); close_fd(control[1]); return system_failure("pipe2");
  }
  std::vector<std::string> values = arguments_;
  values.push_back("--pih-rank=" + std::to_string(manifest.rank));
  values.push_back("--pih-world-size=" + std::to_string(manifest.world_size));
  values.push_back("--pih-engine-epoch=" + std::to_string(manifest.engine_epoch));
  values.push_back("--pih-worker-generation=" + std::to_string(manifest.worker_generation));
  values.push_back("--pih-device-identity=" + std::to_string(manifest.physical_device_identity));
  values.push_back("--pih-startup-device-ordinal=" +
                   std::to_string(manifest.startup_device_ordinal));
  values.push_back("--pih-startup-deadline-ns=" +
                   std::to_string(manifest.startup_deadline_ns));
  values.push_back("--pih-process-manifest=" + std::to_string(manifest.process_manifest_identity));
  values.push_back("--pih-controller-pid=" + std::to_string(controller_));
  values.push_back("--pih-controller-pidfd=" + std::to_string(controller_pidfd_));
  values.push_back("--pih-control-fd=" + std::to_string(control[1]));
  values.push_back("--pih-dspark-enabled=" +
                   std::to_string(expected_dspark_enabled_ ? 1 : 0));
  values.push_back("--pih-metadata-reassembly-bytes=" +
                   std::to_string(maximum_metadata_reassembly_bytes_));
  std::vector<char*> argv; argv.reserve(values.size() + 2);
  argv.push_back(executable_.data());
  for (auto& value : values) argv.push_back(value.data());
  argv.push_back(nullptr);

  const pid_t pid = ::fork();
  if (pid < 0) {
    close_fd(control[0]); close_fd(control[1]); close_fd(errors[0]); close_fd(errors[1]);
    return system_failure("fork");
  }
  if (pid == 0) {
    close_fd(control[0]); close_fd(errors[0]);
    int child_error = 0;
    if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 ||
        static_cast<std::uint64_t>(::getppid()) != controller_ ||
        ::fcntl(control[1], F_SETFD, 0) != 0 ||
        ::fcntl(controller_pidfd_, F_SETFD, 0) != 0) {
      child_error = errno == 0 ? ECHILD : errno;
      (void)::write(errors[1], &child_error, sizeof(child_error)); _exit(126);
    }
    ::execv(executable_.c_str(), argv.data());
    child_error = errno; (void)::write(errors[1], &child_error, sizeof(child_error));
    _exit(127);
  }
  close_fd(control[1]); close_fd(errors[1]);
  const int pidfd = open_pidfd(pid);
  int exec_error = 0;
  const auto read_bytes = ::read(errors[0], &exec_error, sizeof(exec_error));
  close_fd(errors[0]);
  if (pidfd < 0 || read_bytes != 0) {
    if (pidfd >= 0) (void)signal_pidfd(pidfd, SIGKILL); else (void)::kill(pid, SIGKILL);
    (void)::waitpid(pid, nullptr, 0); close_fd(pidfd); close_fd(control[0]);
    errno = exec_error == 0 ? errno : exec_error;
    return system_failure("rank exec handshake");
  }
  processes_.push_back({pid, pidfd, control[0]});
  return DeepSeekRankProcessHandle{
      static_cast<std::uint64_t>(pid), static_cast<std::uint64_t>(pidfd) + 1U,
      static_cast<std::uint64_t>(control[0]) + 1U};
}

Result<DeepSeekRankProcessObservation>
LinuxDeepSeekRankProcessDriver::observe(const DeepSeekRankProcessHandle& handle) {
  auto* process = find(handle);
  if (process == nullptr) return Status::FailedPrecondition("unknown DeepSeek rank process handle");
  pollfd descriptor{process->pidfd, POLLIN, 0};
  const int result = ::poll(&descriptor, 1, 0);
  if (result < 0) return system_failure("pidfd poll");
  if (result == 0) return DeepSeekRankProcessObservation::kRunning;
  siginfo_t info{};
  if (::waitid(P_PIDFD, static_cast<id_t>(process->pidfd), &info,
               WEXITED | WNOHANG | WNOWAIT) != 0)
    return system_failure("waitid pidfd");
  return info.si_code == CLD_EXITED && info.si_status == 0
      ? DeepSeekRankProcessObservation::kExitedSuccess
      : DeepSeekRankProcessObservation::kExitedFailure;
}

Status LinuxDeepSeekRankProcessDriver::terminate(
    const DeepSeekRankProcessHandle& handle) {
  auto* process = find(handle);
  if (process == nullptr) return Status::FailedPrecondition("unknown DeepSeek rank process handle");
  if (signal_pidfd(process->pidfd, SIGTERM) != 0 && errno != ESRCH)
    return system_failure("pidfd_send_signal");
  return Status::Ok();
}

Status LinuxDeepSeekRankProcessDriver::send_challenge(
    const DeepSeekRankProcessHandle& handle,
    const DeepSeekRankExecChallenge& challenge) {
  auto* process = find(handle);
  if (process == nullptr)
    return Status::FailedPrecondition("unknown DeepSeek rank process handle");
  const auto frame = encode_deepseek_rank_challenge(challenge);
  const auto written = ::send(process->control, frame.data(), frame.size(),
                              MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
    return Status::Unavailable("DeepSeek rank challenge channel is backpressured");
  if (written != static_cast<ssize_t>(frame.size()))
    return written < 0 ? system_failure("send rank challenge")
                       : Status::Internal("DeepSeek rank challenge was partially sent");
  return Status::Ok();
}

Result<std::optional<DeepSeekRankExecReady>>
LinuxDeepSeekRankProcessDriver::poll_ready(
    const DeepSeekRankProcessHandle& handle) {
  auto* process = find(handle);
  if (process == nullptr)
    return Status::FailedPrecondition("unknown DeepSeek rank process handle");
  std::array<std::byte, kDeepSeekRankReadyBytes> frame{};
  const auto received = ::recv(process->control, frame.data(), frame.size(),
                               MSG_DONTWAIT | MSG_TRUNC);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
    return std::optional<DeepSeekRankExecReady>{};
  if (received == 0)
    return Status::Unavailable("DeepSeek rank control channel closed before ready");
  if (received < 0) return system_failure("recv rank ready");
  if (received != static_cast<ssize_t>(frame.size()))
    return Status::InvalidArgument("DeepSeek rank ready frame size is invalid");
  auto decoded = decode_deepseek_rank_ready(frame);
  if (!decoded.ok()) return decoded.status();
  return std::optional<DeepSeekRankExecReady>{std::move(*decoded)};
}

Status LinuxDeepSeekRankProcessDriver::send_authority(
    const DeepSeekRankProcessHandle& handle,
    std::span<const std::byte> frame) {
  auto* process = find(handle);
  if (process == nullptr) {
    return Status::FailedPrecondition(
        "unknown DeepSeek rank process handle");
  }
  if (frame.size() != kDeepSeekRankPostExecResourceAuthorityBytes) {
    return Status::InvalidArgument(
        "DeepSeek rank resource authority frame size is invalid");
  }
  const auto written = ::send(process->control, frame.data(), frame.size(),
                              MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek rank resource authority channel is backpressured");
  }
  if (written < 0) return system_failure("send rank resource authority");
  if (written != static_cast<ssize_t>(frame.size())) {
    return Status::Internal(
        "DeepSeek rank resource authority frame was partially sent");
  }
  return Status::Ok();
}

Result<std::optional<DeepSeekRankPostExecResourceObservation>>
LinuxDeepSeekRankProcessDriver::poll_observation(
    const DeepSeekRankProcessHandle& handle) {
  auto* process = find(handle);
  if (process == nullptr) {
    return Status::FailedPrecondition(
        "unknown DeepSeek rank process handle");
  }
  std::array<std::byte,
             kDeepSeekRankPostExecResourceObservationBytes> frame{};
  const auto received = ::recv(process->control, frame.data(), frame.size(),
                               MSG_DONTWAIT | MSG_TRUNC);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return std::optional<DeepSeekRankPostExecResourceObservation>{};
  }
  if (received == 0) {
    return Status::FailedPrecondition(
        "DeepSeek rank control channel closed before resource observation");
  }
  if (received < 0) return system_failure("recv rank resource observation");
  if (received != static_cast<ssize_t>(frame.size())) {
    return Status::InvalidArgument(
        "DeepSeek rank resource observation frame size is invalid");
  }
  auto decoded = decode_deepseek_rank_post_exec_resource_observation(frame);
  if (!decoded.ok()) return decoded.status();
  return std::optional<DeepSeekRankPostExecResourceObservation>{
      std::move(*decoded)};
}

}  // namespace pih
