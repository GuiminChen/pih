#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "worker_spawn.h"
#include "worker_bootstrap.h"
#include "nccl_bootstrap_channel.h"
#include <cerrno>
#include <csignal>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <linux/sched.h>
#include <linux/close_range.h>
#include <poll.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <new>
#include <vector>

namespace pih::deepseek_v41 {
namespace {
struct Fds {
  int executable = -1, bootstrap = -1, cgroup = -1, read = -1, write = -1;
  ~Fds() { for (const int fd : {executable, bootstrap, cgroup, read, write}) if (fd >= 0) ::close(fd); }
};
bool Text(const std::string& value) { return value.size() <= 8192 && value.find('\0') == std::string::npos; }
Status SingleThreaded() {
  // clone followed by exec must not inherit library locks from other threads.
  // Fail closed if procfs cannot establish the documented supervisor contract.
  const auto expected = std::to_string(::getpid());
  struct Directory {
    DIR* value;
    ~Directory() { if (value) ::closedir(value); }
  } directory{::opendir("/proc/self/task")};
  if (!directory.value) return Status::FailedPrecondition("Cannot verify single-threaded worker supervisor");
  unsigned count = 0;
  for (;;) {
    errno = 0;
    const auto* entry = ::readdir(directory.value);
    if (!entry) {
      if (errno) return Status::Unavailable("Cannot enumerate supervisor threads");
      break;
    }
    if (!std::strcmp(entry->d_name, ".") || !std::strcmp(entry->d_name, "..")) continue;
    if (++count != 1 || expected != entry->d_name)
      return Status::FailedPrecondition("Worker launch requires a dedicated single-threaded supervisor");
  }
  return count == 1 ? Status::Ok() : Status::FailedPrecondition("Supervisor thread identity absent");
}
[[noreturn]] void ChildError(int pipe, int error) {
  // No allocation, stdio, C++ destruction or mutex acquisition after clone.
  ssize_t n;
  do { n = ::write(pipe, &error, sizeof(error)); } while (n < 0 && errno == EINTR);
  ::_exit(127);
}
}
WorkerSpawn::~WorkerSpawn() {
  if (errors_ >= 0) ::close(errors_);
  if (binding_.pidfd >= 0) ::close(binding_.pidfd);
}
Status WorkerSpawn::Start(const WorkerExecutable& executable, int bootstrap, int cgroup,
    std::span<const std::string> arguments, const WorkerEnvironment& admitted_environment, Clock::time_point deadline) {
  return StartImpl(executable, bootstrap, cgroup, arguments, admitted_environment, deadline, false);
}
Status WorkerSpawn::StartNcclBroker(const WorkerExecutable& executable, int socket, int cgroup,
    const WorkerEnvironment& environment, Clock::time_point startup, Clock::time_point lifetime) {
  if (startup <= Clock::now() || lifetime <= startup)
    return Status::InvalidArgument("NCCL broker deadlines invalid");
  try {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(lifetime.time_since_epoch()).count();
    if (ns <= 0) return Status::InvalidArgument("NCCL broker lifetime is not representable");
    const std::array<std::string, 2> arguments{"pih-v41-nccl-bootstrap", std::to_string(ns)};
    return StartImpl(executable, socket, cgroup, arguments, environment, startup, true);
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("NCCL broker argument allocation failed"); }
}
Status WorkerSpawn::StartImpl(const WorkerExecutable& executable, int bootstrap, int cgroup,
    std::span<const std::string> arguments, const WorkerEnvironment& admitted_environment, Clock::time_point deadline, bool broker) {
  const auto environment = admitted_environment.entries();
  if (state_ != WorkerSpawnState::kEmpty) return Status::FailedPrecondition("Worker spawn is single-use");
  if (bootstrap < 0 || cgroup < 0 || arguments.empty() || arguments.size() > 128 ||
      environment.size() > 128 || Clock::now() >= deadline)
    return Status::InvalidArgument("Worker spawn descriptors, argv or deadline invalid");
  std::size_t bytes = 0;
  for (const auto& value : arguments) { if (!Text(value)) return Status::InvalidArgument("Worker argument invalid"); bytes += value.size() + 1; }
  if (arguments.front().empty()) return Status::InvalidArgument("Worker argv[0] is empty");
  for (std::size_t i = 0; i < environment.size(); ++i) {
    const auto& value = environment[i]; const auto equal = value.find('=');
    if (!Text(value) || equal == 0 || equal == std::string::npos)
      return Status::InvalidArgument("Worker explicit environment entry invalid");
    for (std::size_t j = 0; j < i; ++j)
      if (std::string_view(environment[j]).substr(0, environment[j].find('=')) == std::string_view(value).substr(0, equal))
        return Status::InvalidArgument("Worker environment contains duplicate keys");
    bytes += value.size() + 1;
  }
  if (bytes > 131072) return Status::ResourceExhausted("Worker argv/environment exceeds bound");
  struct sigaction child_policy{};
  if (::sigaction(SIGCHLD, nullptr, &child_policy) || child_policy.sa_handler == SIG_IGN ||
      (child_policy.sa_flags & SA_NOCLDWAIT))
    return Status::FailedPrecondition("Worker supervisor must retain child exit statuses");
  try {
    Fds fds;
    const auto admitted_executable = executable.Descriptor();
    if (!admitted_executable.ok()) return admitted_executable.status();
    fds.executable = ::fcntl(*admitted_executable, F_DUPFD_CLOEXEC, 64);
    fds.bootstrap = ::fcntl(bootstrap, F_DUPFD_CLOEXEC, 64);
    fds.cgroup = ::fcntl(cgroup, F_DUPFD_CLOEXEC, 64);
    if (fds.executable < 0 || fds.bootstrap < 0 || fds.cgroup < 0)
      return Status::Unavailable("Cannot retain worker launch descriptors");
    struct stat binary{}, group{};
    unsigned char magic[4]{};
    if (::fstat(fds.executable, &binary) || !S_ISREG(binary.st_mode) || !(binary.st_mode & 0111) ||
        ::pread(fds.executable, magic, sizeof(magic), 0) != sizeof(magic) ||
        std::memcmp(magic, "\177ELF", 4) || ::fstat(fds.cgroup, &group) || !S_ISDIR(group.st_mode))
      return Status::InvalidArgument("Worker requires an admitted ELF executable and cgroup directory");
    if (broker) {
      const auto valid = NcclBootstrapChannel::ValidateSocket(fds.bootstrap);
      if (!valid.ok()) return valid;
    } else {
    const int seals = ::fcntl(fds.bootstrap, F_GET_SEALS);
    const int required = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    if (seals < 0 || (seals & required) != required)
      return Status::FailedPrecondition("Worker bootstrap descriptor is not immutably sealed");
    const auto admitted = SealedWorkerBootstrap::Read(fds.bootstrap);
    if (!admitted.ok()) return admitted.status();
    if (admitted->supervisor_pid != static_cast<std::uint32_t>(::getpid()) ||
        admitted->supervisor_uid != static_cast<std::uint32_t>(::geteuid()))
      return Status::FailedPrecondition("Sealed worker bootstrap names another supervisor");
    }
    int pipes[2]{-1, -1};
    if (::pipe2(pipes, O_CLOEXEC | O_NONBLOCK)) return Status::Unavailable("Cannot create worker exec-status pipe");
    fds.read = pipes[0];
    fds.write = ::fcntl(pipes[1], F_DUPFD_CLOEXEC, 64); ::close(pipes[1]);
    if (fds.write < 0) return Status::Unavailable("Cannot retain worker exec-status writer");
    std::vector<char*> argv, env;
    argv.reserve(arguments.size() + 1); env.reserve(environment.size() + 1);
    for (const auto& value : arguments) argv.push_back(const_cast<char*>(value.c_str()));
    for (const auto& value : environment) env.push_back(const_cast<char*>(value.c_str()));
    argv.push_back(nullptr); env.push_back(nullptr);
    auto* const argv_data = argv.data(); auto* const env_data = env.data();
    const auto parent = ::getpid();
    sigset_t empty_mask; ::sigemptyset(&empty_mask);
    struct sigaction defaults{}; defaults.sa_handler = SIG_DFL; ::sigemptyset(&defaults.sa_mask);
    int pidfd = -1;
    clone_args clone{};
    clone.flags = CLONE_PIDFD | CLONE_INTO_CGROUP;
    clone.pidfd = reinterpret_cast<std::uintptr_t>(&pidfd);
    clone.exit_signal = SIGCHLD;
    clone.cgroup = static_cast<std::uint64_t>(fds.cgroup);
    const auto single_threaded = SingleThreaded();
    if (!single_threaded.ok()) return single_threaded;
    if (Clock::now() >= deadline) return Status::DeadlineExceeded("Worker launch preparation expired");
    state_ = WorkerSpawnState::kFailed;
    const auto pid = ::syscall(SYS_clone3, &clone, sizeof(clone));
    if (pid < 0) return Status::Unavailable("clone3 worker launch failed; no fork or out-of-cgroup fallback");
    if (pid == 0) {
      if (::prctl(PR_SET_PDEATHSIG, SIGKILL) || ::getppid() != parent) ChildError(fds.write, errno ? errno : ECHILD);
      if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) ChildError(fds.write, errno);
      if (::sigprocmask(SIG_SETMASK, &empty_mask, nullptr)) ChildError(fds.write, errno);
      for (const int signal : {SIGTERM, SIGINT, SIGHUP, SIGPIPE, SIGCHLD})
        if (::sigaction(signal, &defaults, nullptr)) ChildError(fds.write, errno);
      if (::syscall(SYS_close_range, 4U, ~0U, CLOSE_RANGE_CLOEXEC)) ChildError(fds.write, errno);
      if (::dup3(fds.bootstrap, 3, 0) < 0) ChildError(fds.write, errno);
      // Child diagnostics must not corrupt the supervisor's token byte stream.
      if (::dup2(STDERR_FILENO, STDOUT_FILENO) < 0) ChildError(fds.write, errno);
      ::syscall(SYS_execveat, fds.executable, "", argv_data, env_data, AT_EMPTY_PATH);
      ChildError(fds.write, errno);
    }
    // No allocations after clone in parent before recording all process custody.
    binding_ = {pidfd, static_cast<std::int32_t>(pid)};
    errors_ = fds.read; fds.read = -1;
    deadline_ = deadline; state_ = WorkerSpawnState::kCheckingExec;
    return Status::Ok();
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Worker launch metadata allocation failed"); }
}
Result<bool> WorkerSpawn::PollExec() {
  if (state_ == WorkerSpawnState::kAwaitingChannels) return true;
  if (state_ != WorkerSpawnState::kCheckingExec) return Status::FailedPrecondition("Worker has no pending exec observation");
  if (Clock::now() >= deadline_) { state_ = WorkerSpawnState::kFailed; return Status::DeadlineExceeded("Worker exec observation expired"); }
  const auto n = ::read(errors_, error_bytes_.data() + received_, error_bytes_.size() - received_);
  if (n < 0 && (errno == EAGAIN || errno == EINTR)) return false;
  if (n < 0) { state_ = WorkerSpawnState::kFailed; return Status::Unavailable("Worker exec pipe failed"); }
  if (n > 0) {
    received_ += static_cast<std::size_t>(n);
    if (received_ != error_bytes_.size()) return false;
    std::memcpy(&child_errno_, error_bytes_.data(), sizeof(child_errno_));
    state_ = WorkerSpawnState::kFailed; return Status::FailedPrecondition("Worker child setup or exec failed; retained errno and pidfd");
  }
  state_ = WorkerSpawnState::kFailed;
  if (received_) return Status::FailedPrecondition("Worker exec error frame truncated");
  pollfd process{binding_.pidfd, POLLIN, 0};
  if (::poll(&process, 1, 0) != 0) return Status::FailedPrecondition("Worker exited before post-exec admission");
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Worker exec observation completed late");
  ::close(errors_); errors_ = -1;
  state_ = WorkerSpawnState::kAwaitingChannels;
  return true; // EOF is not readiness: require channel authentication + ready notice.
}
Status WorkerSpawn::Kill() {
  if (binding_.pidfd < 0 || state_ == WorkerSpawnState::kReaped)
    return Status::FailedPrecondition("Worker has no retained process to kill");
  if (state_ == WorkerSpawnState::kKilled) return Status::Ok();
  if (::syscall(SYS_pidfd_send_signal, binding_.pidfd, SIGKILL, nullptr, 0) && errno != ESRCH)
    return Status::Unavailable("Worker pidfd kill failed");
  state_ = WorkerSpawnState::kKilled; return Status::Ok();
}
Result<bool> WorkerSpawn::PollKilled(Clock::time_point deadline) {
  if (state_ == WorkerSpawnState::kReaped) return true;
  if (state_ != WorkerSpawnState::kKilled) return Status::FailedPrecondition("Worker has not entered startup fault retirement");
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Worker startup fault reap expired");
  siginfo_t info{};
  if (::waitid(P_PIDFD, static_cast<id_t>(binding_.pidfd), &info, WEXITED | WNOHANG)) {
    if (errno == EINTR) return false;
    return Status::FailedPrecondition("Worker startup reap requires exclusive parent ownership");
  }
  if (!info.si_pid) return false;
  if (info.si_pid != binding_.pid || (info.si_code != CLD_EXITED && info.si_code != CLD_KILLED && info.si_code != CLD_DUMPED))
    return Status::FailedPrecondition("Worker startup reap identity invalid");
  state_ = WorkerSpawnState::kReaped; return true;
}
}  // namespace pih::deepseek_v41
