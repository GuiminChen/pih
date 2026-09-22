#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "pih/platform/linux/linux_engine_pidfd_reap_operations.h"

#include <algorithm>
#include <cerrno>
#include <fstream>
#include <limits>
#include <set>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

namespace pih {
namespace {

Status failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

Result<std::uint64_t> pidfd_target(int fd) {
  std::ifstream input("/proc/self/fdinfo/" + std::to_string(fd));
  if (!input) return Status::Unavailable("cannot open pidfd fdinfo");
  std::string label;
  std::uint64_t value = 0;
  while (input >> label >> value) {
    if (label == "Pid:") return value;
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return Status::FailedPrecondition("pidfd has no process identity");
}

}  // namespace

Result<LinuxEnginePidfdReapOperations>
LinuxEnginePidfdReapOperations::Create(
    std::span<const LinuxEnginePidfdBinding> bindings) {
  if (bindings.empty() || bindings.size() > 5)
    return Status::InvalidArgument("Linux engine pidfd bindings are invalid");
  std::set<std::uint64_t> identities;
  std::set<std::int32_t> descriptors;
  for (const auto& binding : bindings) {
    if (binding.pidfd_identity == 0 || binding.pidfd < 0 ||
        !identities.insert(binding.pidfd_identity).second ||
        !descriptors.insert(binding.pidfd).second)
      return Status::InvalidArgument(
          "Linux engine pidfd binding is invalid");
    auto target = pidfd_target(binding.pidfd);
    if (!target.ok() || *target == 0) return target.ok()
        ? Status::FailedPrecondition("pidfd target identity is zero")
        : target.status();
  }
  return LinuxEnginePidfdReapOperations(
      std::vector<LinuxEnginePidfdBinding>(bindings.begin(), bindings.end()));
}

Result<std::optional<EnginePidfdKernelReapObservation>>
LinuxEnginePidfdReapOperations::poll_and_reap(
    const EnginePidfdReapTarget& target) {
  const auto found = std::find_if(
      bindings_.begin(), bindings_.end(), [&](const auto& binding) {
        return binding.pidfd_identity == target.process.pidfd_identity;
      });
  if (found == bindings_.end())
    return Status::FailedPrecondition("pidfd target is not bound to producer");
  auto actual_target = pidfd_target(found->pidfd);
  if (!actual_target.ok()) return actual_target.status();
  if (*actual_target != target.process.process_identity)
    return Status::FailedPrecondition("pidfd process identity drifted");
  siginfo_t info{};
  int result = 0;
  do {
    result = ::waitid(P_PIDFD, static_cast<id_t>(found->pidfd), &info,
                      WEXITED | WNOHANG);
  } while (result != 0 && errno == EINTR);
  if (result != 0) {
    if (errno == ECHILD)
      return Status::FailedPrecondition(
          "pidfd producer is not the parent or designated subreaper");
    return failure("waitid P_PIDFD engine process");
  }
  if (info.si_pid == 0)
    return std::optional<EnginePidfdKernelReapObservation>{};
  if (static_cast<std::uint64_t>(info.si_pid) !=
      target.process.process_identity)
    return Status::FailedPrecondition("waitid process identity drifted");
  if (info.si_code != CLD_EXITED && info.si_code != CLD_KILLED &&
      info.si_code != CLD_DUMPED)
    return Status::FailedPrecondition("waitid exit classification drifted");
  return std::optional<EnginePidfdKernelReapObservation>{{
      target.process.process_identity, target.process.pidfd_identity,
      true, true}};
}

}  // namespace pih
