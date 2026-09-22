#pragma once
#include "supervisor_deployment.h"
#include "supervisor_operation.h"
#include "supervisor_output.h"

namespace pih::deepseek_v41 {
struct SupervisorCompletion final {
  uint64_t generation{}, prompt_tokens{}, completion_tokens{}, visible_bytes{};
  TokenFinish finish = TokenFinish::kNone;
};
// Dedicated CPU controller process; serialized calls only. All asynchronous
// borrowers are members and remain alive if retirement cannot be reconciled.
// The plugin must retain this owner and refuse unload on !retired().
class SupervisorExecution final {
 public:
  using Cancelled = bool (*)(void*);
  explicit SupervisorExecution(std::unique_ptr<SupervisorRequest> request);
  SupervisorExecution(const SupervisorExecution&) = delete;
  SupervisorExecution& operator=(const SupervisorExecution&) = delete;
  Result<SupervisorCompletion> Run(const SupervisorConfig& config, void* context,
      SupervisorOutput::Writer writer, Cancelled cancelled) noexcept;
  bool retired() const noexcept { return retired_; }
 private:
  Result<SupervisorCompletion> RunImpl(const SupervisorConfig&, void*,
      SupervisorOutput::Writer, Cancelled);
  bool Cleanup(std::chrono::milliseconds timeout) noexcept;
  std::array<WorkerCgroup, 9> groups_;
  std::unique_ptr<WorkerEnvironment> environment_;
  std::unique_ptr<WorkerExecutable> worker_, helper_;
  std::unique_ptr<SupervisorRequest> request_;
  SupervisorOutput output_;
  SupervisorOperation operation_;
  bool retired_ = true;
};
// Fail before launch unless the service owns a dedicated single-threaded
// child-free CPU process with normal SIGCHLD reaping semantics.
Status AdmitSupervisorProcess();
}

