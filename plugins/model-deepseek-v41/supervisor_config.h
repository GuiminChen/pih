#pragma once
#include "supervisor_request.h"
#include "worker_cgroup.h"

namespace pih::deepseek_v41 {
struct SupervisorExecutableConfig final {
  std::string path;
  Sha256Digest sha256;
  std::uint64_t byte_budget = 0;
};
struct SupervisorConfig final {
  SupervisorExecutableConfig worker, helper;
  std::vector<std::string> library_directories;
  std::string delegated_cgroup, tokenizer_directory, rendered_prompt;
  WorkerCgroupLimits rank_limits, helper_limits;
  SamplingIdentity identity;
  SamplingParameters sampling;
  TokenStopConfig stopping;
  SupervisorRequestBudgets budgets;
  std::uint32_t maximum_positions = 0, startup_ms = 0, sequence_ms = 0, retirement_ms = 0, grace_ms = 0;
  std::uint64_t first_plan = 0;
  // Ordered rank-local artifacts/device/budgets plus common model fields.
  // Runtime sets PID/UID, absolute deadlines, fresh endpoint names and request
  // sequence fields. NCCL IDs remain zero until SupervisorOperation binds them.
  std::vector<WorkerBootstrap> placements;
  static Result<SupervisorConfig> Load(const std::filesystem::path& path, const Sha256Digest& trusted_digest);
  static Result<SupervisorConfig> Parse(std::string_view json, const Sha256Digest& trusted_digest);
};
}  // namespace pih::deepseek_v41
