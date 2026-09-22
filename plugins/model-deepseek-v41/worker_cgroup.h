#pragma once
#include "pih/core/result.h"
#include <cstdint>
#include <string>

namespace pih::deepseek_v41 {
struct WorkerCgroupLimits final {
  std::uint64_t memory_bytes = 0, pids = 0, cpu_quota_us = 0, cpu_period_us = 100000;
};
// Owns only a newly created direct child of an explicitly delegated cgroup-v2
// directory. Never adopts/reconfigures an existing group or deletes in destructor.
class WorkerCgroup final {
 public:
  WorkerCgroup() = default;
  WorkerCgroup(const WorkerCgroup&) = delete;
  WorkerCgroup& operator=(const WorkerCgroup&) = delete;
  ~WorkerCgroup();
  Status Create(int delegated_parent_fd, const std::string& unique_leaf, WorkerCgroupLimits limits);
  Result<int> ReadyDescriptor() const;
  Result<bool> Empty() const;
  Status Kill();
  // Explicitly remove only the same owned, empty direct child. rmdir refuses
  // nested groups. Reaping zombies remains separate from populated=0.
  Status Remove();
  bool created() const noexcept { return created_; }
  bool removed() const noexcept { return removed_; }
 private:
  Status SameMember() const;
  int parent_ = -1, directory_ = -1;
  std::string leaf_;
  bool created_ = false, ready_ = false, killed_ = false, removed_ = false;
};
}  // namespace pih::deepseek_v41
