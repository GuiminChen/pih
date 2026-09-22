#include "pih/model/deepseek_rank_spawn_resource_collector.h"

#include <algorithm>
#include <utility>

namespace pih {
namespace {

bool stable_usage(const DeepSeekRankSpawnUsageSnapshot& lhs,
                  const DeepSeekRankSpawnUsageSnapshot& rhs) {
  if (lhs.uid_tasks_current != rhs.uid_tasks_current ||
      lhs.controller_open_fds != rhs.controller_open_fds ||
      lhs.cgroup_ancestors.size() != rhs.cgroup_ancestors.size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.cgroup_ancestors.size(); ++index) {
    const auto& left = lhs.cgroup_ancestors[index];
    const auto& right = rhs.cgroup_ancestors[index];
    if (left.scope_root != right.scope_root ||
        left.current_tasks != right.current_tasks ||
        left.maximum_tasks != right.maximum_tasks) {
      return false;
    }
  }
  return true;
}

}  // namespace

Result<StableDeepSeekRankSpawnResourceCollector>
StableDeepSeekRankSpawnResourceCollector::Create(
    DeepSeekRankSpawnAuthorityProbe& probe,
    std::uint32_t maximum_usage_samples) {
  if (maximum_usage_samples < 2 || maximum_usage_samples > 8) {
    return Status::InvalidArgument(
        "DeepSeek rank usage stabilization bound is invalid");
  }
  return StableDeepSeekRankSpawnResourceCollector(
      probe, maximum_usage_samples);
}

Result<DeepSeekRankSpawnResourceObservation>
StableDeepSeekRankSpawnResourceCollector::collect() {
  auto authority = probe_->read_authority();
  if (!authority.ok()) return authority.status();
  auto previous = probe_->sample_usage();
  if (!previous.ok()) return previous.status();
  std::uint64_t maximum_node_file_allocated =
      previous->node_file_allocated;
  for (std::uint32_t sample = 1; sample < maximum_usage_samples_; ++sample) {
    auto current = probe_->sample_usage();
    if (!current.ok()) return current.status();
    maximum_node_file_allocated = std::max(
        maximum_node_file_allocated, current->node_file_allocated);
    if (!stable_usage(*previous, *current)) {
      previous = std::move(current);
      continue;
    }
    auto verified_authority = probe_->read_authority();
    if (!verified_authority.ok()) return verified_authority.status();
    if (*verified_authority != *authority) {
      return Status::FailedPrecondition(
          "DeepSeek rank spawn authority changed during collection");
    }
    return DeepSeekRankSpawnResourceObservation{
        current->uid_tasks_current,
        authority->rlimit_nproc_soft,
        authority->rlimit_nproc_enforced,
        std::move(current->cgroup_ancestors),
        current->controller_open_fds,
        authority->rlimit_nofile_soft,
        authority->rlimit_nofile_hard,
        authority->fs_nr_open,
        maximum_node_file_allocated,
        authority->node_file_maximum,
        authority->vm_max_map_count};
  }
  return Status::Unavailable(
      "DeepSeek rank spawn usage did not stabilize");
}

}  // namespace pih
