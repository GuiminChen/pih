#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "pih/model/deepseek_rank_spawn_coordinator.h"

namespace pih {

struct DeepSeekRankSpawnAuthoritySnapshot final {
  std::optional<std::uint64_t> rlimit_nproc_soft;
  bool rlimit_nproc_enforced = true;
  std::uint64_t rlimit_nofile_soft = 0;
  std::uint64_t rlimit_nofile_hard = 0;
  std::uint64_t fs_nr_open = 0;
  std::uint64_t node_file_maximum = 0;
  std::uint64_t vm_max_map_count = 0;
  friend bool operator==(const DeepSeekRankSpawnAuthoritySnapshot&,
                         const DeepSeekRankSpawnAuthoritySnapshot&) = default;
};

struct DeepSeekRankSpawnUsageSnapshot final {
  std::uint64_t uid_tasks_current = 0;
  std::vector<DeepSeekRankSpawnCgroupPidsObservation> cgroup_ancestors;
  std::uint64_t controller_open_fds = 0;
  std::uint64_t node_file_allocated = 0;
};

class DeepSeekRankSpawnAuthorityProbe {
 public:
  virtual ~DeepSeekRankSpawnAuthorityProbe() = default;
  virtual Result<DeepSeekRankSpawnAuthoritySnapshot> read_authority() = 0;
  virtual Result<DeepSeekRankSpawnUsageSnapshot> sample_usage() = 0;
};

class StableDeepSeekRankSpawnResourceCollector final
    : public DeepSeekRankSpawnResourceCollector {
 public:
  static Result<StableDeepSeekRankSpawnResourceCollector> Create(
      DeepSeekRankSpawnAuthorityProbe& probe,
      std::uint32_t maximum_usage_samples = 4);

  Result<DeepSeekRankSpawnResourceObservation> collect() override;

 private:
  StableDeepSeekRankSpawnResourceCollector(
      DeepSeekRankSpawnAuthorityProbe& probe,
      std::uint32_t maximum_usage_samples) noexcept
      : probe_(&probe), maximum_usage_samples_(maximum_usage_samples) {}

  DeepSeekRankSpawnAuthorityProbe* probe_ = nullptr;
  std::uint32_t maximum_usage_samples_ = 0;
};

}  // namespace pih
