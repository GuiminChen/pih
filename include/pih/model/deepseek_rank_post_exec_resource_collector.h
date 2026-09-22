#pragma once

#include <cstdint>
#include <string_view>

#include "pih/model/deepseek_rank_post_exec_resource_receipt.h"

namespace pih {

inline constexpr std::string_view
    kDeepSeekRankPostExecResourceCollectorAbi =
        "pih_deepseek_rank_post_exec_resource_collector_v1";

struct DeepSeekRankPostExecResourceIdentity final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t rank = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t challenge_identity = 0;
  Sha256Digest capacity_plan_instance_root{};
  Sha256Digest os_resource_envelope_root{};
};

struct DeepSeekRankPostExecResourceSnapshot final {
  std::uint64_t process_identity = 0;
  std::uint64_t task_count = 0;
  std::uint64_t open_fd_count = 0;
  std::uint64_t scm_rights_inflight_fd_count = 0;
  std::uint64_t vma_count = 0;
  std::uint64_t rlimit_nofile_soft = 0;
  std::uint64_t rlimit_nofile_hard = 0;
  std::uint64_t fs_nr_open = 0;
  std::uint64_t vm_max_map_count = 0;
  bool non_dumpable = false;
  friend bool operator==(const DeepSeekRankPostExecResourceSnapshot&,
                         const DeepSeekRankPostExecResourceSnapshot&) =
      default;
};

class DeepSeekRankPostExecResourceProbe {
 public:
  virtual ~DeepSeekRankPostExecResourceProbe() = default;
  virtual Result<DeepSeekRankPostExecResourceSnapshot> sample() = 0;
};

class DeepSeekRankScmRightsInFlightProbe {
 public:
  virtual ~DeepSeekRankScmRightsInFlightProbe() = default;
  virtual Result<std::uint64_t> sample_inflight_fd_count() = 0;
};

class StableDeepSeekRankPostExecResourceCollector final {
 public:
  static Result<StableDeepSeekRankPostExecResourceCollector> Create(
      DeepSeekRankPostExecResourceProbe& probe,
      std::uint32_t maximum_samples = 4);

  Result<DeepSeekRankPostExecResourceObservation> collect(
      const DeepSeekRankPostExecResourceIdentity& identity);

 private:
  StableDeepSeekRankPostExecResourceCollector(
      DeepSeekRankPostExecResourceProbe& probe,
      std::uint32_t maximum_samples) noexcept
      : probe_(&probe), maximum_samples_(maximum_samples) {}

  DeepSeekRankPostExecResourceProbe* probe_ = nullptr;
  std::uint32_t maximum_samples_ = 0;
};

}  // namespace pih
