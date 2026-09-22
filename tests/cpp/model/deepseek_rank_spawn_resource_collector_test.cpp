#include "pih/model/deepseek_rank_spawn_resource_collector.h"

#include <gtest/gtest.h>

#include <utility>

namespace pih {
namespace {

Sha256Digest digest(std::uint8_t value) {
  Sha256Digest result{};
  result.bytes.fill(static_cast<std::byte>(value));
  return result;
}

DeepSeekRankSpawnAuthoritySnapshot authority() {
  return {100, true, 200, 300, 1000, 2000, 3000};
}

DeepSeekRankSpawnUsageSnapshot usage(
    std::uint64_t uid_tasks = 10,
    std::uint64_t controller_fds = 20,
    std::uint64_t node_files = 30) {
  return {uid_tasks,
          {{digest(1), 5, 100}, {digest(2), 10, 200}},
          controller_fds,
          node_files};
}

class Probe final : public DeepSeekRankSpawnAuthorityProbe {
 public:
  Result<DeepSeekRankSpawnAuthoritySnapshot> read_authority() override {
    ++authority_calls;
    if (!authority_status.ok()) return authority_status;
    const auto index = std::min<std::size_t>(
        authority_calls - 1, authorities.size() - 1);
    return authorities[index];
  }

  Result<DeepSeekRankSpawnUsageSnapshot> sample_usage() override {
    ++usage_calls;
    if (!usage_status.ok()) return usage_status;
    const auto index = std::min<std::size_t>(
        usage_calls - 1, usages.size() - 1);
    return usages[index];
  }

  std::vector<DeepSeekRankSpawnAuthoritySnapshot> authorities{authority()};
  std::vector<DeepSeekRankSpawnUsageSnapshot> usages{usage()};
  Status authority_status = Status::Ok();
  Status usage_status = Status::Ok();
  std::uint32_t authority_calls = 0;
  std::uint32_t usage_calls = 0;
};

TEST(StableDeepSeekRankSpawnResourceCollectorTest,
     PublishesOnlyConsecutiveStableUsageAndAuthority) {
  Probe probe;
  auto collector = StableDeepSeekRankSpawnResourceCollector::Create(probe);
  ASSERT_TRUE(collector.ok());
  auto observed = collector->collect();
  ASSERT_TRUE(observed.ok());
  EXPECT_EQ(probe.usage_calls, 2U);
  EXPECT_EQ(probe.authority_calls, 2U);
  EXPECT_EQ(observed->uid_tasks_current, 10U);
  EXPECT_EQ(observed->controller_open_fds, 20U);
  EXPECT_EQ(observed->node_file_allocated, 30U);
  EXPECT_EQ(observed->rlimit_nproc_soft, 100U);
  EXPECT_EQ(observed->cgroup_ancestors.size(), 2U);
}

TEST(StableDeepSeekRankSpawnResourceCollectorTest,
     StabilizesAfterDriftAndUsesConservativeNodeMaximum) {
  Probe probe;
  probe.usages = {usage(10, 20, 90), usage(11, 20, 30),
                  usage(11, 20, 40)};
  auto collector = StableDeepSeekRankSpawnResourceCollector::Create(probe);
  ASSERT_TRUE(collector.ok());
  auto observed = collector->collect();
  ASSERT_TRUE(observed.ok());
  EXPECT_EQ(probe.usage_calls, 3U);
  EXPECT_EQ(observed->uid_tasks_current, 11U);
  EXPECT_EQ(observed->node_file_allocated, 90U);
}

TEST(StableDeepSeekRankSpawnResourceCollectorTest,
     RejectsUnstableUsageAndAuthorityDrift) {
  Probe unstable;
  unstable.usages = {usage(10), usage(11), usage(12), usage(13)};
  auto collector = StableDeepSeekRankSpawnResourceCollector::Create(unstable);
  ASSERT_TRUE(collector.ok());
  EXPECT_EQ(collector->collect().status().code(), StatusCode::kUnavailable);
  EXPECT_EQ(unstable.usage_calls, 4U);
  EXPECT_EQ(unstable.authority_calls, 1U);

  Probe changed;
  auto changed_authority = authority();
  changed_authority.rlimit_nofile_soft++;
  changed.authorities.push_back(changed_authority);
  auto changed_collector =
      StableDeepSeekRankSpawnResourceCollector::Create(changed);
  ASSERT_TRUE(changed_collector.ok());
  EXPECT_EQ(changed_collector->collect().status().code(),
            StatusCode::kFailedPrecondition);
  EXPECT_EQ(changed.authority_calls, 2U);
}

TEST(StableDeepSeekRankSpawnResourceCollectorTest,
     PropagatesProbeFailureAndBoundsSampling) {
  Probe probe;
  EXPECT_FALSE(
      StableDeepSeekRankSpawnResourceCollector::Create(probe, 1).ok());
  EXPECT_FALSE(
      StableDeepSeekRankSpawnResourceCollector::Create(probe, 9).ok());
  auto collector = StableDeepSeekRankSpawnResourceCollector::Create(probe, 2);
  ASSERT_TRUE(collector.ok());
  probe.usage_status = Status::Internal("usage unavailable");
  EXPECT_EQ(collector->collect().status().code(), StatusCode::kInternal);
  EXPECT_EQ(probe.authority_calls, 1U);
  EXPECT_EQ(probe.usage_calls, 1U);
}

}  // namespace
}  // namespace pih
