#include "pih/model/deepseek_rank_post_exec_resource_collector.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih {
namespace {

Sha256Digest collector_digest(std::uint8_t seed) {
  Sha256Digest value{};
  value.bytes.fill(static_cast<std::byte>(seed));
  return value;
}

DeepSeekRankPostExecResourceIdentity collector_identity() {
  return {7, 8, 1, 100, 400, collector_digest(1), collector_digest(2)};
}

DeepSeekRankPostExecResourceSnapshot collector_snapshot(
    std::uint64_t open_fds = 20) {
  return {100, 8, open_fds, 0, 200, 24, 100, 1000, 210, true};
}

class ResourceProbe final : public DeepSeekRankPostExecResourceProbe {
 public:
  Result<DeepSeekRankPostExecResourceSnapshot> sample() override {
    ++calls;
    if (!failure.ok()) return failure;
    if (next < samples.size()) return samples[next++];
    return samples.back();
  }

  std::vector<DeepSeekRankPostExecResourceSnapshot> samples{
      collector_snapshot()};
  std::size_t next = 0;
  std::uint32_t calls = 0;
  Status failure = Status::Ok();
};

TEST(DeepSeekRankPostExecResourceCollectorTest,
     PublishesOnlyAfterTwoExactConsecutiveSamples) {
  EXPECT_EQ(kDeepSeekRankPostExecResourceCollectorAbi,
            "pih_deepseek_rank_post_exec_resource_collector_v1");
  ResourceProbe probe;
  probe.samples = {collector_snapshot(19), collector_snapshot(20),
                   collector_snapshot(20)};
  auto collector = StableDeepSeekRankPostExecResourceCollector::Create(
      probe, 4).value();
  auto observation = collector.collect(collector_identity());

  ASSERT_TRUE(observation.ok()) << observation.status().message();
  EXPECT_EQ(probe.calls, 3U);
  EXPECT_EQ(observation->process_identity, 100U);
  EXPECT_EQ(observation->open_fd_count, 20U);
  EXPECT_EQ(observation->acknowledged_capacity_plan_instance_root,
            collector_digest(1));
  EXPECT_EQ(observation->acknowledged_os_resource_envelope_root,
            collector_digest(2));
  EXPECT_TRUE(observation->non_dumpable);
}

TEST(DeepSeekRankPostExecResourceCollectorTest,
     RejectsUnstableWindowAndBoundsSampleCount) {
  ResourceProbe probe;
  probe.samples = {collector_snapshot(19), collector_snapshot(20),
                   collector_snapshot(21), collector_snapshot(22)};
  auto collector = StableDeepSeekRankPostExecResourceCollector::Create(
      probe, 4).value();
  EXPECT_EQ(collector.collect(collector_identity()).status().code(),
            StatusCode::kUnavailable);
  EXPECT_EQ(probe.calls, 4U);

  EXPECT_FALSE(StableDeepSeekRankPostExecResourceCollector::Create(
                   probe, 1).ok());
  EXPECT_FALSE(StableDeepSeekRankPostExecResourceCollector::Create(
                   probe, 9).ok());
}

TEST(DeepSeekRankPostExecResourceCollectorTest,
     RejectsIdentityAndSnapshotAuthorityDrift) {
  ResourceProbe probe;
  auto collector = StableDeepSeekRankPostExecResourceCollector::Create(
      probe).value();
  auto identity = collector_identity();
  identity.capacity_plan_instance_root = {};
  EXPECT_FALSE(collector.collect(identity).ok());
  EXPECT_EQ(probe.calls, 0U);

  identity = collector_identity();
  probe.samples[0].process_identity++;
  EXPECT_FALSE(collector.collect(identity).ok());
  EXPECT_EQ(probe.calls, 1U);
  probe.samples[0] = collector_snapshot();
  probe.samples[0].rlimit_nofile_hard = 1001;
  EXPECT_FALSE(collector.collect(identity).ok());
  probe.samples[0] = collector_snapshot();
  probe.samples[0].non_dumpable = false;
  EXPECT_FALSE(collector.collect(identity).ok());
}

TEST(DeepSeekRankPostExecResourceCollectorTest,
     PropagatesProbeFailureWithoutPublishing) {
  ResourceProbe probe;
  probe.failure = Status::Internal("injected resource sample failure");
  auto collector = StableDeepSeekRankPostExecResourceCollector::Create(
      probe).value();
  EXPECT_EQ(collector.collect(collector_identity()).status().code(),
            StatusCode::kInternal);
  EXPECT_EQ(probe.calls, 1U);
}

}  // namespace
}  // namespace pih
