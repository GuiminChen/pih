#include "pih/model/deepseek_nccl_communicator.h"

#include <gtest/gtest.h>

#include <deque>

namespace pih {
namespace {

DeepSeekNcclCommunicatorManifest comm_manifest() {
  return {.engine_epoch = 3,
          .communicator_generation = 4,
          .bootstrap_lease_id = 5,
          .bootstrap_commitment_id = 6,
          .edge_id = 1,
          .local_global_rank = 1,
          .peer_global_rank = 2,
          .communicator_local_rank = 0,
          .device_identity = 7,
          .context_identity = 8,
          .config_identity = 9};
}

class FakeCommunicatorDriver final : public DeepSeekNcclCommunicatorDriver {
 public:
  Result<DeepSeekNcclAsyncStatus> init_rank_config(
      std::uint64_t, std::uint32_t nranks, std::uint32_t rank) override {
    ++init_calls; observed_nranks = nranks; observed_rank = rank; return init_result;
  }
  Result<DeepSeekNcclAsyncStatus> async_status() override {
    if (async_results.empty()) return DeepSeekNcclAsyncStatus::kSuccess;
    auto result = async_results.front(); async_results.pop_front(); return result;
  }
  Result<std::uint32_t> communicator_count() override { return count; }
  Result<std::uint32_t> communicator_user_rank() override { return user_rank; }
  Result<std::uint64_t> device_identity() override { return device; }
  Result<std::uintptr_t> context_identity() override { return context; }
  Result<DeepSeekNcclAsyncStatus> finalize() override { ++finalize_calls; return finalize_result; }
  Status destroy() override { ++destroy_calls; return destroy_status; }
  Status abort() override { ++abort_calls; return Status::Ok(); }

  DeepSeekNcclAsyncStatus init_result = DeepSeekNcclAsyncStatus::kSuccess;
  DeepSeekNcclAsyncStatus finalize_result = DeepSeekNcclAsyncStatus::kSuccess;
  std::deque<DeepSeekNcclAsyncStatus> async_results;
  std::uint32_t count = 2;
  std::uint32_t user_rank = 0;
  std::uint64_t device = 7;
  std::uintptr_t context = 8;
  Status destroy_status = Status::Ok();
  int init_calls = 0;
  int finalize_calls = 0;
  int destroy_calls = 0;
  int abort_calls = 0;
  std::uint32_t observed_nranks = 0;
  std::uint32_t observed_rank = 0;
};

TEST(DeepSeekNcclCommunicatorTest, ValidatesPinnedReleaseConfig) {
  auto sm89 = DeepSeekNcclReleaseConfig::Create(DeepSeekGpuArchitecture::kSm89,
                                                 23102, 128);
  ASSERT_TRUE(sm89.ok());
  EXPECT_EQ(sm89->blocking, 0);
  EXPECT_EQ(sm89->min_ctas, 1);
  EXPECT_EQ(sm89->max_ctas, 32);
  EXPECT_EQ(sm89->cga_cluster_size, 0);
  EXPECT_EQ(sm89->net_name, "Socket");
  EXPECT_EQ(sm89->traffic_class, std::numeric_limits<std::int32_t>::min());
  EXPECT_EQ(sm89->graph_usage_mode, 0);
  EXPECT_EQ(sm89->graph_stream_ordering, 0);
  EXPECT_EQ(sm89->launch_order_implicit, 0);
  EXPECT_EQ(sm89->max_p2p_peers, 1);
  EXPECT_EQ(sm89->num_rma_contexts, 1);
  EXPECT_EQ(sm89->num_rma_signals, 1);
  EXPECT_EQ(sm89->rma_eager_init, 0);
  EXPECT_EQ(sm89->host_cft_disabled, 1);
  EXPECT_FALSE(sm89->communicator_name_present);
  auto sm90 = DeepSeekNcclReleaseConfig::Create(DeepSeekGpuArchitecture::kSm90,
                                                 23102, 128);
  ASSERT_TRUE(sm90.ok());
  EXPECT_EQ(sm90->cga_cluster_size, 4);
  EXPECT_FALSE(DeepSeekNcclReleaseConfig::Create(
      static_cast<DeepSeekGpuArchitecture>(255), 23102, 128).ok());
  EXPECT_FALSE(DeepSeekNcclReleaseConfig::Create(
      DeepSeekGpuArchitecture::kSm90, 23007, 128).ok());
}

TEST(DeepSeekNcclCommunicatorTest, InitializesWarmsSealsAndCleanlyDestroys) {
  auto endpoint = DeepSeekNcclCommunicator::Create(comm_manifest());
  ASSERT_TRUE(endpoint.ok());
  FakeCommunicatorDriver driver;
  ASSERT_TRUE(endpoint->accept_bootstrap(true).ok());
  ASSERT_TRUE(endpoint->begin_init(driver).ok());
  EXPECT_EQ(driver.observed_nranks, 2U);
  EXPECT_EQ(driver.observed_rank, 0U);
  ASSERT_TRUE(endpoint->reconcile(driver).ok());
  ASSERT_TRUE(endpoint->mark_warmed(true, true).ok());
  ASSERT_TRUE(endpoint->seal().ok());
  EXPECT_EQ(endpoint->state(), DeepSeekNcclCommunicatorState::kSealed);
  ASSERT_TRUE(endpoint->begin_finalize(driver).ok());
  ASSERT_TRUE(endpoint->destroy(driver).ok());
  EXPECT_EQ(endpoint->state(), DeepSeekNcclCommunicatorState::kDestroyed);
  EXPECT_EQ(driver.finalize_calls, 1);
  EXPECT_EQ(driver.destroy_calls, 1);
  EXPECT_EQ(driver.abort_calls, 0);
}

TEST(DeepSeekNcclCommunicatorTest, PollsNonblockingInitAndFinalize) {
  auto endpoint = DeepSeekNcclCommunicator::Create(comm_manifest());
  ASSERT_TRUE(endpoint.ok());
  FakeCommunicatorDriver driver;
  driver.init_result = DeepSeekNcclAsyncStatus::kInProgress;
  ASSERT_TRUE(endpoint->accept_bootstrap(true).ok());
  ASSERT_TRUE(endpoint->begin_init(driver).ok());
  EXPECT_EQ(endpoint->state(), DeepSeekNcclCommunicatorState::kInitInProgress);
  driver.async_results = {DeepSeekNcclAsyncStatus::kInProgress,
                          DeepSeekNcclAsyncStatus::kSuccess};
  EXPECT_EQ(endpoint->poll_init(driver).code(), StatusCode::kUnavailable);
  ASSERT_TRUE(endpoint->poll_init(driver).ok());
  ASSERT_TRUE(endpoint->reconcile(driver).ok());
  ASSERT_TRUE(endpoint->mark_warmed(true, true).ok());
  ASSERT_TRUE(endpoint->seal().ok());
  driver.finalize_result = DeepSeekNcclAsyncStatus::kInProgress;
  ASSERT_TRUE(endpoint->begin_finalize(driver).ok());
  driver.async_results = {DeepSeekNcclAsyncStatus::kSuccess};
  ASSERT_TRUE(endpoint->poll_finalize(driver).ok());
  ASSERT_TRUE(endpoint->destroy(driver).ok());
}

TEST(DeepSeekNcclCommunicatorTest, ReconciliationFailureRequiresSingleAbort) {
  auto endpoint = DeepSeekNcclCommunicator::Create(comm_manifest());
  ASSERT_TRUE(endpoint.ok());
  FakeCommunicatorDriver driver;
  driver.user_rank = 1;
  ASSERT_TRUE(endpoint->accept_bootstrap(true).ok());
  ASSERT_TRUE(endpoint->begin_init(driver).ok());
  EXPECT_FALSE(endpoint->reconcile(driver).ok());
  EXPECT_EQ(endpoint->state(), DeepSeekNcclCommunicatorState::kFailed);
  ASSERT_TRUE(endpoint->abort(driver).ok());
  ASSERT_TRUE(endpoint->abort(driver).ok());
  EXPECT_EQ(driver.abort_calls, 1);
}

TEST(DeepSeekNcclCommunicatorTest, RejectsWrongEndpointAcknowledgement) {
  auto endpoint = DeepSeekNcclCommunicator::Create(comm_manifest());
  ASSERT_TRUE(endpoint.ok());
  EXPECT_FALSE(endpoint->accept_bootstrap(false).ok());
  EXPECT_EQ(endpoint->state(), DeepSeekNcclCommunicatorState::kFailed);
  FakeCommunicatorDriver driver;
  EXPECT_TRUE(endpoint->abort(driver).ok());
  EXPECT_EQ(driver.abort_calls, 0);
  EXPECT_EQ(endpoint->state(), DeepSeekNcclCommunicatorState::kAborted);
}

}  // namespace
}  // namespace pih
