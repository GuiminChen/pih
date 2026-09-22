#include "pih/model/deepseek_rank_worker_materialization_lifecycle.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class Startup final : public DeepSeekRankWorkerStartupProtocol {
 public:
  Status run_resource_barrier() override {
    ++resource_calls;
    return resource_status;
  }
  Status run_artifact_transfer() override {
    ++artifact_calls;
    return artifact_status;
  }
  Status run_materialization_completion() override {
    ++completion_calls;
    return completion_status;
  }
  [[nodiscard]] bool materialization_completion_sent() const noexcept override {
    return completion_sent;
  }

  Status resource_status = Status::Ok();
  Status artifact_status = Status::Ok();
  Status completion_status = Status::Ok();
  bool completion_sent = true;
  std::uint32_t resource_calls = 0;
  std::uint32_t artifact_calls = 0;
  std::uint32_t completion_calls = 0;
};

class Application final : public DeepSeekRankWorkerMaterializationApplication {
 public:
  Status materialize() override {
    ++materialize_calls;
    return materialize_status;
  }
  Status begin_completion() override {
    ++begin_completion_calls;
    return begin_completion_status;
  }

  Status materialize_status = Status::Ok();
  Status begin_completion_status = Status::Ok();
  std::uint32_t materialize_calls = 0;
  std::uint32_t begin_completion_calls = 0;
};

TEST(DeepSeekRankWorkerMaterializationLifecycleTest,
     AdvancesOnlyAfterEachWorkerOwnedPredecessor) {
  Startup startup;
  Application application;
  auto lifecycle =
      DeepSeekRankWorkerMaterializationLifecycle::Create(startup, application)
          .value();

  EXPECT_EQ(kDeepSeekRankWorkerMaterializationLifecycleAbi,
            "pih_deepseek_rank_worker_materialization_lifecycle_v1");
  EXPECT_TRUE(lifecycle.advance().ok());
  EXPECT_EQ(lifecycle.state(),
            DeepSeekRankWorkerMaterializationState::kAwaitingArtifactTransfer);
  EXPECT_EQ(application.materialize_calls, 0U);
  EXPECT_TRUE(lifecycle.advance().ok());
  EXPECT_EQ(lifecycle.state(),
            DeepSeekRankWorkerMaterializationState::kMaterializing);
  EXPECT_EQ(startup.artifact_calls, 1U);
  EXPECT_TRUE(lifecycle.advance().ok());
  EXPECT_EQ(lifecycle.state(),
            DeepSeekRankWorkerMaterializationState::kSendingCompletion);
  EXPECT_EQ(application.materialize_calls, 1U);
  EXPECT_TRUE(lifecycle.advance().ok());
  EXPECT_TRUE(lifecycle.ready());
  EXPECT_EQ(application.begin_completion_calls, 1U);
  EXPECT_EQ(startup.completion_calls, 1U);
  EXPECT_TRUE(lifecycle.advance().ok());
  EXPECT_EQ(application.materialize_calls, 1U);
  EXPECT_EQ(application.begin_completion_calls, 1U);
}

TEST(DeepSeekRankWorkerMaterializationLifecycleTest,
     BackpressureDoesNotSkipOrRepeatMaterialization) {
  Startup startup;
  startup.artifact_status = Status::Unavailable("pending artifact transfer");
  Application application;
  auto lifecycle =
      DeepSeekRankWorkerMaterializationLifecycle::Create(startup, application)
          .value();

  ASSERT_TRUE(lifecycle.advance().ok());
  EXPECT_EQ(lifecycle.advance().code(), StatusCode::kUnavailable);
  EXPECT_EQ(lifecycle.state(),
            DeepSeekRankWorkerMaterializationState::kAwaitingArtifactTransfer);
  EXPECT_EQ(application.materialize_calls, 0U);
  startup.artifact_status = Status::Ok();
  ASSERT_TRUE(lifecycle.advance().ok());
  ASSERT_TRUE(lifecycle.advance().ok());
  EXPECT_EQ(application.materialize_calls, 1U);
}

TEST(DeepSeekRankWorkerMaterializationLifecycleTest,
     RejectsCompletionBeforeTheWireSendIsObserved) {
  Startup startup;
  startup.completion_sent = false;
  Application application;
  auto lifecycle =
      DeepSeekRankWorkerMaterializationLifecycle::Create(startup, application)
          .value();

  ASSERT_TRUE(lifecycle.advance().ok());
  ASSERT_TRUE(lifecycle.advance().ok());
  ASSERT_TRUE(lifecycle.advance().ok());
  EXPECT_EQ(lifecycle.advance().code(), StatusCode::kInternal);
  EXPECT_TRUE(lifecycle.failed());
  EXPECT_FALSE(lifecycle.ready());
}

TEST(DeepSeekRankWorkerMaterializationLifecycleTest,
     CompletionBackpressureRetriesOnlyTheExistingSender) {
  Startup startup;
  startup.completion_status = Status::Unavailable("pending completion send");
  Application application;
  auto lifecycle =
      DeepSeekRankWorkerMaterializationLifecycle::Create(startup, application)
          .value();

  ASSERT_TRUE(lifecycle.advance().ok());
  ASSERT_TRUE(lifecycle.advance().ok());
  ASSERT_TRUE(lifecycle.advance().ok());
  EXPECT_EQ(lifecycle.advance().code(), StatusCode::kUnavailable);
  EXPECT_EQ(application.begin_completion_calls, 1U);
  EXPECT_EQ(startup.completion_calls, 1U);

  startup.completion_status = Status::Ok();
  ASSERT_TRUE(lifecycle.advance().ok());
  EXPECT_TRUE(lifecycle.ready());
  EXPECT_EQ(application.begin_completion_calls, 1U);
  EXPECT_EQ(startup.completion_calls, 2U);
}

TEST(DeepSeekRankWorkerMaterializationLifecycleTest,
     MaterializationFailureIsTerminalAndDoesNotPublishCompletion) {
  Startup startup;
  Application application;
  application.materialize_status = Status::Internal("injected CUDA failure");
  auto lifecycle =
      DeepSeekRankWorkerMaterializationLifecycle::Create(startup, application)
          .value();

  ASSERT_TRUE(lifecycle.advance().ok());
  ASSERT_TRUE(lifecycle.advance().ok());
  EXPECT_EQ(lifecycle.advance().code(), StatusCode::kInternal);
  EXPECT_TRUE(lifecycle.failed());
  EXPECT_EQ(application.begin_completion_calls, 0U);
  EXPECT_EQ(startup.completion_calls, 0U);
  EXPECT_EQ(lifecycle.advance().code(), StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace pih
