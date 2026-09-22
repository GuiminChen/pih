#include "pih/model/deepseek_indexer_projection_coordinator.h"
#include "pih/model/deepseek_fixed_state_layout.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class FixedOps final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class ProjectionOps final : public DeepSeekIndexerProjectionOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("zero"); return Status::Ok();
  }
  Status project(DeepSeekIndexerProjectionLaunch) override {
    calls.push_back("project");
    return fail ? Status::Internal("injected") : Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override {
    calls.push_back("d2h"); return Status::Ok();
  }
  bool fail = false;
  std::vector<std::string> calls;
};

struct Fixture final {
  DeepSeekFixedStateLayout layout =
      DeepSeekFixedStateLayout::Build(std::vector<std::uint32_t>{2}, false)
          .value();
  FixedOps fixed_ops;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x100000, layout.total_bytes()}, {0x200000, layout.total_bytes()}, 8,
      fixed_ops).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          7, 1, 1, banks, ratio4, ratio128).value();
  ProjectionOps operations;
  std::uint32_t host_error = 9;
  DeepSeekIndexerProjectionCoordinator coordinator =
      DeepSeekIndexerProjectionCoordinator::Create(
          operations, &host_error).value();
};

DeepSeekIndexerProjectionLaunch launch() {
  return {1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 8, 9, 17, 2, 1024};
}

TEST(DeepSeekIndexerProjectionCoordinatorTest,
     ClaimsSharedErrorAndPostsProjectionBeforeD2h) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(17).ok());
  ASSERT_TRUE(fixture.coordinator.launch(launch(), fixture.transaction).ok());
  EXPECT_EQ(fixture.host_error, 0U);
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "project", "d2h"}));
}

TEST(DeepSeekIndexerProjectionCoordinatorTest,
     KernelFailurePoisonsCoordinatorAndSkipsD2h) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(17).ok());
  fixture.operations.fail = true;
  EXPECT_FALSE(fixture.coordinator.launch(launch(), fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "project"}));
  fixture.operations.fail = false;
  EXPECT_FALSE(fixture.coordinator.launch(launch(), fixture.transaction).ok());
}

} }  // namespace pih
