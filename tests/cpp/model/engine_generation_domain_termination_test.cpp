#include "pih/model/engine_generation_domain_termination.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class Driver final : public EngineGenerationDomainDriver {
 public:
  Status force_kill_domain() override {
    ++kills;
    return kill_status;
  }
  Result<bool> domain_empty() override {
    ++polls;
    if (!poll_status.ok()) return poll_status;
    return empty;
  }
  int kills = 0;
  int polls = 0;
  bool empty = false;
  Status kill_status = Status::Ok();
  Status poll_status = Status::Ok();
};

TEST(EngineGenerationDomainTerminationTest, KillsExactlyOnceAndPollsEmpty) {
  Driver driver;
  auto value = EngineGenerationDomainTermination::Create(driver).value();
  ASSERT_TRUE(value.force_kill().ok());
  EXPECT_TRUE(value.force_kill_issued());
  EXPECT_FALSE(value.force_kill().ok());
  EXPECT_EQ(driver.kills, 1);
  ASSERT_TRUE(value.poll_empty().ok());
  EXPECT_FALSE(*value.poll_empty());
  driver.empty = true;
  ASSERT_TRUE(value.poll_empty().ok());
  EXPECT_TRUE(*value.poll_empty());
}

TEST(EngineGenerationDomainTerminationTest, FailedKillDoesNotCommitIssued) {
  Driver driver;
  driver.kill_status = Status::Unavailable("cgroup.kill unavailable");
  auto value = EngineGenerationDomainTermination::Create(driver).value();
  EXPECT_FALSE(value.force_kill().ok());
  EXPECT_FALSE(value.force_kill_issued());
  EXPECT_EQ(driver.kills, 1);
  driver.kill_status = Status::Ok();
  EXPECT_TRUE(value.force_kill().ok());
  EXPECT_EQ(driver.kills, 2);
}

TEST(EngineGenerationDomainTerminationTest, PreservesUnknownVisibility) {
  Driver driver;
  driver.poll_status = Status::Unavailable("cgroup.events unavailable");
  auto value = EngineGenerationDomainTermination::Create(driver).value();
  auto empty = value.poll_empty();
  ASSERT_FALSE(empty.ok());
  EXPECT_EQ(empty.status().code(), StatusCode::kUnavailable);
  EXPECT_EQ(driver.polls, 1);
}

}  // namespace
}  // namespace pih
