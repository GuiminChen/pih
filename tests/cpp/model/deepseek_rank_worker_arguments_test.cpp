#include "pih/model/deepseek_rank_worker_arguments.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

std::vector<std::string_view> valid() {
  return {"--model=/weights", "--pih-rank=1",
          "--pih-world-size=2", "--pih-engine-epoch=7",
          "--pih-worker-generation=8",
          "--pih-device-identity=10",
          "--pih-startup-device-ordinal=5",
          "--pih-process-manifest=11",
          "--pih-startup-deadline-ns=200",
          "--pih-controller-pid=90",
          "--pih-controller-pidfd=12",
          "--pih-control-fd=13",
          "--pih-dspark-enabled=1",
          "--pih-metadata-reassembly-bytes=134217728"};
}

TEST(DeepSeekRankWorkerArgumentsTest, ParsesExactReservedIdentitySet) {
  auto parsed = DeepSeekRankWorkerArguments::Parse(valid());
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  EXPECT_EQ(parsed->manifest.rank, 1U);
  EXPECT_EQ(parsed->manifest.world_size, 2U);
  EXPECT_EQ(parsed->manifest.startup_device_ordinal, 5);
  EXPECT_EQ(parsed->manifest.startup_deadline_ns, 200U);
  EXPECT_EQ(parsed->controller_process_identity, 90U);
  EXPECT_EQ(parsed->controller_pidfd, 12);
  EXPECT_TRUE(parsed->expected_dspark_enabled);
  EXPECT_EQ(parsed->maximum_metadata_reassembly_bytes, 134217728U);
  ASSERT_EQ(parsed->application_arguments.size(), 1U);
  EXPECT_EQ(parsed->application_arguments[0], "--model=/weights");
}

TEST(DeepSeekRankWorkerArgumentsTest, RejectsMissingDuplicateAndUnknownReserved) {
  {
    auto values = valid(); values.pop_back();
    EXPECT_FALSE(DeepSeekRankWorkerArguments::Parse(values).ok());
  }
  {
    auto values = valid(); values.push_back("--pih-rank=1");
    EXPECT_FALSE(DeepSeekRankWorkerArguments::Parse(values).ok());
  }
  {
    auto values = valid(); values.push_back("--pih-secret=1");
    EXPECT_FALSE(DeepSeekRankWorkerArguments::Parse(values).ok());
  }
}

TEST(DeepSeekRankWorkerArgumentsTest, RejectsNoncanonicalOrOutOfRangeNumbers) {
  for (const std::string_view replacement : {
           "--pih-rank=-1", "--pih-rank=01",
           "--pih-rank=2", "--pih-rank=4294967296"}) {
    auto values = valid(); values[1] = replacement;
    EXPECT_FALSE(DeepSeekRankWorkerArguments::Parse(values).ok());
  }
  auto values = valid(); values[2] = "--pih-world-size=5";
  EXPECT_FALSE(DeepSeekRankWorkerArguments::Parse(values).ok());
  values = valid(); values[6] = "--pih-startup-device-ordinal=2147483648";
  EXPECT_FALSE(DeepSeekRankWorkerArguments::Parse(values).ok());
  values = valid(); values[12] = "--pih-dspark-enabled=2";
  EXPECT_FALSE(DeepSeekRankWorkerArguments::Parse(values).ok());
  values = valid();
  values[13] = "--pih-metadata-reassembly-bytes=134217729";
  EXPECT_FALSE(DeepSeekRankWorkerArguments::Parse(values).ok());
}

TEST(DeepSeekRankWorkerArgumentsTest, OwnsApplicationArguments) {
  auto values = valid();
  auto parsed = DeepSeekRankWorkerArguments::Parse(values).value();
  values[0] = "overwritten";

  ASSERT_EQ(parsed.application_arguments.size(), 1U);
  EXPECT_EQ(parsed.application_arguments[0], "--model=/weights");
}

}  // namespace
}  // namespace pih
