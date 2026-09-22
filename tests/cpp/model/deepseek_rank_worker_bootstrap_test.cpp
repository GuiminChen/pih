#include "pih/model/deepseek_rank_worker_bootstrap.h"

#include <gtest/gtest.h>

#include <utility>

namespace pih {
namespace {

std::vector<std::string_view> valid_arguments() {
  return {"worker", "--model=/weights", "--pih-rank=1",
          "--pih-world-size=2", "--pih-engine-epoch=7",
          "--pih-worker-generation=8", "--pih-device-identity=10",
          "--pih-startup-device-ordinal=5",
          "--pih-process-manifest=11",
          "--pih-startup-deadline-ns=200",
          "--pih-controller-pid=90", "--pih-controller-pidfd=12",
          "--pih-control-fd=13", "--pih-dspark-enabled=0",
          "--pih-metadata-reassembly-bytes=134217728"};
}

Sha256Digest commitment() {
  Sha256Digest value{};
  value.bytes.fill(std::byte{7});
  return value;
}

class Operations final : public DeepSeekRankWorkerHandshakeOperations {
 public:
  Result<std::optional<std::vector<std::byte>>> receive_challenge(
      std::int32_t) override {
    ++receives;
    if (receives == 1) return std::optional<std::vector<std::byte>>{};
    DeepSeekRankExecChallenge challenge{
        1, {7, 8, 2, 1, 10, 11, commitment(), 5, 200},
        {100, 200, 300}, 90, 400};
    const auto frame = encode_deepseek_rank_challenge(challenge);
    return std::optional<std::vector<std::byte>>{
        std::vector<std::byte>(frame.begin(), frame.end())};
  }
  Result<DeepSeekRankExecObservation> collect_observation(
      const DeepSeekRankWorkerArguments& arguments) override {
    return DeepSeekRankExecObservation{
        arguments.manifest, 100, 90, 90, 10, commitment(), 5, 9, true, true};
  }
  Status send_ready(std::int32_t, std::span<const std::byte>) override {
    ++sends;
    return sends == 1 ? Status::Unavailable("busy") : Status::Ok();
  }
  int receives = 0;
  int sends = 0;
};

class Runtime final : public DeepSeekRankWorkerBootstrapRuntime {
 public:
  Result<std::uint64_t> monotonic_now_ns() override { return times[index++]; }
  Status wait_for_control(std::int32_t fd, DeepSeekRankWorkerWaitEvent event,
                          std::uint64_t deadline) override {
    fds.push_back(fd); events.push_back(event); deadlines.push_back(deadline);
    return wait_status;
  }
  std::vector<std::uint64_t> times{100, 101, 102};
  std::size_t index = 0;
  Status wait_status = Status::Ok();
  std::vector<std::int32_t> fds;
  std::vector<DeepSeekRankWorkerWaitEvent> events;
  std::vector<std::uint64_t> deadlines;
};

TEST(DeepSeekRankWorkerBootstrapTest, WaitsByHandshakePhaseAndReturnsAppArgs) {
  Operations operations;
  Runtime runtime;
  auto arguments = DeepSeekRankWorkerArguments::Parse(
      valid_arguments()).value();
  auto result = DeepSeekRankWorkerBootstrap::Run(
      std::move(arguments), operations, runtime);
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->arguments.application_arguments.size(), 2U);
  EXPECT_EQ(result->arguments.application_arguments[1], "--model=/weights");
  EXPECT_EQ(result->exec_ready.receipt.process_identity, 100U);
  EXPECT_EQ(result->exec_ready.receipt.pidfd_identity, 200U);
  EXPECT_EQ(result->exec_ready.receipt.control_identity, 300U);
  EXPECT_EQ(result->exec_ready.receipt.physical_device_uuid_commitment,
            commitment());
  EXPECT_FALSE(result->arguments.expected_dspark_enabled);
  EXPECT_EQ(result->arguments.maximum_metadata_reassembly_bytes,
            134217728U);
  EXPECT_EQ(result->exec_ready.challenge_identity, 400U);
  EXPECT_EQ(runtime.events,
            (std::vector<DeepSeekRankWorkerWaitEvent>{
                DeepSeekRankWorkerWaitEvent::kChallengeReadable,
                DeepSeekRankWorkerWaitEvent::kReadyWritable}));
  EXPECT_EQ(runtime.fds, (std::vector<std::int32_t>{13, 13}));
  EXPECT_EQ(runtime.deadlines, (std::vector<std::uint64_t>{200, 200}));
}

TEST(DeepSeekRankWorkerBootstrapTest, PropagatesClockWaitAndDeadlineFailures) {
  {
    Operations operations;
    Runtime runtime;
    runtime.times = {200};
    auto result = DeepSeekRankWorkerBootstrap::Run(
        valid_arguments(), operations, runtime);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kDeadlineExceeded);
  }
  {
    Operations operations;
    Runtime runtime;
    runtime.wait_status = Status::Internal("poll failed");
    auto result = DeepSeekRankWorkerBootstrap::Run(
        valid_arguments(), operations, runtime);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInternal);
  }
}

}  // namespace
}  // namespace pih
