#include "pih/model/deepseek_dspark_gpu_state_digest.h"

#include <array>

#include <gtest/gtest.h>

namespace pih { namespace {

class Ops final : public DeepSeekDsparkGpuStateDigestOperations {
 public:
  Status launch_component_sha256(
      const DeepSeekDsparkGpuStateDigestSubmission& value) override {
    for (std::size_t i = 0; i < value.components.size(); ++i) {
      const auto& component = value.components[i];
      auto digest = sha256({reinterpret_cast<const std::byte*>(
                                component.device_address),
                            static_cast<std::size_t>(component.bytes)});
      if (!digest.ok()) return digest.status();
      value.host_component_digests[i] = *digest;
    }
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return ready ? DeepSeekExpertAsyncStatus::kSuccess
                 : DeepSeekExpertAsyncStatus::kInProgress;
  }
  bool ready = false;
};

TEST(DeepSeekDsparkGpuStateDigestTest, WaitsAndBindsOrderedGpuComponents) {
  std::array<std::byte, 4> fixed{std::byte{1}, std::byte{2},
                                 std::byte{3}, std::byte{4}};
  std::array<std::byte, 3> page{std::byte{5}, std::byte{6}, std::byte{7}};
  std::array<DeepSeekDsparkDeviceStateComponent, 2> components{{
      {DeepSeekDsparkStateComponentKind::kFixed, 0,
       reinterpret_cast<std::uintptr_t>(fixed.data()), fixed.size()},
      {DeepSeekDsparkStateComponentKind::kRatio4Main, 3,
       reinterpret_cast<std::uintptr_t>(page.data()), page.size()}}};
  std::array<Sha256Digest, 2> digests{};
  std::uint32_t error = 0;
  Ops ops;
  auto coordinator = DeepSeekDsparkGpuStateDigestCoordinator::Create(2, ops);
  ASSERT_TRUE(coordinator.ok());
  DeepSeekDsparkGpuStateDigestSubmission submission{
      1, 9, 4, 5, 2, components, 1, 64, digests, 2, &error, 3, 4};
  ASSERT_TRUE(coordinator->launch(submission).ok());
  auto waiting = coordinator->poll(); ASSERT_TRUE(waiting.ok());
  EXPECT_EQ(waiting->status(), DeepSeekExpertAsyncStatus::kInProgress);
  ops.ready = true;
  auto complete = coordinator->poll(); ASSERT_TRUE(complete.ok());
  EXPECT_EQ(complete->status(), DeepSeekExpertAsyncStatus::kSuccess);
  EXPECT_NE(complete->local_state_hash(), Sha256Digest{});

  page[0] = std::byte{8};
  Ops second_ops;
  auto second = DeepSeekDsparkGpuStateDigestCoordinator::Create(2, second_ops);
  ASSERT_TRUE(second.ok());
  ASSERT_TRUE(second->launch(submission).ok());
  second_ops.ready = true;
  auto changed = second->poll(); ASSERT_TRUE(changed.ok());
  EXPECT_NE(changed->local_state_hash(), complete->local_state_hash());
}

TEST(DeepSeekDsparkGpuStateDigestTest, RejectsAmbiguousOrderAndPoisonsGpuFailure) {
  std::array<std::byte, 1> byte{std::byte{1}};
  std::array<DeepSeekDsparkDeviceStateComponent, 2> unordered{{
      {DeepSeekDsparkStateComponentKind::kRatio128, 1,
       reinterpret_cast<std::uintptr_t>(byte.data()), 1},
      {DeepSeekDsparkStateComponentKind::kFixed, 0,
       reinterpret_cast<std::uintptr_t>(byte.data()), 1}}};
  std::array<Sha256Digest, 2> digests{};
  std::uint32_t error = 0;
  Ops ops;
  auto coordinator = DeepSeekDsparkGpuStateDigestCoordinator::Create(2, ops);
  ASSERT_TRUE(coordinator.ok());
  EXPECT_FALSE(coordinator->launch(
      {0, 1, 1, 2, 0, unordered, 1, 64, digests, 2, &error, 3, 4}).ok());

  std::array<DeepSeekDsparkDeviceStateComponent, 1> valid{{
      {DeepSeekDsparkStateComponentKind::kFixed, 0,
       reinterpret_cast<std::uintptr_t>(byte.data()), 1}}};
  std::array<Sha256Digest, 1> valid_digest{};
  ASSERT_TRUE(coordinator->launch(
      {0, 1, 1, 2, 0, valid, 1, 32, valid_digest, 2, &error, 3, 4}).ok());
  error = 1; ops.ready = true;
  auto failed = coordinator->poll(); ASSERT_TRUE(failed.ok());
  EXPECT_EQ(failed->status(), DeepSeekExpertAsyncStatus::kError);
  EXPECT_FALSE(coordinator->poll().ok());
}

} }  // namespace pih
