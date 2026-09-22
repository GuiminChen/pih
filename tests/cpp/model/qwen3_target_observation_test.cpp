#include "pih/model/qwen3_target_observation.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

std::string observation_run_json(std::string_view gpu = "RTX_4090_D",
                                 std::string_view sm = "sm_89") {
  return "{\"schema\":\"pih.qwen_bf16_numerical_run.v1\","
         "\"model_sha256\":\"" + std::string(64, 'a') +
         "\",\"fixture_sha256\":\"" + std::string(64, 'b') +
         "\",\"tolerance_sha256\":\"" + std::string(64, 'c') +
         "\",\"kernel_bundle_sha256\":\"" + std::string(64, 'd') +
         "\",\"build_sha256\":\"" + std::string(64, 'e') +
         "\",\"environment_sha256\":\"" + std::string(64, 'f') +
         "\",\"target_gpu\":\"" + std::string(gpu) +
         "\",\"target_sm\":\"" + std::string(sm) +
         "\",\"run_generation\":7}";
}

QwenNumericalRunIdentity observation_identity(
    std::string_view gpu = "RTX_4090_D", std::string_view sm = "sm_89") {
  return QwenNumericalRunIdentity::Parse(observation_run_json(gpu, sm)).value();
}

QwenObservedDevice rtx4090d() {
  QwenObservedDevice device{};
  device.visible_device_count = 1;
  device.current_ordinal = 0;
  device.name = "NVIDIA GeForce RTX 4090 D";
  device.compute_major = 8;
  device.compute_minor = 9;
  device.total_global_memory_bytes = 24ULL * 1024 * 1024 * 1024;
  device.uuid[0] = std::byte{1};
  device.pci_domain = 0;
  device.pci_bus = 3;
  device.pci_device = 0;
  device.driver_version = 13020;
  device.runtime_version = 13020;
  return device;
}

TEST(QwenTargetObservationTest, BindsExact4090DAndProducesStableDigest) {
  auto first = QwenTargetObservation::Create(observation_identity(), rtx4090d());
  auto second = QwenTargetObservation::Create(observation_identity(), rtx4090d());
  ASSERT_TRUE(first.ok()) << first.status().message();
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(first->semantic_digest(), second->semantic_digest());
  EXPECT_EQ(first->target_gpu(), QwenTargetGpu::kRtx4090D);
  EXPECT_EQ(first->target_sm(), 89);
}

TEST(QwenTargetObservationTest, AcceptsExactH100PcieIdentity) {
  auto device = rtx4090d();
  device.name = "NVIDIA H100 PCIe";
  device.compute_major = 9;
  device.compute_minor = 0;
  device.total_global_memory_bytes = 80ULL * 1000 * 1000 * 1000;
  auto result = QwenTargetObservation::Create(
      observation_identity("H100_PCIE_80GB", "sm_90"), device);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result->target_gpu(), QwenTargetGpu::kH100Pcie80Gb);
}

TEST(QwenTargetObservationTest, RejectsRelabelTopologyAndVersionDrift) {
  auto device = rtx4090d();
  device.name = "NVIDIA GeForce RTX 4090";
  EXPECT_FALSE(
      QwenTargetObservation::Create(observation_identity(), device).ok());
  device = rtx4090d();
  device.compute_minor = 0;
  EXPECT_FALSE(
      QwenTargetObservation::Create(observation_identity(), device).ok());
  device = rtx4090d();
  device.visible_device_count = 2;
  EXPECT_FALSE(
      QwenTargetObservation::Create(observation_identity(), device).ok());
  device = rtx4090d();
  device.uuid.fill(std::byte{0});
  EXPECT_FALSE(
      QwenTargetObservation::Create(observation_identity(), device).ok());
  device = rtx4090d();
  device.driver_version = 0;
  EXPECT_FALSE(
      QwenTargetObservation::Create(observation_identity(), device).ok());
}

}  // namespace
}  // namespace pih
