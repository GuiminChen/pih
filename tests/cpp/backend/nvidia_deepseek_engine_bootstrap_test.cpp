#include "pih/backend/cuda/nvidia_deepseek_engine_bootstrap.h"
#include "pih/backend/cuda/nvidia_deepseek_nccl_boundary_bootstrap.h"

#include <gtest/gtest.h>

#include <filesystem>

namespace pih { namespace {

TEST(NvidiaDeepSeekEngineBootstrapTest, FreezesOneToFourUniqueDeviceTopology) {
  auto valid = NvidiaDeepSeekBootstrapConfig::Create(
      7, 4, {0, 1, 2, 3}, DeepSeekRoutedExpertResidency::kHostSpill,
      2, 2, 1000);
  ASSERT_TRUE(valid.ok()) << valid.status().message();
  EXPECT_EQ(valid->world_size(), 4U);
  EXPECT_EQ(valid->device_ordinals(),
            std::vector<std::int32_t>({0, 1, 2, 3}));
  EXPECT_TRUE(valid->in_process_compute_supported());
  EXPECT_FALSE(valid->production_compute_supported());
  EXPECT_EQ(valid->execution_topology(),
            DeepSeekExecutionTopology::kInProcessRankSetDevelopment);
  auto pp1 = NvidiaDeepSeekBootstrapConfig::Create(
      7, 1, {0}, DeepSeekRoutedExpertResidency::kHostSpill,
      2, 2, 1000).value();
  EXPECT_TRUE(pp1.in_process_compute_supported());
  EXPECT_FALSE(pp1.production_compute_supported());
  EXPECT_EQ(valid->attention_reserved_tokens_per_sequence(), 4096U);
  EXPECT_FALSE(NvidiaDeepSeekBootstrapConfig::Create(
      7, 2, {0, 0}, DeepSeekRoutedExpertResidency::kHostSpill,
      2, 2, 1000).ok());
  EXPECT_FALSE(NvidiaDeepSeekBootstrapConfig::Create(
      7, 2, {0}, DeepSeekRoutedExpertResidency::kHostSpill,
      2, 2, 1000).ok());
}

TEST(NvidiaDeepSeekEngineBootstrapTest,
     FreezesExplicitAttentionContextReservation) {
  auto valid = NvidiaDeepSeekBootstrapConfig::Create(
      7, 1, {0}, DeepSeekRoutedExpertResidency::kHostSpill,
      2, 2, 1000, 32768);
  ASSERT_TRUE(valid.ok());
  EXPECT_EQ(valid->attention_reserved_tokens_per_sequence(), 32768U);
  EXPECT_FALSE(NvidiaDeepSeekBootstrapConfig::Create(
      7, 1, {0}, DeepSeekRoutedExpertResidency::kHostSpill,
      2, 2, 1000, 0).ok());
  EXPECT_FALSE(NvidiaDeepSeekBootstrapConfig::Create(
      7, 1, {0}, DeepSeekRoutedExpertResidency::kHostSpill,
      2, 2, 1000, 1048577).ok());
}

TEST(NvidiaDeepSeekEngineBootstrapTest, ResidencyOwnsExactPagerResources) {
  auto resident = NvidiaDeepSeekBootstrapConfig::Create(
      7, 1, {0}, DeepSeekRoutedExpertResidency::kFullResident,
      0, 0, 1000);
  ASSERT_TRUE(resident.ok());
  EXPECT_TRUE(resident->in_process_compute_supported());
  EXPECT_FALSE(resident->production_compute_supported());
  EXPECT_FALSE(NvidiaDeepSeekBootstrapConfig::Create(
      7, 1, {0}, DeepSeekRoutedExpertResidency::kFullResident,
      2, 2, 1000).ok());
  EXPECT_FALSE(NvidiaDeepSeekBootstrapConfig::Create(
      7, 1, {0}, DeepSeekRoutedExpertResidency::kHostSpill,
      1, 2, 1000).ok());
}

TEST(NvidiaDeepSeekEngineBootstrapTest,
     FreezesDevelopmentAndProductionExecutionTopologies) {
  auto production = NvidiaDeepSeekBootstrapConfig::Create(
      7, 2, {0, 1}, DeepSeekRoutedExpertResidency::kHostSpill,
      2, 2, 1000, 4096, DeepSeekExecutionTopology::kOneProcessPerRank);
  ASSERT_TRUE(production.ok()) << production.status().message();
  EXPECT_EQ(production->execution_topology(),
            DeepSeekExecutionTopology::kOneProcessPerRank);
  EXPECT_FALSE(production->in_process_compute_supported());
  EXPECT_FALSE(production->production_compute_supported());

  EXPECT_FALSE(NvidiaDeepSeekBootstrapConfig::Create(
      7, 1, {0}, DeepSeekRoutedExpertResidency::kHostSpill,
      2, 2, 1000, 4096, static_cast<DeepSeekExecutionTopology>(99)).ok());
}

TEST(NvidiaDeepSeekEngineBootstrapTest,
     RequiresAnExactH100DsparkRuntimePermit) {
  for (std::uint32_t world_size = 1; world_size <= 4; ++world_size) {
    auto enabled = DeepSeekPipelinePlan::Create(world_size, true).value();
    for (const auto family : {RuntimeProfileGpuFamily::kRtx4090D24GiB,
                              RuntimeProfileGpuFamily::kH100Pcie80GiB}) {
      auto rejected = NvidiaDeepSeekEngineBootstrap::ValidateDsparkCandidate(
          family, true, enabled);
      ASSERT_FALSE(rejected.ok());
      EXPECT_EQ(rejected.code(), StatusCode::kFailedPrecondition);
      if (family == RuntimeProfileGpuFamily::kH100Pcie80GiB) {
        EXPECT_EQ(rejected.message(),
                  "deepseek_dspark_runtime_permit_required");
      } else {
        EXPECT_EQ(rejected.message(),
                  "DeepSeek DSpark production is restricted to H100 PCIe");
      }
    }
  }
  auto disabled = DeepSeekPipelinePlan::Create(4, false).value();
  EXPECT_FALSE(NvidiaDeepSeekEngineBootstrap::ValidateDsparkCandidate(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, true, disabled).ok());
  EXPECT_TRUE(NvidiaDeepSeekEngineBootstrap::ValidateDsparkCandidate(
      RuntimeProfileGpuFamily::kRtx4090D24GiB, false, disabled).ok());
  EXPECT_TRUE(NvidiaDeepSeekEngineBootstrap::ValidateDsparkCandidate(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, false, disabled).ok());
}

TEST(NvidiaDeepSeekEngineBootstrapTest,
     RepositoryFactoryRejectsBeforeCudaResourceCreation) {
  auto config = NvidiaDeepSeekBootstrapConfig::Create(
      7, 1, {0}, DeepSeekRoutedExpertResidency::kFullResident,
      0, 0, 1000).value();
  auto capacity = DeepSeekPipelineCapacity::Create(1, 8, 8, 1, false).value();
  auto missing = NvidiaDeepSeekEngineBootstrap::
      BuildFromUncalibratedRepository(
          std::filesystem::path("missing-deepseek-repository"),
          1ULL << 32U, false, std::move(capacity), config);
  ASSERT_FALSE(missing.ok());
  EXPECT_EQ(missing.status().message(),
            "deepseek_target_generation_authority_required");

  const auto absent_root = Sha256Digest::ParseHex(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                               .value();
  auto target_capacity =
      DeepSeekPipelineCapacity::Create(1, 8, 8, 1, false).value();
  auto missing_target =
      NvidiaDeepSeekEngineBootstrap::BuildFromTargetGeneration(
          std::filesystem::path("missing-deepseek-target-generation"),
          absent_root, 1ULL << 32U, false, std::move(target_capacity), config);
  ASSERT_FALSE(missing_target.ok());
  EXPECT_NE(missing_target.status().message().find("open"),
            std::string::npos);

  auto dspark_capacity =
      DeepSeekPipelineCapacity::Create(1, 8, 8, 1, true).value();
  auto dspark = NvidiaDeepSeekEngineBootstrap::
      BuildFromUncalibratedRepository(
          std::filesystem::path("missing-deepseek-repository"),
          1ULL << 32U, true, std::move(dspark_capacity), config);
  ASSERT_FALSE(dspark.ok());
  EXPECT_EQ(dspark.status().code(), StatusCode::kFailedPrecondition);
  EXPECT_EQ(dspark.status().message(),
            "deepseek_dspark_runtime_permit_required");
}

class RejectingUniqueIdSource final : public DeepSeekNcclUniqueIdSource {
 public:
  Result<std::array<std::byte,
                    DeepSeekNcclBootstrapLease::kUniqueIdBytes>>
  get_unique_id() override {
    ++calls;
    return Status::Internal("must not be called");
  }
  int calls = 0;
};

class RejectingPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t, std::uint64_t) override {
    ++calls;
    return Status::Internal("must not allocate");
  }
  void deallocate(Allocation) noexcept override {}
  int calls = 0;
};

TEST(NvidiaDeepSeekNcclBoundaryBootstrapTest,
     RejectsInvalidTopologyBeforeUniqueIdOrCudaAccess) {
  RejectingUniqueIdSource source;
  RejectingPinnedAllocator pinned;
  auto result = NvidiaDeepSeekNcclBoundaryBootstrap::Build(
      {7, 11, 13, 2, 1, 17, 1000, 2000}, {}, {}, {}, {},
      pinned, nullptr, source);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(source.calls, 0);
  EXPECT_EQ(pinned.calls, 0);
}

} }  // namespace pih
