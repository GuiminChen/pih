#include "pih/model/deepseek_rank_materialization_allocation_census.h"

#include <atomic>
#include <gtest/gtest.h>
#include <new>

namespace pih {
namespace {

Sha256Digest digest(std::uint8_t value) {
  Sha256Digest result{};
  result.bytes.fill(static_cast<std::byte>(value));
  return result;
}

class TestCudaAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    void* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("test CUDA allocation failed");
    return Allocation{data, bytes, alignment, next_generation_++,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }

 private:
  std::atomic<std::uint64_t> next_generation_{1};
};

class TestPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    void* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("test pinned allocation failed");
    return Allocation{data, bytes, alignment, next_generation_++, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }

 private:
  std::atomic<std::uint64_t> next_generation_{100};
};

DeepSeekRankMaterializationAllocationAuthority authority(
    RuntimeProfileResidency residency) {
  DeepSeekRankMaterializationAllocationAuthority value;
  value.device_ordinal = 0;
  value.residency = residency;
  value.physical_gpu_identity = digest(1);
  value.cuda_allocation_identity = 10;
  value.cuda_owner_identity = digest(2);
  value.cuda_resource_identity = digest(3);
  if (residency == RuntimeProfileResidency::kHostSpill) {
    value.pinned_registration_identity = 11;
    value.pinned_owner_identity = digest(4);
    value.pinned_resource_identity = digest(5);
    value.pinned_numa_node = 0;
  }
  return value;
}

TEST(DeepSeekRankMaterializationAllocationCensusTest,
     CapturesTrackedHostSpillBackingsAndRejectsRelease) {
  TestCudaAllocator device;
  TestPinnedAllocator pinned;
  constexpr std::uint64_t kWeightBytes = 4096;
  const auto staging_bytes = 2 * DeepSeekExpertBundleLayout::kBundleBytes;
  auto census = DeepSeekRankMaterializationAllocationCensus::Create(
      authority(RuntimeProfileResidency::kHostSpill), kWeightBytes,
      staging_bytes, device, pinned);
  ASSERT_TRUE(census.ok()) << census.status().message();
  auto weight = census->get()->device_allocator().allocate(
      kWeightBytes, DeepSeekWeightMaterializationPlan::kFinalAlignment);
  ASSERT_TRUE(weight.ok()) << weight.status().message();
  auto staging = census->get()->pinned_allocator()->allocate(
      staging_bytes, DeepSeekExpertBundleLayout::kAlignment);
  ASSERT_TRUE(staging.ok()) << staging.status().message();
  auto observed = (*census)->capture();
  ASSERT_TRUE(observed.ok()) << observed.status().message();
  EXPECT_GT(observed->sample_identity, 0U);
  EXPECT_LE(observed->sample_started_ns, observed->sample_completed_ns);
  EXPECT_EQ(observed->cuda_resident_weight_allocation_bytes, kWeightBytes);
  EXPECT_EQ(observed->pinned_staging_allocation_bytes, staging_bytes);
  EXPECT_NE(observed->cuda_allocation_root, Sha256Digest{});
  EXPECT_NE(observed->pinned_allocation_root, Sha256Digest{});
  (*census)->pinned_allocator()->deallocate(*staging);
  (*census)->device_allocator().deallocate(*weight);
  EXPECT_FALSE((*census)->capture().ok());
}

TEST(DeepSeekRankMaterializationAllocationCensusTest,
     FullResidentUsesCanonicalZeroPinnedSentinel) {
  TestCudaAllocator device;
  TestPinnedAllocator pinned;
  constexpr std::uint64_t kWeightBytes = 4096;
  auto census = DeepSeekRankMaterializationAllocationCensus::Create(
      authority(RuntimeProfileResidency::kFullResident), kWeightBytes, 0,
      device, pinned);
  ASSERT_TRUE(census.ok()) << census.status().message();
  EXPECT_EQ((*census)->pinned_allocator(), nullptr);
  auto weight = (*census)->device_allocator().allocate(
      kWeightBytes, DeepSeekWeightMaterializationPlan::kFinalAlignment);
  ASSERT_TRUE(weight.ok()) << weight.status().message();
  auto observed = (*census)->capture();
  ASSERT_TRUE(observed.ok()) << observed.status().message();
  EXPECT_EQ(observed->pinned_staging_allocation_bytes, 0U);
  EXPECT_EQ(observed->pinned_allocation_root, Sha256Digest{});
  (*census)->device_allocator().deallocate(*weight);
}

TEST(DeepSeekRankMaterializationAllocationCensusTest,
     RejectsPinnedAuthorityAndResidencySplice) {
  TestCudaAllocator device;
  TestPinnedAllocator pinned;
  auto full = authority(RuntimeProfileResidency::kFullResident);
  full.pinned_registration_identity = 11;
  full.pinned_owner_identity = digest(4);
  full.pinned_resource_identity = digest(5);
  full.pinned_numa_node = 0;
  EXPECT_FALSE(DeepSeekRankMaterializationAllocationCensus::Create(
                   full, 4096, DeepSeekExpertBundleLayout::kBundleBytes,
                   device, pinned)
                   .ok());
  auto spill = authority(RuntimeProfileResidency::kHostSpill);
  spill.pinned_registration_identity = 0;
  spill.pinned_owner_identity = Sha256Digest{};
  spill.pinned_resource_identity = Sha256Digest{};
  spill.pinned_numa_node = -1;
  EXPECT_FALSE(DeepSeekRankMaterializationAllocationCensus::Create(
                   spill, 4096, 0, device, pinned)
                   .ok());
}

TEST(DeepSeekRankMaterializationAllocationCensusTest,
     AuthorityRootBindsEveryControllerSuppliedCoordinate) {
  const auto baseline = authority(RuntimeProfileResidency::kHostSpill);
  const auto root =
      compile_deepseek_rank_materialization_allocation_authority_root(
          baseline);
  ASSERT_TRUE(root.ok()) << root.status().message();

  auto mutated = baseline;
  mutated.physical_gpu_identity = digest(6);
  EXPECT_NE(*root,
            compile_deepseek_rank_materialization_allocation_authority_root(
                mutated)
                .value());
  mutated = baseline;
  mutated.cuda_allocation_identity++;
  EXPECT_NE(*root,
            compile_deepseek_rank_materialization_allocation_authority_root(
                mutated)
                .value());
  mutated = baseline;
  mutated.cuda_owner_identity = digest(7);
  EXPECT_NE(*root,
            compile_deepseek_rank_materialization_allocation_authority_root(
                mutated)
                .value());
  mutated = baseline;
  mutated.cuda_resource_identity = digest(8);
  EXPECT_NE(*root,
            compile_deepseek_rank_materialization_allocation_authority_root(
                mutated)
                .value());
  mutated = baseline;
  mutated.pinned_registration_identity++;
  EXPECT_NE(*root,
            compile_deepseek_rank_materialization_allocation_authority_root(
                mutated)
                .value());
  mutated = baseline;
  mutated.pinned_owner_identity = digest(9);
  EXPECT_NE(*root,
            compile_deepseek_rank_materialization_allocation_authority_root(
                mutated)
                .value());
  mutated = baseline;
  mutated.pinned_resource_identity = digest(10);
  EXPECT_NE(*root,
            compile_deepseek_rank_materialization_allocation_authority_root(
                mutated)
                .value());
  mutated = baseline;
  mutated.pinned_numa_node = 1;
  EXPECT_NE(*root,
            compile_deepseek_rank_materialization_allocation_authority_root(
                mutated)
                .value());
}

TEST(DeepSeekRankMaterializationAllocationCensusTest,
     FullResidentAuthorityCanonicalizesOnlyTheZeroPinnedShape) {
  const auto full = authority(RuntimeProfileResidency::kFullResident);
  EXPECT_TRUE(
      compile_deepseek_rank_materialization_allocation_authority_root(full)
          .ok());
  auto invalid = full;
  invalid.pinned_numa_node = 0;
  EXPECT_FALSE(
      compile_deepseek_rank_materialization_allocation_authority_root(invalid)
          .ok());
}

}  // namespace
}  // namespace pih
