#include "pih/model/deepseek_rank_expert_device_resources.h"
#include "pih/model/deepseek_rank_compute_infrastructure.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

class RecordingCudaAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++calls;
    requested_bytes.push_back(bytes);
    requested_alignments.push_back(alignment);
    if (fail_on_call == calls) {
      return Status::ResourceExhausted("injected device allocation failure");
    }
    const auto address = UINT64_C(0x100000000) +
                         static_cast<std::uint64_t>(calls) *
                             UINT64_C(0x100000000);
    return Allocation{
        reinterpret_cast<void*>(address), bytes, alignment,
        static_cast<std::uint64_t>(calls),
        Device::Create(DeviceType::kCuda, device_ordinal).value()};
  }

  void deallocate(Allocation allocation) noexcept override {
    ++releases;
    if (runtime_alive != nullptr && !*runtime_alive) {
      released_after_runtime = true;
    }
    released_generations.push_back(allocation.generation);
  }

  int calls = 0;
  int releases = 0;
  int fail_on_call = 0;
  std::int32_t device_ordinal = 0;
  std::vector<std::uint64_t> requested_bytes;
  std::vector<std::uint64_t> requested_alignments;
  std::vector<std::uint64_t> released_generations;
  bool* runtime_alive = nullptr;
  bool released_after_runtime = false;
};

TEST(DeepSeekRankExpertDeviceResourcesTest,
     OwnsSlotsAndEveryComputeBackingWithCanonicalIdentity) {
  RecordingCudaAllocator allocator;
  constexpr std::uint32_t kSlots = 3;
  constexpr std::uint32_t kTokens = 17;
  constexpr std::uint64_t kContext = 91;
  auto layout = DeepSeekExpertComputeArenaLayout::Create(kTokens).value();
  {
    auto resources = DeepSeekRankExpertDeviceResources::Allocate(
        allocator, kSlots, kTokens, kContext, 0);
    ASSERT_TRUE(resources.ok()) << resources.status().message();
    EXPECT_EQ(allocator.calls, 6);
    ASSERT_EQ(resources->slot_bases().size(), kSlots);
    EXPECT_EQ(resources->slots().slot_count(), kSlots);
    EXPECT_EQ(resources->maximum_tokens(), kTokens);
    EXPECT_EQ(resources->compute_backing_bytes(), layout.required_bytes());
    EXPECT_EQ(resources->source_hidden_bytes(), UINT64_C(17) * 4096 * 2);
    EXPECT_EQ(resources->accumulator_bytes(), UINT64_C(17) * 4096 * 4);
    EXPECT_EQ(resources->arena().route_input_bf16.address,
              resources->compute_backing_address());
    for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
      EXPECT_EQ(allocator.requested_bytes[slot],
                DeepSeekExpertPager::kBundleBytes);
    }
    for (const auto alignment : allocator.requested_alignments) {
      EXPECT_EQ(alignment, DeepSeekExpertComputeArenaLayout::kAlignment);
    }
  }
  EXPECT_EQ(allocator.releases, 6);
}

TEST(DeepSeekRankExpertDeviceResourcesTest,
     PartialAllocationFailureReleasesEveryPriorBacking) {
  RecordingCudaAllocator allocator;
  allocator.fail_on_call = 5;
  EXPECT_FALSE(DeepSeekRankExpertDeviceResources::Allocate(
                   allocator, 3, 17, 91, 0)
                   .ok());
  EXPECT_EQ(allocator.calls, 5);
  EXPECT_EQ(allocator.releases, 4);
}

TEST(DeepSeekRankExpertDeviceResourcesTest,
     ResidentOnlyAllocationKeepsComputeWorkspaceWithoutSlots) {
  RecordingCudaAllocator allocator;
  auto resources = DeepSeekRankExpertDeviceResources::Allocate(
      allocator, 0, 17, 91, 0);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_EQ(resources->slots().slot_count(), 0U);
  EXPECT_TRUE(resources->slot_bases().empty());
  EXPECT_EQ(allocator.calls, 3);
  EXPECT_NE(resources->compute_backing_address(), 0U);
  EXPECT_NE(resources->source_hidden_bf16(), 0U);
  EXPECT_NE(resources->accumulator_f32(), 0U);
  const auto shared = resources->shared_arena();
  EXPECT_EQ(shared.activation_e4m3.address, resources->arena().activation_e4m3.address);
  EXPECT_EQ(shared.shared_output_bf16.address, resources->arena().expert_output_bf16.address);
  EXPECT_GE(shared.shared_output_bf16.bytes, 17U * 4096U * 2U);
  EXPECT_NE(shared.shared_output_bf16.address, resources->source_hidden_bf16());
  EXPECT_NE(shared.shared_output_bf16.address, resources->accumulator_f32());
}

TEST(DeepSeekRankExpertDeviceResourcesTest,
     RejectsAllocatorDeviceMismatchImmediately) {
  RecordingCudaAllocator allocator;
  allocator.device_ordinal = 1;
  EXPECT_FALSE(DeepSeekRankExpertDeviceResources::Allocate(
                   allocator, 2, 8, 91, 0)
                   .ok());
  EXPECT_EQ(allocator.calls, 1);
  EXPECT_EQ(allocator.releases, 1);
}

TEST(DeepSeekRankExpertDeviceResourcesTest, RejectsInvalidRuntimeIdentity) {
  RecordingCudaAllocator allocator;
  EXPECT_FALSE(DeepSeekRankExpertDeviceResources::Allocate(
                   allocator, 2, 8, 0, 0)
                   .ok());
  EXPECT_EQ(allocator.calls, 0);
}

TEST(DeepSeekRankComputeInfrastructureTest, OwnsEveryDeviceBacking) {
  RecordingCudaAllocator allocator;
  {
    auto device_resources = DeepSeekRankExpertDeviceResources::Allocate(
        allocator, 2, 8, 91, 0).value();
    auto infrastructure = DeepSeekRankComputeInfrastructure::Create(
        std::move(device_resources));
    ASSERT_TRUE(infrastructure.ok()) << infrastructure.status().message();
    EXPECT_EQ(infrastructure->get()->device_resources().slots().slot_count(),
              2U);
  }
  EXPECT_EQ(allocator.releases, 5);
}

}  // namespace
}  // namespace pih
