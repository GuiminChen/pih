#include "pih/model/engine_gpu_process_inventory.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest lease_digest() {
  Sha256Digest value{}; value.bytes.fill(std::byte{5}); return value;
}
EngineGpuProcessInventoryReceipt receipt(bool visible,
                                         std::vector<std::uint64_t> p0 = {},
                                         std::vector<std::uint64_t> p1 = {}) {
  return {7, lease_digest(), 1, 100, 101, visible,
          {{10, std::move(p0)}, {20, std::move(p1)}}};
}

TEST(EngineGpuProcessInventoryTest, DistinguishesUnknownOccupiedAndEmpty) {
  const std::uint64_t devices[]{10, 20};
  auto gate = EngineGpuProcessInventoryGate::Create(
      7, lease_digest(), devices).value();
  ASSERT_EQ(*gate.accept(receipt(false)),
            EngineGpuProcessInventoryState::kUnknown);
  auto occupied = receipt(true, {}, {100});
  occupied.sample_identity = 2; occupied.sample_started_ns = 101;
  occupied.sample_completed_ns = 102;
  ASSERT_EQ(*gate.accept(occupied), EngineGpuProcessInventoryState::kOccupied);
  auto empty = receipt(true); empty.sample_identity = 3;
  empty.sample_started_ns = 102; empty.sample_completed_ns = 103;
  ASSERT_EQ(*gate.accept(empty), EngineGpuProcessInventoryState::kEmpty);
  EXPECT_EQ(gate.state(), EngineGpuProcessInventoryState::kEmpty);
}

TEST(EngineGpuProcessInventoryTest, RejectsInvalidManifest) {
  EXPECT_FALSE(EngineGpuProcessInventoryGate::Create(0, lease_digest(),
                                                      std::array{10ULL}).ok());
  EXPECT_FALSE(EngineGpuProcessInventoryGate::Create(7, {},
                                                      std::array{10ULL}).ok());
  EXPECT_FALSE(EngineGpuProcessInventoryGate::Create(
      7, lease_digest(), std::array<std::uint64_t, 0>{}).ok());
  EXPECT_FALSE(EngineGpuProcessInventoryGate::Create(
      7, lease_digest(), std::array{20ULL, 10ULL}).ok());
  EXPECT_FALSE(EngineGpuProcessInventoryGate::Create(
      7, lease_digest(), std::array{10ULL, 10ULL}).ok());
}

TEST(EngineGpuProcessInventoryTest, IdentityCoverageAndTimeDriftPoison) {
  const std::uint64_t devices[]{10, 20};
  for (int mutation = 0; mutation < 7; ++mutation) {
    auto gate = EngineGpuProcessInventoryGate::Create(
        7, lease_digest(), devices).value();
    auto value = receipt(true);
    if (mutation == 0) value.engine_generation = 8;
    if (mutation == 1) value.allocation_lease_digest.bytes[0] = std::byte{9};
    if (mutation == 2) value.devices.pop_back();
    if (mutation == 3) value.devices[1].physical_gpu_identity = 10;
    if (mutation == 4) value.sample_identity = 2;
    if (mutation == 5) value.sample_completed_ns = 99;
    if (mutation == 6) value.devices[0].sorted_process_identities = {2, 1};
    EXPECT_FALSE(gate.accept(value).ok()) << mutation;
    EXPECT_TRUE(gate.poisoned()) << mutation;
  }
}

}  // namespace
}  // namespace pih
