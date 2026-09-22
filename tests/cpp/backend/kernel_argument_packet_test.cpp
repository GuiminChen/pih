#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "pih/backend/cuda/kernel_argument_packet.h"

namespace pih {
namespace {

KernelSignatureManifest signature() {
  const std::string digest(64, 'b');
  const std::vector<KernelParameterSpec> parameters{
      {"elements", KernelWireType::kU32, 0, "nonzero_elements_v1"},
      {"input", KernelWireType::kDevicePointerU64, 8, "bf16_read_span_v1"},
      {"output", KernelWireType::kDevicePointerU64, 16,
       "bf16_write_span_v1"}};
  return KernelSignatureManifest::Create("residual", digest, digest, parameters, 24)
      .value();
}

KernelPointerContract pointer_contract(std::string id, KernelPointerAccess access,
                                       std::int32_t device = 0,
                                       std::uint32_t alias_group = 0,
                                       bool allow_alias = false) {
  return KernelPointerContract::Create(
             std::move(id), access, KernelPointerOwnerClass::kActivation, 0,
             device, 16, 2, alias_group, allow_alias)
      .value();
}

std::vector<KernelPointerContract> contracts() {
  return {pointer_contract("bf16_read_span_v1", KernelPointerAccess::kRead),
          pointer_contract("bf16_write_span_v1", KernelPointerAccess::kWrite)};
}

VerifiedDevicePointer pointer(std::uintptr_t address,
                              const KernelPointerContract& contract,
                              std::uint64_t generation = 1,
                              KernelPointerOwnerClass owner =
                                  KernelPointerOwnerClass::kActivation,
                              std::int32_t rank = 0) {
  const std::array<std::int64_t, 1> shape{8};
  const auto cuda =
      Device::Create(DeviceType::kCuda, contract.device_index()).value();
  auto view = TensorView::Create(reinterpret_cast<void*>(address),
                                 DType::kBFloat16, shape, {}, cuda, generation);
  return VerifiedDevicePointer::Create(view.value(), contract, owner, rank,
                                       generation)
      .value();
}

template <typename T>
T read_cell(const KernelArgumentPacket& packet, std::size_t ordinal) {
  T value{};
  std::memcpy(&value, packet.argument_cell(ordinal), sizeof(value));
  return value;
}

TEST(KernelArgumentPacketTest, WritesTypedCellsAndPublishesOnlyWhenComplete) {
  const auto pointer_contracts = contracts();
  auto packet = KernelArgumentPacket::Create(signature(), pointer_contracts);
  ASSERT_TRUE(packet.ok());
  EXPECT_FALSE(packet->ready_kernel_params().ok());
  ASSERT_TRUE(packet->set_u32(0, 17).ok());
  ASSERT_TRUE(packet->set_device_pointer(
      1, pointer(0x1000, pointer_contracts[0])).ok());
  EXPECT_FALSE(packet->ready());
  ASSERT_TRUE(packet->set_device_pointer(
      2, pointer(0x2000, pointer_contracts[1])).ok());
  ASSERT_TRUE(packet->ready());
  EXPECT_TRUE(packet->ready_kernel_params().ok());
  EXPECT_EQ(read_cell<std::uint32_t>(*packet, 0), 17);
  EXPECT_EQ(read_cell<std::uint64_t>(*packet, 1), 0x1000);
}

TEST(KernelArgumentPacketTest, RejectsWrongWidthDuplicateOrdinalAndNullPointer) {
  const auto pointer_contracts = contracts();
  auto packet = KernelArgumentPacket::Create(signature(), pointer_contracts);
  ASSERT_TRUE(packet.ok());
  EXPECT_FALSE(packet->set_u64(0, 1).ok());
  EXPECT_FALSE(packet->set_u32(9, 1).ok());
  ASSERT_TRUE(packet->set_u32(0, 1).ok());
  EXPECT_FALSE(packet->set_u32(0, 2).ok());
}

TEST(KernelArgumentPacketTest, PointerBindingRejectsCpuStaleSpanAndAlignment) {
  const auto contract = pointer_contract("bf16_read_span_v1",
                                         KernelPointerAccess::kRead);
  const std::array<std::int64_t, 1> shape{8};
  auto cpu = TensorView::Create(reinterpret_cast<void*>(0x1000),
                                DType::kBFloat16, shape, {}, Device::Cpu(), 1);
  ASSERT_TRUE(cpu.ok());
  EXPECT_FALSE(VerifiedDevicePointer::Create(
                   cpu.value(), contract, KernelPointerOwnerClass::kActivation,
                   0, 1)
                   .ok());

  const auto cuda = Device::Create(DeviceType::kCuda, 0).value();
  auto stale = TensorView::Create(reinterpret_cast<void*>(0x1000),
                                  DType::kBFloat16, shape, {}, cuda, 0);
  ASSERT_TRUE(stale.ok());
  EXPECT_FALSE(VerifiedDevicePointer::Create(
                   stale.value(), contract,
                   KernelPointerOwnerClass::kActivation, 0, 0)
                   .ok());
  auto live = TensorView::Create(reinterpret_cast<void*>(0x1002),
                                 DType::kBFloat16, shape, {}, cuda, 9);
  ASSERT_TRUE(live.ok());
  auto oversized = KernelPointerContract::Create(
      "bf16_read_span_v1", KernelPointerAccess::kRead,
      KernelPointerOwnerClass::kActivation, 0, 0, 18, 2, 0, false);
  ASSERT_TRUE(oversized.ok());
  EXPECT_FALSE(VerifiedDevicePointer::Create(
                   live.value(), oversized.value(),
                   KernelPointerOwnerClass::kActivation, 0, 9)
                   .ok());
  auto aligned = KernelPointerContract::Create(
      "bf16_read_span_v1", KernelPointerAccess::kRead,
      KernelPointerOwnerClass::kActivation, 0, 0, 16, 4, 0, false);
  ASSERT_TRUE(aligned.ok());
  EXPECT_FALSE(VerifiedDevicePointer::Create(
                   live.value(), aligned.value(),
                   KernelPointerOwnerClass::kActivation, 0, 9)
                   .ok());
  EXPECT_FALSE(VerifiedDevicePointer::Create(
                   live.value(), contract, KernelPointerOwnerClass::kWeight, 0,
                   9)
                   .ok());
}

TEST(KernelArgumentPacketTest, RejectsMissingContractWrongIdAndPointerOverlap) {
  const auto manifest = signature();
  EXPECT_FALSE(KernelArgumentPacket::Create(manifest).ok());
  auto pointer_contracts = contracts();
  auto packet = KernelArgumentPacket::Create(manifest, pointer_contracts);
  ASSERT_TRUE(packet.ok());
  EXPECT_FALSE(packet->set_device_pointer(
      1, pointer(0x1000, pointer_contracts[1])).ok());
  ASSERT_TRUE(packet->set_device_pointer(
      1, pointer(0x1000, pointer_contracts[0])).ok());
  EXPECT_FALSE(packet->set_device_pointer(
      2, pointer(0x1008, pointer_contracts[1])).ok());
}

TEST(KernelArgumentPacketTest, AllowsOnlyExplicitExactAliasGroup) {
  const std::vector<KernelPointerContract> alias_contracts{
      pointer_contract("bf16_read_span_v1", KernelPointerAccess::kRead, 0, 7,
                       true),
      pointer_contract("bf16_write_span_v1", KernelPointerAccess::kWrite, 0, 7,
                       true)};
  auto packet = KernelArgumentPacket::Create(signature(), alias_contracts);
  ASSERT_TRUE(packet.ok());
  ASSERT_TRUE(packet->set_device_pointer(
      1, pointer(0x1000, alias_contracts[0])).ok());
  EXPECT_TRUE(packet->set_device_pointer(
      2, pointer(0x1000, alias_contracts[1])).ok());

  auto stale_packet = KernelArgumentPacket::Create(signature(), alias_contracts);
  ASSERT_TRUE(stale_packet.ok());
  ASSERT_TRUE(stale_packet->set_device_pointer(
      1, pointer(0x1000, alias_contracts[0], 1)).ok());
  EXPECT_FALSE(stale_packet->set_device_pointer(
      2, pointer(0x1000, alias_contracts[1], 2)).ok());
}

TEST(KernelArgumentPacketTest, PointerContractRejectsInvalidMetadata) {
  EXPECT_FALSE(KernelPointerContract::Create(
                   "", KernelPointerAccess::kRead,
                   KernelPointerOwnerClass::kActivation, 0, 0, 16, 2, 0,
                   false)
                   .ok());
  EXPECT_FALSE(KernelPointerContract::Create(
                   "id", static_cast<KernelPointerAccess>(255),
                   KernelPointerOwnerClass::kActivation, 0, 0, 16, 2, 0,
                   false)
                   .ok());
  EXPECT_FALSE(KernelPointerContract::Create(
                   "id", KernelPointerAccess::kRead,
                   static_cast<KernelPointerOwnerClass>(255), 0, 0, 16, 2, 0,
                   false)
                   .ok());
  EXPECT_FALSE(KernelPointerContract::Create(
                   "id", KernelPointerAccess::kRead,
                   KernelPointerOwnerClass::kActivation, -1, 0, 16, 2, 0,
                   false)
                   .ok());
  EXPECT_FALSE(KernelPointerContract::Create(
                   "id", KernelPointerAccess::kRead,
                   KernelPointerOwnerClass::kActivation, 0, 0, 16, 3, 0,
                   false)
                   .ok());
  EXPECT_FALSE(KernelPointerContract::Create(
                   "id", KernelPointerAccess::kRead,
                   KernelPointerOwnerClass::kActivation, 0, 0, 16, 2, 9,
                   false)
                   .ok());
}

TEST(KernelArgumentPacketTest, ResetZeroesCellsAndRetainsStableStorage) {
  const auto pointer_contracts = contracts();
  auto packet = KernelArgumentPacket::Create(signature(), pointer_contracts);
  ASSERT_TRUE(packet.ok());
  const void* first_cell = packet->argument_cell(0);
  ASSERT_TRUE(packet->set_u32(0, 99).ok());
  packet->reset();
  EXPECT_FALSE(packet->ready());
  EXPECT_EQ(packet->argument_cell(0), first_cell);
  EXPECT_EQ(read_cell<std::uint32_t>(*packet, 0), 0);
  EXPECT_TRUE(packet->set_u32(0, 7).ok());
}

TEST(KernelArgumentPacketTest, MoveRebuildsPointersIntoOwnedCellStorage) {
  const auto pointer_contracts = contracts();
  auto created = KernelArgumentPacket::Create(signature(), pointer_contracts);
  ASSERT_TRUE(created.ok());
  KernelArgumentPacket moved = std::move(created).value();
  ASSERT_TRUE(moved.set_u32(0, 42).ok());
  EXPECT_EQ(read_cell<std::uint32_t>(moved, 0), 42);
}

TEST(KernelArgumentPacketTest, RejectsNonfiniteFloatingCells) {
  const std::string digest(64, 'c');
  const std::vector<KernelParameterSpec> parameters{
      {"epsilon", KernelWireType::kFloat32, 0, "finite_positive_epsilon_v1"}};
  auto manifest = KernelSignatureManifest::Create("rms", digest, digest, parameters, 4);
  ASSERT_TRUE(manifest.ok());
  auto packet = KernelArgumentPacket::Create(manifest.value());
  ASSERT_TRUE(packet.ok());
  EXPECT_FALSE(packet->set_float32(0, std::numeric_limits<float>::infinity()).ok());
  EXPECT_TRUE(packet->set_float32(0, 1.0e-6F).ok());
}

}  // namespace
}  // namespace pih
