#include "pih/model/qwen3_bf16_resource_set.h"

#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

TensorView resource_view(std::uintptr_t address, std::int32_t rank,
                         std::uint64_t generation) {
  const std::array<std::int64_t, 1> shape{16};
  auto device = Device::Create(DeviceType::kCuda, rank);
  EXPECT_TRUE(device.ok()) << device.status().message();
  auto view = TensorView::Create(reinterpret_cast<void*>(address),
                                 DType::kUInt8, shape, {},
                                 std::move(device).value(), generation);
  EXPECT_TRUE(view.ok()) << view.status().message();
  return std::move(view).value();
}

std::vector<QwenBf16SlotResource> complete_resources(
    std::int32_t rank = 0, std::uint64_t generation = 7) {
  std::vector<QwenBf16SlotResource> resources;
  resources.reserve(QwenBf16ResourceSet::kSlotCount);
  for (std::size_t index = 0; index < QwenBf16ResourceSet::kSlotCount;
       ++index) {
    resources.push_back(
        {static_cast<QwenBf16ActivationSlot>(index),
         resource_view(0x10000 + index * 0x1000, rank, generation + index)});
  }
  return resources;
}

std::vector<TensorView> complete_weights(std::int32_t rank = 0,
                                         std::uint64_t generation = 1000) {
  std::vector<TensorView> weights;
  weights.reserve(Qwen3Manifest::kOfficialTensorCount);
  std::uintptr_t address = 0x10000000;
  for (const auto& expected : Qwen3Manifest::expected_tensors()) {
    std::vector<std::int64_t> shape;
    shape.reserve(expected.shape.size());
    for (const auto extent : expected.shape) {
      shape.push_back(static_cast<std::int64_t>(extent));
    }
    auto device = Device::Create(DeviceType::kCuda, rank);
    EXPECT_TRUE(device.ok());
    auto view = TensorView::Create(reinterpret_cast<void*>(address),
                                   DType::kBFloat16, shape, {}, *device,
                                   generation++);
    EXPECT_TRUE(view.ok()) << view.status().message();
    weights.push_back(std::move(view).value());
    address += 0x1000;
  }
  return weights;
}

TEST(QwenBf16ResourceSetTest, FreezesEveryNamedSlotExactlyOnce) {
  auto resources = complete_resources();
  auto set = QwenBf16ResourceSet::Create(41, 0, resources);
  ASSERT_TRUE(set.ok()) << set.status().message();
  EXPECT_EQ(set->request_generation(), 41);
  EXPECT_EQ(set->owning_rank(), 0);
  EXPECT_EQ(set->size(), QwenBf16ResourceSet::kSlotCount);
  for (std::size_t index = 0; index < set->size(); ++index) {
    const auto slot = static_cast<QwenBf16ActivationSlot>(index);
    auto bound = set->view(slot);
    ASSERT_TRUE(bound.ok()) << bound.status().message();
    EXPECT_EQ(bound->generation(), 7 + index);
  }
}

TEST(QwenBf16ResourceSetTest, RejectsMissingDuplicateAndUnknownSlots) {
  auto missing = complete_resources();
  missing.pop_back();
  EXPECT_FALSE(QwenBf16ResourceSet::Create(1, 0, missing).ok());

  auto duplicate = complete_resources();
  duplicate.back().slot = QwenBf16ActivationSlot::kTokenIds;
  EXPECT_FALSE(QwenBf16ResourceSet::Create(1, 0, duplicate).ok());

  auto unknown = complete_resources();
  unknown.back().slot = static_cast<QwenBf16ActivationSlot>(255);
  EXPECT_FALSE(QwenBf16ResourceSet::Create(1, 0, unknown).ok());
}

TEST(QwenBf16ResourceSetTest, RejectsGenerationAndDeviceDrift) {
  EXPECT_FALSE(QwenBf16ResourceSet::Create(0, 0, complete_resources()).ok());
  EXPECT_FALSE(QwenBf16ResourceSet::Create(1, -1, complete_resources()).ok());

  auto wrong_rank = complete_resources();
  wrong_rank[3] = {QwenBf16ActivationSlot::kNormalized,
                   resource_view(0x90000, 1, 10)};
  EXPECT_FALSE(QwenBf16ResourceSet::Create(1, 0, wrong_rank).ok());

  auto zero_generation = complete_resources();
  zero_generation[4] = {QwenBf16ActivationSlot::kQuery,
                        resource_view(0x91000, 0, 0)};
  EXPECT_FALSE(QwenBf16ResourceSet::Create(1, 0, zero_generation).ok());
}

TEST(QwenBf16WeightResourceSetTest, FreezesManifestIndexedWeightViews) {
  auto weights = complete_weights();
  auto set = QwenBf16WeightResourceSet::Create(0, weights);
  ASSERT_TRUE(set.ok()) << set.status().message();
  EXPECT_EQ(set->size(), Qwen3Manifest::kOfficialTensorCount);
  auto first = set->view(0);
  auto last = set->view(Qwen3Manifest::kOfficialTensorCount - 1);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(last.ok());
  EXPECT_EQ(first->generation(), 1000);
  EXPECT_EQ(last->generation(),
            1000 + Qwen3Manifest::kOfficialTensorCount - 1);
  EXPECT_FALSE(set->view(Qwen3Manifest::kOfficialTensorCount).ok());
}

TEST(QwenBf16WeightResourceSetTest, RejectsCardinalityShapeAndDeviceDrift) {
  auto missing = complete_weights();
  missing.pop_back();
  EXPECT_FALSE(QwenBf16WeightResourceSet::Create(0, missing).ok());

  auto wrong_shape = complete_weights();
  wrong_shape[0] = resource_view(0x80000000, 0, 1);
  EXPECT_FALSE(QwenBf16WeightResourceSet::Create(0, wrong_shape).ok());

  auto wrong_rank = complete_weights();
  const std::array<std::int64_t, 2> head_shape{151936, 1024};
  auto device = Device::Create(DeviceType::kCuda, 1);
  auto view = TensorView::Create(
                                 reinterpret_cast<void*>(
                                     static_cast<std::uintptr_t>(0x81000000)),
                                 DType::kBFloat16, head_shape, {}, *device, 1);
  ASSERT_TRUE(view.ok());
  wrong_rank[0] = *view;
  EXPECT_FALSE(QwenBf16WeightResourceSet::Create(0, wrong_rank).ok());
}

}  // namespace
}  // namespace pih
