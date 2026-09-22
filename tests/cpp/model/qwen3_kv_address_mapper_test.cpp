#include "pih/model/qwen3_kv_address_mapper.h"

#include <cstdint>

#include <gtest/gtest.h>

namespace pih {
namespace {

QwenKvSlotState owned(std::uint32_t generation, std::uint32_t owner,
                      std::uint16_t valid_tokens) {
  return {generation, owner, valid_tokens, QwenKvSlotLifecycle::kOwned, 0, 0};
}

TEST(QwenKvAddressMapperTest, FreezesEveryPhysicalStride) {
  auto mapper = QwenKvAddressMapper::Create(
      2, 2 * QwenKvSlotPool::kSlotPayloadBytes);
  ASSERT_TRUE(mapper.ok());
  const auto state = owned(7, 3, 16);
  auto first = mapper->map({0, 7}, state, 3, 0, QwenKvPlane::kKey, 0, 0, 0);
  auto value = mapper->map({0, 7}, state, 3, 0, QwenKvPlane::kValue, 0, 0, 0);
  auto next_layer =
      mapper->map({0, 7}, state, 3, 1, QwenKvPlane::kKey, 0, 0, 0);
  auto last = mapper->map({1, 7}, state, 3, 27, QwenKvPlane::kValue, 15, 7,
                          127);
  ASSERT_TRUE(first.ok() && value.ok() && next_layer.ok() && last.ok());
  EXPECT_EQ(first->byte_offset, 0);
  EXPECT_EQ(value->byte_offset, 32 * 1024);
  EXPECT_EQ(next_layer->byte_offset, 64 * 1024);
  EXPECT_EQ(last->byte_offset + last->byte_span,
            2 * QwenKvSlotPool::kSlotPayloadBytes);
}

TEST(QwenKvAddressMapperTest, RejectsTailStaleOwnerAndLayoutDrift) {
  auto mapper = QwenKvAddressMapper::Create(
                    1, QwenKvSlotPool::kSlotPayloadBytes)
                    .value();
  const auto state = owned(4, 9, 1);
  EXPECT_FALSE(mapper.map({0, 3}, state, 9, 0, QwenKvPlane::kKey, 0, 0, 0)
                   .ok());
  EXPECT_FALSE(mapper.map({0, 4}, state, 8, 0, QwenKvPlane::kKey, 0, 0, 0)
                   .ok());
  EXPECT_FALSE(mapper.map({0, 4}, state, 9, 0, QwenKvPlane::kKey, 1, 0, 0)
                   .ok());
  EXPECT_FALSE(mapper.map({0, 4}, state, 9, 28, QwenKvPlane::kKey, 0, 0, 0)
                   .ok());
  EXPECT_FALSE(QwenKvAddressMapper::Create(
                   2, QwenKvSlotPool::kSlotPayloadBytes)
                   .ok());
}

}  // namespace
}  // namespace pih
