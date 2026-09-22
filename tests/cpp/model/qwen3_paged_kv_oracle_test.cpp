#include "pih/model/qwen3_paged_kv_oracle.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_kv_address_mapper.h"

namespace pih {
namespace {

TEST(QwenPagedKvOracleTest, MaterializesCrossBlockTokenMajorCache) {
  constexpr std::uint32_t kTokens = 17;
  const std::size_t backing_elements =
      2 * QwenKvSlotPool::kSlotPayloadBytes / sizeof(BFloat16);
  std::vector<BFloat16> backing(backing_elements);
  std::vector<QwenKvSlotState> states{
      {3, 9, 16, QwenKvSlotLifecycle::kOwned, 0, 0},
      {5, 9, 1, QwenKvSlotLifecycle::kOwned, 0, 0}};
  const std::vector<QwenKvBlockHandle> handles{{0, 3}, {1, 5}};
  auto mapper = QwenKvAddressMapper::Create(
                    2, 2 * QwenKvSlotPool::kSlotPayloadBytes)
                    .value();
  for (std::uint32_t token = 0; token < kTokens; ++token) {
    const auto handle = handles[token / 16];
    const auto& state = states[handle.slot];
    const auto local = token % 16;
    for (std::uint32_t head = 0; head < 8; ++head) {
      for (std::uint32_t column = 0; column < 128; ++column) {
        const float key = static_cast<float>(token * 100 + head * 10) +
                          static_cast<float>(column) / 128.0F;
        const float value = -key;
        auto key_address = mapper.map(handle, state, 9, 4, QwenKvPlane::kKey,
                                      local, head, column).value();
        auto value_address =
            mapper.map(handle, state, 9, 4, QwenKvPlane::kValue, local, head,
                       column).value();
        backing[key_address.byte_offset / 2] = BFloat16::FromFloat(key);
        backing[value_address.byte_offset / 2] = BFloat16::FromFloat(value);
      }
    }
  }
  std::vector<BFloat16> keys(kTokens * 8 * 128);
  std::vector<BFloat16> values(keys.size());
  ASSERT_TRUE(qwen_materialize_paged_kv_oracle(
                  backing, states, handles, 9, 4, kTokens, keys, values)
                  .ok());
  const std::size_t last = keys.size() - 1;
  EXPECT_EQ(keys.front().bits, BFloat16::FromFloat(0.0F).bits);
  EXPECT_EQ(keys[last].bits, BFloat16::FromFloat(1670.9921875F).bits);
  EXPECT_EQ(values[last].bits, BFloat16::FromFloat(-1670.9921875F).bits);
}

TEST(QwenPagedKvOracleTest, RejectsStaleDuplicateAndUncommittedAtomically) {
  std::vector<BFloat16> backing(
      2 * QwenKvSlotPool::kSlotPayloadBytes / sizeof(BFloat16));
  std::vector<QwenKvSlotState> states{
      {3, 9, 16, QwenKvSlotLifecycle::kOwned, 0, 0},
      {5, 9, 0, QwenKvSlotLifecycle::kOwned, 0, 0}};
  std::vector<BFloat16> keys(17 * 8 * 128, BFloat16::FromFloat(7.0F));
  std::vector<BFloat16> values(keys);
  const auto before = keys;
  EXPECT_FALSE(qwen_materialize_paged_kv_oracle(
                   backing, states, std::vector<QwenKvBlockHandle>{{0, 3},
                                                                   {1, 4}},
                   9, 0, 17, keys, values)
                   .ok());
  EXPECT_EQ(keys, before);
  EXPECT_FALSE(qwen_materialize_paged_kv_oracle(
                   backing, states, std::vector<QwenKvBlockHandle>{{0, 3},
                                                                   {0, 3}},
                   9, 0, 17, keys, values)
                   .ok());
  EXPECT_EQ(keys, before);
}

}  // namespace
}  // namespace pih
