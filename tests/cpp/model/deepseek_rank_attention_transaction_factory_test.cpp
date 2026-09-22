#include "pih/model/deepseek_rank_attention_transaction_factory.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class FactoryAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("factory allocation");
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

class FactoryFixedOperations final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

TEST(DeepSeekRankAttentionTransactionFactoryTest,
     BindsUniqueSequencesToPreallocatedStateSlots) {
  FactoryAllocator allocator;
  FactoryFixedOperations operations;
  auto pool = DeepSeekRankAttentionStatePool::Allocate(
      allocator, {0, {2, 3}, false, false, false}, 2, 256, 19,
      operations, 17, 0).value();
  DeepSeekRankAttentionTransactionFactory factory(pool);
  const std::array<DeepSeekAttentionSequenceBinding, 2> bindings{{
      {41, 1}, {42, 0}}};

  auto transactions = factory.Create(bindings);
  ASSERT_TRUE(transactions.ok()) << transactions.status().message();
  ASSERT_EQ(transactions->size(), 2U);
  EXPECT_TRUE((*transactions)[0]->begin(7).ok());
  EXPECT_TRUE((*transactions)[0]->reserve_ratio4_append(2, 0).ok());
  EXPECT_TRUE((*transactions)[0]->reserve_ratio128_append(3, 0).ok());
  EXPECT_FALSE((*transactions)[0]->reserve_ratio4_append(2, 1).ok());
  EXPECT_NE((*transactions)[0].get(), (*transactions)[1].get());
}

TEST(DeepSeekRankAttentionTransactionFactoryTest,
     RejectsDuplicateOrOutOfRangeBindingsBeforeMutation) {
  FactoryAllocator allocator;
  FactoryFixedOperations operations;
  auto pool = DeepSeekRankAttentionStatePool::Allocate(
      allocator, {0, {2, 3}, false, false, false}, 2, 256, 19,
      operations, 17, 0).value();
  DeepSeekRankAttentionTransactionFactory factory(pool);
  const std::array duplicate_sequence{
      DeepSeekAttentionSequenceBinding{41, 0},
      DeepSeekAttentionSequenceBinding{41, 1}};
  const std::array duplicate_slot{
      DeepSeekAttentionSequenceBinding{41, 0},
      DeepSeekAttentionSequenceBinding{42, 0}};
  const std::array invalid_slot{DeepSeekAttentionSequenceBinding{41, 2}};
  const std::array invalid_sequence{DeepSeekAttentionSequenceBinding{0, 0}};

  EXPECT_FALSE(factory.Create(duplicate_sequence).ok());
  EXPECT_FALSE(factory.Create(duplicate_slot).ok());
  EXPECT_FALSE(factory.Create(invalid_slot).ok());
  EXPECT_FALSE(factory.Create(invalid_sequence).ok());
  EXPECT_FALSE(factory.Create({}).ok());
  EXPECT_EQ(pool.ratio4_pool().free_pairs(), 4U);
  EXPECT_EQ(pool.ratio128_pool().free_pages(), 4U);
}

}  // namespace
}  // namespace pih
