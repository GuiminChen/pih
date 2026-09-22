#include "pih/model/qwen3_bf16_kv_recycler.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class ScrubDriver final : public QwenBf16KvScrubDriver {
 public:
  Status clear_and_wait(std::uintptr_t destination, std::uint64_t bytes,
                        DriverStreamHandle stream,
                        DriverEventHandle event) override {
    calls.push_back({destination, bytes, stream, event});
    return status;
  }
  struct Call { std::uintptr_t destination; std::uint64_t bytes;
                DriverStreamHandle stream; DriverEventHandle event; };
  std::vector<Call> calls;
  Status status = Status::Ok();
};

Result<QwenKvSlotPool> ready_pool() {
  auto pool = QwenKvSlotPool::Create(
      4, 4 * QwenKvSlotPool::kSlotPayloadBytes,
      4 * sizeof(QwenKvSlotState));
  if (!pool.ok()) return pool.status();
  Status ready = pool->complete_startup_sanitize(
      4 * QwenKvSlotPool::kSlotPayloadBytes,
      4 * sizeof(QwenKvSlotState), true);
  if (!ready.ok()) return ready;
  return pool;
}

TEST(QwenBf16SynchronousKvRecyclerTest, ClearsExactReleasedSlotAndReturnsCredit) {
  auto pool = ready_pool().value();
  auto handles = pool.reserve(0, 2).value();
  ASSERT_TRUE(pool.publish(0, handles).ok());
  const QwenKvCompletionEvent last_use{41, 43};
  ASSERT_TRUE(pool.release(0, handles, last_use).ok());
  ScrubDriver driver;
  constexpr std::uintptr_t kBase = 0x10000000;
  auto recycler = QwenBf16SynchronousKvRecycler::Create(
      kBase, 4 * QwenKvSlotPool::kSlotPayloadBytes, 4, 47, 53, 59,
      driver).value();
  ASSERT_TRUE(recycler.recycle(pool, handles, last_use).ok());
  ASSERT_EQ(driver.calls.size(), 2U);
  EXPECT_EQ(driver.calls[0].destination, kBase);
  EXPECT_EQ(driver.calls[1].destination,
            kBase + QwenKvSlotPool::kSlotPayloadBytes);
  EXPECT_EQ(driver.calls[0].bytes, QwenKvSlotPool::kSlotPayloadBytes);
  EXPECT_EQ(driver.calls[0].stream, 47U);
  EXPECT_EQ(driver.calls[0].event, 53U);
  EXPECT_EQ(pool.clean_credits(), 4U);
}

TEST(QwenBf16SynchronousKvRecyclerTest, ScrubFailurePoisonsPool) {
  auto pool = ready_pool().value();
  auto handles = pool.reserve(0, 1).value();
  ASSERT_TRUE(pool.publish(0, handles).ok());
  const QwenKvCompletionEvent last_use{41, 43};
  ASSERT_TRUE(pool.release(0, handles, last_use).ok());
  ScrubDriver driver;
  driver.status = Status::Internal("memset failed");
  auto recycler = QwenBf16SynchronousKvRecycler::Create(
      0x10000000, 4 * QwenKvSlotPool::kSlotPayloadBytes, 4, 47, 53, 59,
      driver).value();
  EXPECT_FALSE(recycler.recycle(pool, handles, last_use).ok());
  EXPECT_TRUE(pool.failed());
  EXPECT_EQ(pool.clean_credits(), 3U);
}

}  // namespace
}  // namespace pih
