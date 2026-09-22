#include "pih/model/qwen3_bf16_greedy_decode.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih {
namespace {

class Driver final : public QwenBf16GreedyStepDriver {
 public:
  explicit Driver(std::vector<std::int64_t> output)
      : output_(std::move(output)) {}

  Result<std::int64_t> execute(std::span<const std::int64_t> tokens,
                               std::uint64_t first_position) override {
    inputs.emplace_back(tokens.begin(), tokens.end());
    positions.push_back(first_position);
    if (fail) return Status::Internal("injected decode failure");
    return output_.at(inputs.size() - 1);
  }

  bool fail = false;
  std::vector<std::vector<std::int64_t>> inputs;
  std::vector<std::uint64_t> positions;

 private:
  std::vector<std::int64_t> output_;
};

TEST(QwenBf16GreedyDecodeTest, PrefillsThenDecodesOneTokenUntilEos) {
  const std::int64_t prompt[] = {11, 12, 13};
  auto decode = QwenBf16GreedyDecode::Create(prompt, 4, 99);
  ASSERT_TRUE(decode.ok());
  Driver driver({21, 22, 99});

  ASSERT_TRUE(decode->run_next(driver).ok());
  ASSERT_TRUE(decode->run_next(driver).ok());
  ASSERT_TRUE(decode->run_next(driver).ok());

  EXPECT_EQ(driver.inputs,
            (std::vector<std::vector<std::int64_t>>{{11, 12, 13}, {21}, {22}}));
  EXPECT_EQ(driver.positions, (std::vector<std::uint64_t>{0, 3, 4}));
  EXPECT_EQ(std::vector<std::int64_t>(decode->generated().begin(),
                                      decode->generated().end()),
            (std::vector<std::int64_t>{21, 22, 99}));
  EXPECT_EQ(decode->state(), QwenBf16GreedyDecodeState::kCompleted);
  EXPECT_FALSE(decode->run_next(driver).ok());
}

TEST(QwenBf16GreedyDecodeTest, StopsAtBoundWithoutEos) {
  const std::int64_t prompt[] = {1};
  auto decode = QwenBf16GreedyDecode::Create(prompt, 2, 99);
  ASSERT_TRUE(decode.ok());
  Driver driver({2, 3});
  ASSERT_TRUE(decode->run_next(driver).ok());
  ASSERT_TRUE(decode->run_next(driver).ok());
  EXPECT_EQ(decode->state(), QwenBf16GreedyDecodeState::kCompleted);
  EXPECT_EQ(decode->next_position(), 3);
}

TEST(QwenBf16GreedyDecodeTest, FailurePoisonsWithoutPublishingToken) {
  const std::int64_t prompt[] = {1};
  auto decode = QwenBf16GreedyDecode::Create(prompt, 2, 99);
  ASSERT_TRUE(decode.ok());
  Driver driver({2});
  driver.fail = true;
  EXPECT_FALSE(decode->run_next(driver).ok());
  EXPECT_TRUE(decode->generated().empty());
  EXPECT_EQ(decode->state(), QwenBf16GreedyDecodeState::kPoisoned);
  EXPECT_FALSE(decode->run_next(driver).ok());
  EXPECT_EQ(driver.inputs.size(), 1);
}

TEST(QwenBf16GreedyDecodeTest, RejectsUnboundedAndInvalidRequests) {
  const std::int64_t prompt[] = {1};
  EXPECT_FALSE(QwenBf16GreedyDecode::Create({}, 1, 99).ok());
  EXPECT_FALSE(QwenBf16GreedyDecode::Create(prompt, 0, 99).ok());
  EXPECT_FALSE(QwenBf16GreedyDecode::Create(prompt, 1, -1).ok());
  const std::int64_t invalid[] = {-1};
  EXPECT_FALSE(QwenBf16GreedyDecode::Create(invalid, 1, 99).ok());
}

}  // namespace
}  // namespace pih
