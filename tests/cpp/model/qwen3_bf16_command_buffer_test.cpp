#include "pih/model/qwen3_bf16_command_buffer.h"

#include <cstddef>

#include <gtest/gtest.h>

#include "pih/core/sha256.h"

namespace pih {
namespace {

Qwen3Config official_config() {
  return Qwen3Config{1024, 3072, 28, 16, 8, 128, 151936, 40960,
                     1'000'000.0, 0.000001, 151643, 151645};
}

TEST(QwenBf16CommandBufferTest, ExpandsLogicalScheduleIntoFixedSubmissions) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  ASSERT_TRUE(schedule.ok());
  auto weights = QwenBf16WeightBindingPlan::Create(*schedule);
  ASSERT_TRUE(weights.ok());
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *weights);
  ASSERT_TRUE(commands.ok()) << commands.status().message();
  EXPECT_EQ(commands->size(), 509);

  std::size_t kernels = 0;
  std::size_t linears = 0;
  std::size_t weighted = 0;
  for (const auto& command : *commands) {
    kernels += command.backend == QwenBf16CommandBackend::kKernel;
    linears += command.backend == QwenBf16CommandBackend::kLinear;
    weighted += command.has_weight();
  }
  EXPECT_EQ(kernels, 312);
  EXPECT_EQ(linears, 197);
  EXPECT_EQ(weighted, 311);
}

TEST(QwenBf16CommandBufferTest, ExpandsRopeIntoQueryThenKeyKernels) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  ASSERT_TRUE(schedule.ok());
  auto weights = QwenBf16WeightBindingPlan::Create(*schedule);
  ASSERT_TRUE(weights.ok());
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *weights);
  ASSERT_TRUE(commands.ok());
  std::size_t rope_commands = 0;
  for (const auto& command : *commands) {
    if (command.execution_step.operation == QwenBf16ExecutionOp::kRope) {
      EXPECT_EQ(command.primitive, QwenBf16Primitive::kRope);
      EXPECT_EQ(command.subcommand, rope_commands % 2);
      ++rope_commands;
    }
  }
  EXPECT_EQ(rope_commands, 56);
}

TEST(QwenBf16CommandBufferTest, FreezesFirstAndLastBackendCommands) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  ASSERT_TRUE(schedule.ok());
  auto weights = QwenBf16WeightBindingPlan::Create(*schedule);
  ASSERT_TRUE(weights.ok());
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *weights);
  ASSERT_TRUE(commands.ok());
  EXPECT_EQ((*commands)[0].primitive, QwenBf16Primitive::kEmbedding);
  EXPECT_EQ((*commands)[1].primitive, QwenBf16Primitive::kRopeAngles);
  EXPECT_EQ((*commands)[507].linear_kind, QwenBf16LinearKind::kLmHead);
  EXPECT_EQ((*commands)[508].primitive, QwenBf16Primitive::kGreedyArgmax);
  EXPECT_EQ(commands->command_begin(0), 0);
  EXPECT_EQ(commands->command_end(0), 1);
  EXPECT_EQ(commands->command_end(480), 509);
  for (std::size_t step = 0; step < schedule->size(); ++step) {
    const auto count = commands->command_end(step) - commands->command_begin(step);
    EXPECT_EQ(count, (*schedule)[step].operation == QwenBf16ExecutionOp::kRope
                         ? 2
                         : 1);
  }
}

TEST(QwenBf16CommandBufferTest, FreezesCanonicalDispatchWire) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  ASSERT_TRUE(schedule.ok());
  auto weights = QwenBf16WeightBindingPlan::Create(*schedule);
  ASSERT_TRUE(weights.ok());
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *weights);
  ASSERT_TRUE(commands.ok());

  const auto wire = commands->canonical_wire();
  static_assert(wire.size() == 8160);
  EXPECT_EQ(wire[0], std::byte{'X'});
  EXPECT_EQ(wire[3], std::byte{'B'});
  EXPECT_EQ(wire[4], std::byte{1});
  EXPECT_EQ(wire[8], std::byte{0xfd});
  EXPECT_EQ(wire[9], std::byte{0x01});
  EXPECT_EQ(wire[12], std::byte{16});

  constexpr std::size_t kFirst = 16;
  EXPECT_EQ(wire[kFirst], std::byte{0});
  EXPECT_EQ(wire[kFirst + 1], std::byte{0});
  EXPECT_EQ(wire[kFirst + 10], std::byte{1});
  EXPECT_EQ(wire[kFirst + 11], std::byte{0});
  EXPECT_EQ(wire[kFirst + 12], std::byte{0xff});
  EXPECT_EQ(wire[kFirst + 15], std::byte{0xff});

  constexpr std::size_t kLast = 16 + 508 * 16;
  EXPECT_EQ(wire[kLast], std::byte{0});
  EXPECT_EQ(wire[kLast + 1], std::byte{21});
  EXPECT_EQ(wire[kLast + 8], std::byte{0xe0});
  EXPECT_EQ(wire[kLast + 9], std::byte{0x01});
  EXPECT_EQ(wire[kLast + 12], std::byte{0xff});
  EXPECT_EQ(wire[kLast + 15], std::byte{0xff});

  auto root = sha256(wire);
  ASSERT_TRUE(root.ok());
  EXPECT_EQ(root->hex(),
            "a29002096d0e749ab84bdbc83c5401f4585c94ccaf6031258803648cdd79b12e");
}

}  // namespace
}  // namespace pih
