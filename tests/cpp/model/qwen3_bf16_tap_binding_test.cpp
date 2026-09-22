#include <array>
#include <cstddef>

#include <gtest/gtest.h>

#include "pih/model/qwen3_bf16_tap_binding.h"

namespace pih {
namespace {

Qwen3Config config() {
  return {1024, 3072, 28, 16, 8, 128, 151936, 40960,
          1'000'000.0, 0.000001, 151643, 151645};
}

QwenBf16CommandBuffer commands() {
  auto schedule = QwenBf16ExecutionSchedule::Create(config()).value();
  auto weights = QwenBf16WeightBindingPlan::Create(schedule).value();
  return QwenBf16CommandBuffer::Create(schedule, weights).value();
}

TEST(QwenBf16TapBindingTest, ResolvesExactProducerCommandAndOutput) {
  const std::array requests{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLayerHidden, 13, 0, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kQueryAfterRope, 7, 0, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kKvValue, 27, 129, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLogits, 28, 0, 1},
  };
  auto taps = QwenNumericalTapPlan::Create(requests, 1ULL << 20).value();

  auto bindings = QwenBf16TapBindingPlan::Create(taps, commands());

  ASSERT_TRUE(bindings.ok()) << bindings.status().message();
  ASSERT_EQ(bindings->size(), 4);
  EXPECT_EQ((*bindings)[0].output_slot, QwenBf16ActivationSlot::kHidden);
  EXPECT_EQ((*bindings)[0].producer.operation,
            QwenBf16ExecutionOp::kMlpResidual);
  EXPECT_EQ((*bindings)[0].producer.layer, 13);
  EXPECT_EQ((*bindings)[1].output_slot, QwenBf16ActivationSlot::kQuery);
  EXPECT_EQ((*bindings)[1].producer_subcommand, 0);
  EXPECT_EQ((*bindings)[2].output_slot, QwenBf16ActivationSlot::kKvBacking);
  EXPECT_EQ((*bindings)[2].kv_component, QwenBf16TapKvComponent::kValue);
  EXPECT_EQ((*bindings)[2].request.position, 129);
  EXPECT_EQ((*bindings)[3].output_slot, QwenBf16ActivationSlot::kLogits);
  EXPECT_LT((*bindings)[0].producer_command_index,
            (*bindings)[3].producer_command_index);
}

TEST(QwenBf16TapBindingTest, CompleteBaselineMapsEveryCapture) {
  std::vector<QwenNumericalTapRequest> requests;
  for (const std::uint32_t layer : {0U, 13U, 27U}) {
    requests.push_back({QwenNumericalTapPoint::kLayerHidden, layer, 0, 1});
  }
  for (std::uint32_t layer = 0; layer < 28; ++layer) {
    for (const auto point : {QwenNumericalTapPoint::kQueryAfterNorm,
                             QwenNumericalTapPoint::kKeyAfterNorm,
                             QwenNumericalTapPoint::kQueryAfterRope,
                             QwenNumericalTapPoint::kKeyAfterRope,
                             QwenNumericalTapPoint::kPrefillAttention}) {
      requests.push_back({point, layer, 0, 1});
    }
    for (const std::uint32_t position : {2U, 17U, 129U, 4097U}) {
      requests.push_back({QwenNumericalTapPoint::kKvKey, layer, position, 1});
      requests.push_back({QwenNumericalTapPoint::kKvValue, layer, position, 1});
    }
  }
  requests.push_back({QwenNumericalTapPoint::kFinalNorm, 28, 0, 1});
  requests.push_back({QwenNumericalTapPoint::kLogits, 28, 0, 1});
  auto taps = QwenNumericalTapPlan::Create(requests, 1ULL << 30).value();
  ASSERT_TRUE(validate_qwen_bf16_tap_coverage(taps).ok());

  auto bindings = QwenBf16TapBindingPlan::Create(taps, commands());

  ASSERT_TRUE(bindings.ok()) << bindings.status().message();
  EXPECT_EQ(bindings->size(), 369);
  std::size_t kv_append_command = commands().size();
  const auto frozen_commands = commands();
  for (std::size_t i = 0; i < frozen_commands.size(); ++i) {
    if (frozen_commands[i].execution_step.operation ==
            QwenBf16ExecutionOp::kKvAppend &&
        frozen_commands[i].execution_step.layer == 0) {
      kv_append_command = i;
      break;
    }
  }
  ASSERT_LT(kv_append_command, frozen_commands.size());
  const auto layer_zero_kv =
      bindings->bindings_after_command(kv_append_command);
  ASSERT_EQ(layer_zero_kv.size(), 8);
  for (const auto binding_index : layer_zero_kv) {
    EXPECT_EQ((*bindings)[binding_index].producer.operation,
              QwenBf16ExecutionOp::kKvAppend);
    EXPECT_EQ((*bindings)[binding_index].producer.layer, 0);
  }
}

}  // namespace
}  // namespace pih
