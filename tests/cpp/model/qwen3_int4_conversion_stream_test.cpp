#include "../../../plugins/offline-qwen/qwen3_int4_conversion_stream.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/core/bfloat16.h"

namespace pih {
namespace {

class TestTensorSource final : public QwenInt4ConversionTensorSource {
 public:
  std::map<std::string, std::vector<std::byte>> tensors;
  std::map<std::string, std::size_t> reads;
  Result<std::span<const std::byte>> tensor(std::string_view name) override {
    auto found = tensors.find(std::string(name));
    if (found == tensors.end()) return Status::InvalidArgument("missing tensor");
    ++reads[found->first];
    return std::span<const std::byte>(found->second);
  }
};

std::vector<std::byte> bf16_values() {
  std::vector<std::byte> bytes(256);
  for (std::size_t index = 0; index < 128; ++index) {
    const auto bits = BFloat16::FromFloat(static_cast<float>(index % 8)).bits;
    bytes[index * 2] = static_cast<std::byte>(bits);
    bytes[index * 2 + 1] = static_cast<std::byte>(bits >> 8U);
  }
  return bytes;
}

QwenInt4ConversionStream make_stream(TestTensorSource& source) {
  source.tensors["embed"] = {std::byte{1}, std::byte{2}};
  source.tensors["linear"] = bf16_values();
  std::vector<QwenInt4ArtifactRecordPlan> artifacts{
      {"embed", QwenInt4ArtifactRecordKind::kBf16Embedding, "embed", {}, 2, 0, 4},
      {"linear.packed", QwenInt4ArtifactRecordKind::kPackedValues, "linear", {}, 64, 4, 64},
      {"linear.scales", QwenInt4ArtifactRecordKind::kFp16Scales, "linear", {}, 2, 68, 4}};
  std::vector<QwenInt4LinearRecordPlan> linears{{
      QwenInt4LinearShapeFamily::kQProj, "linear", "linear.packed",
      "linear.scales", 1, 128, 1, 64, 2, 256, 190}};
  return QwenInt4ConversionStream::Create(
      std::move(artifacts), std::move(linears), source).value();
}

TEST(QwenInt4ConversionStreamTest, ScansEveryPayloadWithOneLinearAtATime) {
  TestTensorSource source;
  auto stream = make_stream(source);
  auto receipt = stream.scan_payload_digests();
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(receipt->payload_digests.size(), 3U);
  EXPECT_EQ(receipt->peak_anonymous_workspace_bytes, 450U);
  EXPECT_EQ(source.reads["linear"], 1U);
  EXPECT_EQ(source.reads["embed"], 1U);
}

TEST(QwenInt4ConversionStreamTest, ReusesLinearForValuesAndScalesDuringWritePass) {
  TestTensorSource source;
  auto stream = make_stream(source);
  std::vector<std::byte> packed(64);
  std::vector<std::byte> scales(2);
  ASSERT_EQ(stream.read("linear.packed", 0, packed).value(), 64U);
  ASSERT_EQ(stream.read("linear.scales", 0, scales).value(), 2U);
  EXPECT_EQ(source.reads["linear"], 1U);
  std::vector<std::byte> embedding(2);
  ASSERT_EQ(stream.read("embed", 0, embedding).value(), 2U);
  EXPECT_EQ(embedding[0], std::byte{1});
}

TEST(QwenInt4ConversionStreamTest, RejectsUnknownRangeAndMalformedInventory) {
  TestTensorSource source;
  auto stream = make_stream(source);
  std::vector<std::byte> output(8);
  EXPECT_FALSE(stream.read("unknown", 0, output).ok());
  EXPECT_FALSE(stream.read("embed", 3, output).ok());
  source.tensors["linear"].pop_back();
  EXPECT_FALSE(stream.read("linear.packed", 0, output).ok());
}

}  // namespace
}  // namespace pih
