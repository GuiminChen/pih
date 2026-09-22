#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/safetensors_header.h"

namespace pih {
namespace {

std::vector<std::byte> prefix(std::string json) {
  std::vector<std::byte> bytes(8 + json.size());
  const auto length = static_cast<std::uint64_t>(json.size());
  for (int index = 0; index < 8; ++index) {
    bytes[index] = static_cast<std::byte>((length >> (index * 8)) & 0xFF);
  }
  std::memcpy(bytes.data() + 8, json.data(), json.size());
  return bytes;
}

TEST(SafetensorsHeaderTest, ParsesContiguousTypedTensorRecords) {
  const std::string json =
      R"({"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]},"__metadata__":{"format":"pt"}})";
  auto bytes = prefix(json);
  auto parsed = SafetensorsHeader::ParsePrefix(bytes, bytes.size() + 12);
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  ASSERT_EQ(parsed->tensors().size(), 2);
  EXPECT_EQ(parsed->data_bytes(), 12);
  EXPECT_EQ(parsed->tensor("a")->dtype, DType::kBFloat16);
  EXPECT_EQ(parsed->tensor("a")->shape, (std::vector<std::uint64_t>{2, 2}));
  EXPECT_EQ(parsed->tensor("a")->file_begin, bytes.size());
  EXPECT_EQ(parsed->tensor("b")->file_end, bytes.size() + 12);
}

TEST(SafetensorsHeaderTest, ParsesDeepSeekOpaqueFp8StorageDtypes) {
  const std::string json =
      R"({"packed_scale":{"dtype":"F8_E8M0","shape":[4],"data_offsets":[0,4]},"fp8_weight":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[4,8]}})";
  auto bytes = prefix(json);
  auto parsed = SafetensorsHeader::ParsePrefix(bytes, bytes.size() + 8);
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  ASSERT_NE(parsed->tensor("packed_scale"), nullptr);
  ASSERT_NE(parsed->tensor("fp8_weight"), nullptr);
  EXPECT_EQ(parsed->tensor("packed_scale")->dtype, DType::kFloat8E8M0);
  EXPECT_EQ(parsed->tensor("fp8_weight")->dtype, DType::kFloat8E4M3);
  EXPECT_EQ(dtype_size(DType::kFloat8E8M0).value(), 1U);
  EXPECT_EQ(dtype_size(DType::kFloat8E4M3).value(), 1U);
}

TEST(SafetensorsHeaderTest, RejectsTruncatedOrUnboundedHeaders) {
  std::vector<std::byte> short_prefix(7);
  EXPECT_FALSE(SafetensorsHeader::ParsePrefix(short_prefix, 7).ok());

  auto bytes = prefix(R"({})");
  bytes.resize(8);
  EXPECT_FALSE(SafetensorsHeader::ParsePrefix(bytes, 100).ok());

  std::vector<std::byte> huge(8);
  const std::uint64_t length = SafetensorsHeader::kMaxHeaderBytes + 1;
  for (int index = 0; index < 8; ++index) {
    huge[index] = static_cast<std::byte>((length >> (index * 8)) & 0xFF);
  }
  EXPECT_FALSE(SafetensorsHeader::ParsePrefix(huge, length + 8).ok());
}

TEST(SafetensorsHeaderTest, RejectsOverlapGapAndFileBoundaryMismatch) {
  for (const auto* json : {
           R"({"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[2,6]}})",
           R"({"a":{"dtype":"BF16","shape":[2],"data_offsets":[1,5]}})",
           R"({"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}})"}) {
    auto bytes = prefix(json);
    const std::uint64_t payload = std::string_view(json).find("[1,5]") != std::string_view::npos ? 5 : 6;
    if (std::string_view(json).find("[0,4]") != std::string_view::npos &&
        std::string_view(json).find("\"b\"") == std::string_view::npos) {
      EXPECT_FALSE(SafetensorsHeader::ParsePrefix(bytes, bytes.size() + 5).ok());
    } else {
      EXPECT_FALSE(SafetensorsHeader::ParsePrefix(bytes, bytes.size() + payload).ok());
    }
  }
}

TEST(SafetensorsHeaderTest, RejectsBadDtypeShapeOffsetsAndByteCount) {
  for (const auto* json : {
           R"({"a":{"dtype":"UNKNOWN","shape":[1],"data_offsets":[0,2]}})",
           R"({"a":{"dtype":"BF16","shape":[-1],"data_offsets":[0,2]}})",
           R"({"a":{"dtype":"BF16","shape":[2],"data_offsets":[4,0]}})",
           R"({"a":{"dtype":"BF16","shape":[3],"data_offsets":[0,4]}})",
           R"({"a":{"dtype":"BF16","shape":[9223372036854775807,2],"data_offsets":[0,4]}})"}) {
    auto bytes = prefix(json);
    EXPECT_FALSE(SafetensorsHeader::ParsePrefix(bytes, bytes.size() + 4).ok());
  }
}

TEST(SafetensorsHeaderTest, RejectsDuplicateSemanticFieldsAndExcessRank) {
  auto duplicate = prefix(
      R"({"a":{"dtype":"BF16","dtype":"BF16","shape":[1],"data_offsets":[0,2]}})");
  EXPECT_FALSE(SafetensorsHeader::ParsePrefix(duplicate, duplicate.size() + 2).ok());

  auto rank = prefix(
      R"({"a":{"dtype":"BF16","shape":[1,1,1,1,1,1,1,1,1],"data_offsets":[0,2]}})");
  EXPECT_FALSE(SafetensorsHeader::ParsePrefix(rank, rank.size() + 2).ok());
}

TEST(SafetensorsHeaderTest, RejectsMalformedRawUtf8TensorName) {
  std::string json = "{\"tensor.";
  json.push_back(static_cast<char>(0x80));
  json += "\":{\"dtype\":\"I8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
  auto bytes = prefix(json);

  EXPECT_FALSE(SafetensorsHeader::ParsePrefix(bytes, bytes.size() + 1).ok());
}

}  // namespace
}  // namespace pih
