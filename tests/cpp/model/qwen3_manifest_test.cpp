#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_manifest.h"

namespace pih {
namespace {

Result<SafetensorsHeader> synthetic_header(bool omit_last = false,
                                           bool drift_first = false,
                                           bool add_extra = false) {
  const auto& expected = Qwen3Manifest::expected_tensors();
  std::ostringstream json;
  json << '{';
  std::uint64_t offset = 0;
  const std::size_t count = expected.size() - (omit_last ? 1 : 0);
  for (std::size_t index = 0; index < count; ++index) {
    if (index != 0) json << ',';
    auto shape = expected[index].shape;
    if (drift_first && index == 0) ++shape[0];
    std::uint64_t elements = 1;
    for (const auto dimension : shape) elements *= dimension;
    const auto end = offset + elements * 2;
    json << '"' << expected[index].name
         << "\":{\"dtype\":\"BF16\",\"shape\":[";
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      if (axis != 0) json << ',';
      json << shape[axis];
    }
    json << "],\"data_offsets\":[" << offset << ',' << end << "]}";
    offset = end;
  }
  if (add_extra) {
    if (count != 0) json << ',';
    json << "\"unexpected.weight\":{\"dtype\":\"BF16\",\"shape\":[1],"
            "\"data_offsets\":[" << offset << ',' << offset + 2 << "]}";
    offset += 2;
  }
  json << '}';
  const auto header = json.str();
  std::vector<std::byte> prefix(8 + header.size());
  const auto header_size = static_cast<std::uint64_t>(header.size());
  for (int index = 0; index < 8; ++index) {
    prefix[index] = static_cast<std::byte>((header_size >> (index * 8)) & 0xff);
  }
  std::memcpy(prefix.data() + 8, header.data(), header.size());
  return SafetensorsHeader::ParsePrefix(prefix, prefix.size() + offset);
}

TEST(Qwen3ManifestTest, FreezesAllOfficialSourceTensorShapes) {
  const auto& expected = Qwen3Manifest::expected_tensors();
  ASSERT_EQ(expected.size(), 311);
  EXPECT_EQ(Qwen3Manifest::kOfficialSourcePayloadBytes, 1'503'264'768);
  auto header = synthetic_header();
  ASSERT_TRUE(header.ok()) << header.status().message();
  const auto status = Qwen3Manifest::Validate(header.value());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST(Qwen3ManifestTest, RejectsMissingExtraAndShapeDrift) {
  for (const auto mutation : {1, 2, 3}) {
    auto header = synthetic_header(mutation == 1, mutation == 2, mutation == 3);
    ASSERT_TRUE(header.ok()) << header.status().message();
    EXPECT_FALSE(Qwen3Manifest::Validate(header.value()).ok());
  }
}

}  // namespace
}  // namespace pih
