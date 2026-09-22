#include "../../../plugins/offline-qwen/qwen3_int4_tensor_payload.h"

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "pih/core/bfloat16.h"

namespace pih {
namespace {

TEST(QwenInt4TensorPayloadTest, DecodesLittleEndianAndSealsBothOutputs) {
  std::vector<std::byte> source(128 * 2);
  for (std::size_t index = 0; index < 128; ++index) {
    const auto bits = BFloat16::FromFloat(static_cast<float>(index % 8)).bits;
    source[index * 2] = static_cast<std::byte>(bits);
    source[index * 2 + 1] = static_cast<std::byte>(bits >> 8U);
  }
  const QwenInt4LinearRecordPlan plan{
      QwenInt4LinearShapeFamily::kQProj, "source", "values", "scales",
      1, 128, 1, 64, 2, 256, 190};
  auto payload = QwenInt4TensorPayload::Create(source, plan);
  ASSERT_TRUE(payload.ok()) << payload.status().message();
  EXPECT_EQ(payload->packed_values().size(), 64U);
  EXPECT_EQ(payload->scale_bytes_le().size(), 2U);
  EXPECT_NE(payload->packed_sha256(), Sha256Digest{});
  EXPECT_NE(payload->scales_sha256(), Sha256Digest{});
  EXPECT_EQ(payload->retained_payload_bytes(), 66U);
  EXPECT_EQ(payload->peak_anonymous_workspace_bytes(), 450U);
}

TEST(QwenInt4TensorPayloadTest, RejectsGeometryAndNonfiniteSource) {
  const QwenInt4LinearRecordPlan plan{
      QwenInt4LinearShapeFamily::kQProj, "source", "values", "scales",
      1, 128, 1, 64, 2, 256, 190};
  std::vector<std::byte> short_source(2);
  EXPECT_FALSE(QwenInt4TensorPayload::Create(short_source, plan).ok());
  std::vector<std::byte> nan_source(256, std::byte{0});
  nan_source[1] = std::byte{0x7f};
  nan_source[0] = std::byte{0xc0};
  EXPECT_FALSE(QwenInt4TensorPayload::Create(nan_source, plan).ok());
}

}  // namespace
}  // namespace pih
