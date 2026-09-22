#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/backend/cuda/kernel_signature_manifest.h"

namespace pih {
namespace {

const std::string kDigest(64, 'a');

std::vector<KernelParameterSpec> valid_parameters() {
  return {{"elements", KernelWireType::kU32, 0, "nonzero_elements_v1"},
          {"input", KernelWireType::kDevicePointerU64, 8, "bf16_read_span_v1"},
          {"output", KernelWireType::kDevicePointerU64, 16,
           "bf16_write_span_v1"}};
}

TEST(KernelSignatureManifestTest, FreezesTypedOrderedParameterLayout) {
  const auto parameters = valid_parameters();
  auto manifest = KernelSignatureManifest::Create(
      "qwen_residual_bf16_v1", kDigest, kDigest, parameters, 24);
  ASSERT_TRUE(manifest.ok()) << manifest.status().message();
  EXPECT_EQ(manifest->logical_id(), "qwen_residual_bf16_v1");
  EXPECT_EQ(manifest->parameter_count(), 3);
  EXPECT_EQ(manifest->total_device_parameter_bytes(), 24);
  EXPECT_EQ(manifest->parameter(1).role, "input");
  EXPECT_EQ(manifest->parameter(1).wire_type,
            KernelWireType::kDevicePointerU64);
  EXPECT_EQ(kernel_wire_size(KernelWireType::kBFloat16Bits).value(), 2);
}

TEST(KernelSignatureManifestTest, RejectsIdentityAndRoleDrift) {
  auto parameters = valid_parameters();
  EXPECT_FALSE(KernelSignatureManifest::Create("", kDigest, kDigest, parameters, 24)
                   .ok());
  EXPECT_FALSE(KernelSignatureManifest::Create("id", "ABC", kDigest, parameters, 24)
                   .ok());
  EXPECT_FALSE(KernelSignatureManifest::Create("id", kDigest, std::string(64, 'G'),
                                                parameters, 24)
                   .ok());
  parameters[2].role = parameters[1].role;
  EXPECT_FALSE(KernelSignatureManifest::Create("id", kDigest, kDigest, parameters, 24)
                   .ok());
  parameters = valid_parameters();
  parameters[1].contract_id.clear();
  EXPECT_FALSE(KernelSignatureManifest::Create("id", kDigest, kDigest, parameters, 24)
                   .ok());
}

TEST(KernelSignatureManifestTest, RejectsOverlapMisalignmentAndOrderDrift) {
  auto parameters = valid_parameters();
  parameters[1].device_layout_offset = 4;
  EXPECT_FALSE(KernelSignatureManifest::Create("id", kDigest, kDigest, parameters, 24)
                   .ok());
  parameters = valid_parameters();
  parameters[2].device_layout_offset = 12;
  EXPECT_FALSE(KernelSignatureManifest::Create("id", kDigest, kDigest, parameters, 24)
                   .ok());
  parameters = valid_parameters();
  parameters[2].device_layout_offset = 4;
  EXPECT_FALSE(KernelSignatureManifest::Create("id", kDigest, kDigest, parameters, 24)
                   .ok());
}

TEST(KernelSignatureManifestTest, RejectsTruncationAndHardwareLimitOverflow) {
  const auto parameters = valid_parameters();
  EXPECT_FALSE(KernelSignatureManifest::Create("id", kDigest, kDigest, parameters, 23)
                   .ok());
  EXPECT_FALSE(KernelSignatureManifest::Create(
                   "id", kDigest, kDigest, parameters,
                   KernelSignatureManifest::kMaximumDeviceParameterBytes + 1)
                   .ok());
  const std::vector<KernelParameterSpec> empty;
  EXPECT_FALSE(KernelSignatureManifest::Create("id", kDigest, kDigest, empty, 0).ok());
}

TEST(KernelSignatureManifestTest, RejectsUnknownWireTypeAndExcessParameterCount) {
  auto parameters = valid_parameters();
  parameters[0].wire_type = static_cast<KernelWireType>(255);
  EXPECT_FALSE(KernelSignatureManifest::Create("id", kDigest, kDigest, parameters, 24)
                   .ok());

  std::vector<KernelParameterSpec> excessive;
  excessive.reserve(KernelSignatureManifest::kMaximumParameters + 1);
  for (std::size_t ordinal = 0;
       ordinal <= KernelSignatureManifest::kMaximumParameters; ++ordinal) {
    excessive.push_back({"p" + std::to_string(ordinal), KernelWireType::kU8,
                         static_cast<std::uint32_t>(ordinal), "u8_v1"});
  }
  EXPECT_FALSE(KernelSignatureManifest::Create(
                   "id", kDigest, kDigest, excessive,
                   static_cast<std::uint32_t>(excessive.size()))
                   .ok());
}

}  // namespace
}  // namespace pih
