#include "pih/model/qwen3_numerical_run.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

std::string identity_json(std::string_view gpu = "RTX_4090_D",
                          std::string_view sm = "sm_89",
                          std::uint64_t generation = 7) {
  return "{\"schema\":\"pih.qwen_bf16_numerical_run.v1\","
         "\"model_sha256\":\"" + std::string(64, 'a') +
         "\",\"fixture_sha256\":\"" + std::string(64, 'b') +
         "\",\"tolerance_sha256\":\"" + std::string(64, 'c') +
         "\",\"kernel_bundle_sha256\":\"" + std::string(64, 'd') +
         "\",\"build_sha256\":\"" + std::string(64, 'e') +
         "\",\"environment_sha256\":\"" + std::string(64, 'f') +
         "\",\"target_gpu\":\"" + std::string(gpu) +
         "\",\"target_sm\":\"" + std::string(sm) +
         "\",\"run_generation\":" + std::to_string(generation) + "}";
}

TEST(QwenNumericalRunTest, ParsesExactTargetIdentity) {
  auto identity = QwenNumericalRunIdentity::Parse(identity_json());
  ASSERT_TRUE(identity.ok()) << identity.status().message();
  EXPECT_EQ(identity->target_sm(), 89);
  EXPECT_EQ(identity->target_gpu(), QwenTargetGpu::kRtx4090D);
  EXPECT_EQ(identity->run_generation(), 7);
}

TEST(QwenNumericalRunTest, RejectsGpuSmHashSchemaAndFieldDrift) {
  EXPECT_FALSE(QwenNumericalRunIdentity::Parse(
                   identity_json("RTX_4090_D", "sm_90"))
                   .ok());
  EXPECT_FALSE(QwenNumericalRunIdentity::Parse(identity_json("A100", "sm_89"))
                   .ok());
  EXPECT_FALSE(QwenNumericalRunIdentity::Parse(identity_json("H100_PCIE_80GB",
                                                              "sm_90", 0))
                   .ok());
  auto bad_hash = identity_json();
  bad_hash.replace(bad_hash.find(std::string(64, 'a')), 64, "ABC");
  EXPECT_FALSE(QwenNumericalRunIdentity::Parse(bad_hash).ok());
  auto extra = identity_json();
  extra.insert(extra.size() - 1, ",\"candidate_passed\":true");
  EXPECT_FALSE(QwenNumericalRunIdentity::Parse(extra).ok());
}

TEST(QwenNumericalRunTest, ComparisonRejectsCrossIdentitySplicing) {
  auto reference_identity =
      QwenNumericalRunIdentity::Parse(identity_json()).value();
  auto candidate_identity =
      QwenNumericalRunIdentity::Parse(identity_json("H100_PCIE_80GB", "sm_90"))
          .value();
  const float reference[] = {1.0F};
  const float candidate[] = {1.0F};
  const QwenNumericalPolicy policy{1, 0.0, 0.0, 0.0, 1.0, 0.0};
  EXPECT_FALSE(compare_qwen_numerical_run(reference_identity, reference,
                                          candidate_identity, candidate,
                                          policy)
                   .ok());
  EXPECT_TRUE(compare_qwen_numerical_run(reference_identity, reference,
                                         reference_identity, candidate, policy)
                  .ok());
}

}  // namespace
}  // namespace pih
