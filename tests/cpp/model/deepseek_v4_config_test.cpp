#include "pih/model/deepseek_v4_config.h"

#include <gtest/gtest.h>

#include <string>
#include <filesystem>
#include <fstream>

namespace pih {
namespace {

const char* valid_flash_config() {
  return R"({
    "architectures":["DeepseekV4ForCausalLM"],"attention_bias":false,
    "attention_dropout":0.0,"bos_token_id":0,"eos_token_id":1,
    "expert_dtype":"fp4","hc_eps":1e-6,"hc_mult":4,"hc_sinkhorn_iters":20,
    "head_dim":512,"hidden_act":"silu","hidden_size":4096,
    "index_head_dim":128,"index_n_heads":64,"index_topk":512,
    "max_position_embeddings":1048576,"model_type":"deepseek_v4",
    "moe_intermediate_size":2048,"n_routed_experts":256,
    "n_shared_experts":1,"norm_topk_prob":true,"num_attention_heads":64,
    "num_experts_per_tok":6,"num_hidden_layers":43,"num_hash_layers":3,
    "num_key_value_heads":1,"num_nextn_predict_layers":1,"o_groups":8,
    "o_lora_rank":1024,"q_lora_rank":1024,"qk_rope_head_dim":64,
    "quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3",
      "quant_method":"fp8","scale_fmt":"ue8m0","weight_block_size":[128,128]},
    "rms_norm_eps":1e-6,"rope_scaling":{"beta_fast":32,"beta_slow":1,
      "factor":16,"original_max_position_embeddings":65536,"type":"yarn"},
    "rope_theta":10000,"routed_scaling_factor":1.5,
    "scoring_func":"sqrtsoftplus","sliding_window":128,"swiglu_limit":10.0,
    "tie_word_embeddings":false,"topk_method":"noaux_tc",
    "torch_dtype":"bfloat16","use_cache":true,"vocab_size":129280,
    "compress_rope_theta":160000,
    "compress_ratios":[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0],
    "dspark_block_size":5,"dspark_noise_token_id":128799,
    "dspark_target_layer_ids":[40,41,42],"dspark_markov_rank":256
  })";
}

const char* valid_flash_inference_config() {
  return R"({"n_mtp_layers":3})";
}

TEST(DeepSeekV4ConfigTest, FreezesPublishedFlashGeometry) {
  const auto config = DeepSeekV4Config::Flash0731();
  EXPECT_EQ(config.main_layers, 43U);
  EXPECT_EQ(config.nextn_predict_layers, 1U);
  EXPECT_EQ(config.hidden_size, 4096U);
  EXPECT_EQ(config.hc_streams, 4U);
  EXPECT_EQ(config.routed_experts, 256U);
  EXPECT_EQ(config.activated_experts, 6U);
  EXPECT_EQ(config.vocabulary_size, 129280U);
  EXPECT_DOUBLE_EQ(config.rms_norm_epsilon, 0.000001);
  EXPECT_TRUE(config.validate().ok());
}

TEST(DeepSeekV4ConfigTest, ParsesPinnedTopLevelModelConfigSemantics) {
  auto config = DeepSeekV4Config::ParseFlash0731(valid_flash_config());
  ASSERT_TRUE(config.ok()) << config.status().message();
  EXPECT_EQ(config->main_layers, 43U);
  EXPECT_EQ(config->nextn_predict_layers, 1U);
  EXPECT_EQ(config->maximum_positions, 1048576U);

  for (const auto& replacement : {
           std::pair{"\"expert_dtype\":\"fp4\"",
                     "\"expert_dtype\":\"fp8\""},
           std::pair{"\"num_hash_layers\":3", "\"num_hash_layers\":2"},
           std::pair{"\"weight_block_size\":[128,128]",
                     "\"weight_block_size\":[64,128]"},
           std::pair{"\"dspark_target_layer_ids\":[40,41,42]",
                     "\"dspark_target_layer_ids\":[39,41,42]"}}) {
    std::string changed = valid_flash_config();
    const auto offset = changed.find(replacement.first);
    ASSERT_NE(offset, std::string::npos);
    changed.replace(offset, std::string(replacement.first).size(),
                    replacement.second);
    EXPECT_FALSE(DeepSeekV4Config::ParseFlash0731(changed).ok());
  }

  // vLLM routes this optional checkpoint override to a Blackwell-only
  // NVFP4/DeepGEMM implementation.  PIH V1 supports SM89/SM90 only,
  // so the Flash-0731 loader must reject it before any device allocation.
  std::string blackwell_only = valid_flash_config();
  constexpr std::string_view kQuantizationTail =
      "\"weight_block_size\":[128,128]}";
  const auto quantization_tail = blackwell_only.find(kQuantizationTail);
  ASSERT_NE(quantization_tail, std::string::npos);
  const auto quantization_end =
      quantization_tail + kQuantizationTail.size() - 1;
  blackwell_only.insert(quantization_end, ",\"moe_quant_algo\":\"NVFP4\"");
  EXPECT_FALSE(DeepSeekV4Config::ParseFlash0731(blackwell_only).ok());
}

TEST(DeepSeekV4ConfigTest, RejectsWrongTopLevelPredictionDepth) {
  auto wrong_prediction_depth = DeepSeekV4Config::Flash0731();
  wrong_prediction_depth.nextn_predict_layers = 3;
  EXPECT_FALSE(wrong_prediction_depth.validate().ok());
}

TEST(DeepSeekV4ConfigTest, ParsesReferenceInferenceMtpStagesSeparately) {
  auto inference =
      DeepSeekV4InferenceConfig::ParseFlash0731(valid_flash_inference_config());
  ASSERT_TRUE(inference.ok()) << inference.status().message();
  EXPECT_EQ(inference->mtp_stages, 3U);
  EXPECT_TRUE(inference->validate().ok());

  for (const auto* invalid : {
           R"({})", R"({"n_mtp_layers":1})",
           R"({"n_mtp_layers":2})", R"({"n_mtp_layers":4})",
           R"({"n_mtp_layers":3.0})", R"({"n_mtp_layers":true})",
           R"({"n_mtp_layers":3,"n_mtp_layers":3})"}) {
    EXPECT_FALSE(DeepSeekV4InferenceConfig::ParseFlash0731(invalid).ok());
  }
}

TEST(DeepSeekV4ConfigTest, RejectsMissingNestedAndOversizedConfig) {
  std::string missing = valid_flash_config();
  const auto offset = missing.find("\"type\":\"yarn\"");
  missing.replace(offset, std::string("\"type\":\"yarn\"").size(),
                  "\"type\":\"linear\"");
  EXPECT_FALSE(DeepSeekV4Config::ParseFlash0731(missing).ok());
  EXPECT_FALSE(DeepSeekV4Config::ParseFlash0731(
      std::string(DeepSeekV4Config::kMaxConfigBytes + 1, ' ')).ok());
}

TEST(DeepSeekV4ConfigTest, FileReceiptBindsExactOfficialConfigBytes) {
  const auto path = std::filesystem::temp_directory_path() /
                    "pih-deepseek-v4-flash-0731-config.json";
  const std::string source = valid_flash_config();
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output);
    output.write(source.data(), static_cast<std::streamsize>(source.size()));
  }
  auto receipt = load_deepseek_v4_flash_0731_config_file(path);
  std::filesystem::remove(path);
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(receipt->file_bytes, source.size());
  EXPECT_EQ(receipt->config.main_layers, 43U);
  EXPECT_EQ(receipt->file_sha256,
            sha256(std::as_bytes(std::span(source))).value());
  EXPECT_FALSE(load_deepseek_v4_flash_0731_config_file(path).ok());
}

TEST(DeepSeekV4ConfigTest, InferenceFileReceiptBindsIndependentAuthority) {
  const auto path = std::filesystem::temp_directory_path() /
                    "pih-deepseek-v4-flash-0731-inference-config.json";
  const std::string source = valid_flash_inference_config();
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output);
    output.write(source.data(), static_cast<std::streamsize>(source.size()));
  }
  auto receipt = load_deepseek_v4_flash_0731_inference_config_file(path);
  std::filesystem::remove(path);
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(receipt->config.mtp_stages, 3U);
  EXPECT_EQ(receipt->file_bytes, source.size());
  EXPECT_EQ(receipt->file_sha256,
            sha256(std::as_bytes(std::span(source))).value());
  EXPECT_FALSE(
      load_deepseek_v4_flash_0731_inference_config_file(path).ok());
}

TEST(DeepSeekV4ConfigTest, SelectsExactContinuousPipelineRanges) {
  struct Expected {
    std::uint32_t world;
    bool dspark;
    std::vector<DeepSeekStageRange> ranges;
  };
  const std::vector<Expected> cases = {
      {1, false, {{0, 42}}},
      {2, true, {{0, 22}, {23, 42}}},
      {3, true, {{0, 14}, {15, 30}, {31, 42}}},
      {4, true, {{0, 10}, {11, 22}, {23, 34}, {35, 42}}},
      {2, false, {{0, 21}, {22, 42}}},
      {3, false, {{0, 13}, {14, 28}, {29, 42}}},
      {4, false, {{0, 10}, {11, 21}, {22, 32}, {33, 42}}},
  };
  for (const auto& expected : cases) {
    auto plan = DeepSeekPipelinePlan::Create(expected.world, expected.dspark);
    ASSERT_TRUE(plan.ok()) << plan.status().message();
    EXPECT_EQ(plan->ranges(), expected.ranges);
    EXPECT_EQ(plan->rank(expected.world - 1).owns_dspark, expected.dspark);
    EXPECT_TRUE(plan->rank(0).owns_embedding);
    EXPECT_TRUE(plan->rank(expected.world - 1).owns_lm_head);
  }
  EXPECT_FALSE(DeepSeekPipelinePlan::Create(0, false).ok());
  EXPECT_FALSE(DeepSeekPipelinePlan::Create(5, true).ok());
}

}  // namespace
}  // namespace pih
