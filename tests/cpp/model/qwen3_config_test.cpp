#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "pih/model/qwen3_config.h"

namespace pih {
namespace {

const char* valid_config() {
  return R"({
    "model_type":"qwen3",
    "architectures":["Qwen3ForCausalLM"],
    "torch_dtype":"bfloat16",
    "hidden_size":1024,
    "intermediate_size":3072,
    "num_hidden_layers":28,
    "num_attention_heads":16,
    "num_key_value_heads":8,
    "head_dim":128,
    "vocab_size":151936,
    "max_position_embeddings":40960,
    "rope_theta":1000000.0,
    "rms_norm_eps":0.000001,
    "hidden_act":"silu",
    "tie_word_embeddings":true,
    "use_cache":true,
    "bos_token_id":151643,
    "eos_token_id":151645
  })";
}

TEST(Qwen3ConfigTest, AcceptsOnlyTheFrozenOfficialArchitecture) {
  auto config = Qwen3Config::Parse(valid_config());
  ASSERT_TRUE(config.ok()) << config.status().message();
  EXPECT_EQ(config->hidden_size, 1024);
  EXPECT_EQ(config->q_projection_size(), 2048);
  EXPECT_EQ(config->kv_projection_size(), 1024);
  EXPECT_EQ(config->layers, 28);
  EXPECT_EQ(config->vocabulary_size, 151936);
  EXPECT_EQ(config->bos_token_id, 151643);
  EXPECT_EQ(config->eos_token_id, 151645);
}

TEST(Qwen3ConfigTest, RejectsMissingWrongTypeAndArchitectureDrift) {
  std::string missing = valid_config();
  const auto field = missing.find("\"head_dim\":128,");
  missing.erase(field, std::string("\"head_dim\":128,").size());
  EXPECT_FALSE(Qwen3Config::Parse(missing).ok());

  std::string wrong_type = valid_config();
  wrong_type.replace(wrong_type.find("\"hidden_size\":1024"),
                     std::string("\"hidden_size\":1024").size(),
                     "\"hidden_size\":\"1024\"");
  EXPECT_FALSE(Qwen3Config::Parse(wrong_type).ok());

  std::string drift = valid_config();
  drift.replace(drift.find("\"num_hidden_layers\":28"),
                std::string("\"num_hidden_layers\":28").size(),
                "\"num_hidden_layers\":29");
  EXPECT_FALSE(Qwen3Config::Parse(drift).ok());
}

TEST(Qwen3ConfigTest, RejectsSemanticMismatchesAndOversizedInput) {
  using Replacement = std::pair<std::string_view, std::string_view>;
  for (const auto& replacement : {
           Replacement{"\"tie_word_embeddings\":true", "\"tie_word_embeddings\":false"},
           Replacement{"\"torch_dtype\":\"bfloat16\"", "\"torch_dtype\":\"float16\""},
           Replacement{"\"hidden_act\":\"silu\"", "\"hidden_act\":\"gelu\""},
           Replacement{"\"rope_theta\":1000000.0", "\"rope_theta\":10000.0"},
           Replacement{"\"model_type\":\"qwen3\"", "\"model_type\":\"qwen2\""}}) {
    std::string json = valid_config();
    json.replace(json.find(replacement.first), replacement.first.size(),
                 replacement.second);
    EXPECT_FALSE(Qwen3Config::Parse(json).ok());
  }
  EXPECT_FALSE(Qwen3Config::Parse(std::string(Qwen3Config::kMaxConfigBytes + 1, ' ')).ok());
}

}  // namespace
}  // namespace pih
