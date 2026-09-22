#include "pih/model/safetensors_shard_index.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace pih { namespace {

TEST(SafetensorsShardIndexTest, ParsesAndCanonicalizesRepositoryInventory) {
  auto index = SafetensorsShardIndex::Parse(R"({
    "metadata":{"total_size":166878536440},
    "weight_map":{
      "model.layers.1.weight":"model-00002-of-00048.safetensors",
      "model.embed_tokens.weight":"model-00001-of-00048.safetensors",
      "model.layers.0.weight":"model-00002-of-00048.safetensors"
    }})");
  ASSERT_TRUE(index.ok()) << index.status().message();
  EXPECT_EQ(index->total_size(), 166878536440ULL);
  ASSERT_EQ(index->bindings().size(), 3U);
  EXPECT_EQ(index->bindings()[0].tensor_name, "model.embed_tokens.weight");
  EXPECT_EQ(index->shard_names(),
            (std::vector<std::string>{"model-00001-of-00048.safetensors",
                                      "model-00002-of-00048.safetensors"}));
}

TEST(SafetensorsShardIndexTest, RejectsTraversalWrongTypesAndDuplicates) {
  for (const auto* source : {
      R"({"metadata":{"total_size":1},"weight_map":{"a":"../a.safetensors"}})",
      R"({"metadata":{"total_size":1},"weight_map":{"a":"a.bin"}})",
      R"({"metadata":{"total_size":0},"weight_map":{"a":"a.safetensors"}})",
      R"({"metadata":{"total_size":1},"weight_map":{"a":7}})",
      R"({"metadata":{"total_size":1},"weight_map":{"a":"a.safetensors","a":"b.safetensors"}})"}) {
    EXPECT_FALSE(SafetensorsShardIndex::Parse(source).ok());
  }
}

TEST(SafetensorsShardIndexTest, EnforcesInputAndNameBudgets) {
  EXPECT_FALSE(SafetensorsShardIndex::Parse(
      std::string(SafetensorsShardIndex::kMaximumIndexBytes + 1, ' ')).ok());
  const std::string long_name(
      SafetensorsShardIndex::kMaximumTensorNameBytes + 1, 'x');
  const auto source = std::string(
      "{\"metadata\":{\"total_size\":1},\"weight_map\":{\"") +
      long_name + "\":\"a.safetensors\"}}";
  EXPECT_FALSE(SafetensorsShardIndex::Parse(source).ok());
}

TEST(SafetensorsShardIndexTest, RejectsMalformedRawUtf8TensorName) {
  std::string source =
      "{\"metadata\":{\"total_size\":1},\"weight_map\":{\"tensor.";
  source.push_back(static_cast<char>(0x80));
  source += "\":\"model-00001-of-00048.safetensors\"}}";

  EXPECT_FALSE(SafetensorsShardIndex::Parse(source).ok());
}

TEST(SafetensorsShardIndexTest, ParsesOfficialDeepSeekRepositoryScale) {
  std::string source =
      "{\"metadata\":{\"total_size\":166878536440},\"weight_map\":{";
  source.reserve(5 * 1024 * 1024);
  for (std::size_t index = 0; index < 72'317; ++index) {
    if (index != 0) source.push_back(',');
    char shard[40]{};
    std::snprintf(shard, sizeof(shard),
                  "model-%05zu-of-00048.safetensors", index % 48 + 1);
    source += "\"tensor." + std::to_string(index) + "\":\"" + shard + "\"";
  }
  source += "}}";
  auto index = SafetensorsShardIndex::Parse(source);
  ASSERT_TRUE(index.ok()) << index.status().message();
  EXPECT_TRUE(index->validate_deepseek_flash_0731_repository_geometry().ok());
  EXPECT_EQ(index->bindings().size(), 72'317U);
  EXPECT_EQ(index->shard_names().size(), 48U);
}

TEST(SafetensorsShardIndexTest, FileReceiptBindsExactParsedBytes) {
  const auto path = std::filesystem::temp_directory_path() /
                    "pih-safetensors-shard-index.json";
  const std::string source =
      R"({"metadata":{"total_size":7},"weight_map":{"a":"a.safetensors"}})";
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output);
    output.write(source.data(), static_cast<std::streamsize>(source.size()));
  }
  auto receipt = load_safetensors_shard_index_file(path);
  std::filesystem::remove(path);
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(receipt->file_bytes, source.size());
  EXPECT_EQ(receipt->index.total_size(), 7U);
  EXPECT_EQ(receipt->file_sha256,
            sha256(std::as_bytes(std::span(source))).value());
  EXPECT_FALSE(load_safetensors_shard_index_file(path).ok());
}

} }  // namespace pih
