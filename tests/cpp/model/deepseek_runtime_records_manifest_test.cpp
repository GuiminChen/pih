#include "pih/model/deepseek_runtime_records_manifest.h"

#include <string>

#include <gtest/gtest.h>

namespace pih {
namespace {

constexpr std::string_view kFixture =
    R"({"abi":"deepseek_v4_runtime_records_v1","body_sha256":"d63ff1afd8bb1ccbdef15faa38ae95cd466ba9bdba95c9733fe3089263f674e5","disposition_root":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","dspark_enabled":false,"layout_root":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","model_family":"deepseek_v4_flash_0731","record_bytes":4,"record_count":1,"record_set_root":"fe2d1b6a1de52fec92bba347e1398a583de55e8ec7ecdca75e17846c61fffe1f","records":[{"disposition_record_root":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","dtype":"U8","file_begin":100,"file_end":104,"layout_record_root":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","logical_layer":4294967295,"name":"embed.weight","namespace":"endpoint","owner_rank":0,"role":"embedding","runtime_record_root":"b79ff076ebcd1d677c70db8e76909c337691a15a627dc73fb0119d8647d6deca","shape":[4],"shard":"model-endpoint.safetensors","storage_semantics":"direct_u8_bits","target_logical_root":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","tensor_bytes":4,"transform":"identity_bytes"}],"runtime_records_root":"abab692b33865e100964a1a6cff42997d4dee606ecfebf994de6c88bb4446ea7","schema":"pih.deepseek_v4_flash_0731_runtime_records.v1","support_state":"hardware_evidence_open","world_size":1})";

DeepSeekRuntimeRecordsAuthority authority() {
  return {
      Sha256Digest::ParseHex(
          "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
          .value(),
      Sha256Digest::ParseHex(
          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
          .value(),
      Sha256Digest::ParseHex(
          "fe2d1b6a1de52fec92bba347e1398a583de55e8ec7ecdca75e17846c61fffe1f")
          .value(),
      Sha256Digest::ParseHex(
          "abab692b33865e100964a1a6cff42997d4dee606ecfebf994de6c88bb4446ea7")
          .value(),
      Sha256Digest::ParseHex(
          "d63ff1afd8bb1ccbdef15faa38ae95cd466ba9bdba95c9733fe3089263f674e5")
          .value(),
      Sha256Digest::ParseHex(
          "7497c375ee3a6935185374e392a18e49abdab761ffcf325d79191057cea5cd6c")
          .value(),
      1335,
      4,
      1,
      1,
      false};
}

TEST(DeepSeekRuntimeRecordsManifestTest,
     BindsCanonicalDirectRecordToExplicitProfileOwner) {
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  auto parsed = DeepSeekRuntimeRecordsManifest::Parse(
      kFixture, authority(), pipeline);
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  ASSERT_EQ(parsed->records().size(), 1U);
  const auto& record = parsed->records().front();
  EXPECT_EQ(record.tensor_name, "embed.weight");
  EXPECT_EQ(record.shard_name, "model-endpoint.safetensors");
  EXPECT_EQ(record.role, DeepSeekTensorRole::kEmbedding);
  EXPECT_EQ(record.owner_rank, 0U);
  EXPECT_EQ(record.logical_layer, UINT32_MAX);
  EXPECT_EQ(record.dtype, DType::kUInt8);
  EXPECT_EQ(record.storage_semantics,
            DeepSeekStorageSemantics::kDirectU8Bits);
  EXPECT_EQ(record.file_begin, 100U);
  EXPECT_EQ(record.file_end, 104U);
  EXPECT_EQ(parsed->find("embed.weight"), &record);
  EXPECT_EQ(parsed->find("head.weight"), nullptr);
}

TEST(DeepSeekRuntimeRecordsManifestTest, NativeEncoderReproducesFrozenFixture) {
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  const auto expected = authority();
  auto parsed = DeepSeekRuntimeRecordsManifest::Parse(kFixture, expected, pipeline);
  ASSERT_TRUE(parsed.ok());
  auto records = parsed->records();
  records[0].runtime_record_root = {};  // Derived field is never trusted.
  auto encoded = DeepSeekRuntimeRecordsManifest::Encode(
      records, expected.layout_root, expected.disposition_root, pipeline);
  ASSERT_TRUE(encoded.ok()) << encoded.status().message();
  EXPECT_EQ(encoded->json, kFixture);
  EXPECT_EQ(encoded->authority.object_sha256, expected.object_sha256);
  EXPECT_EQ(encoded->authority.runtime_records_root, expected.runtime_records_root);
  EXPECT_EQ(encoded->authority.object_bytes, expected.object_bytes);
}

TEST(DeepSeekRuntimeRecordsManifestTest, EncoderRejectsMalformedOrOverlappingRecords) {
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  const auto expected = authority();
  auto parsed = DeepSeekRuntimeRecordsManifest::Parse(kFixture, expected, pipeline).value();
  const auto encode = [&](const auto& records) {
    return DeepSeekRuntimeRecordsManifest::Encode(
        records, expected.layout_root, expected.disposition_root, pipeline);
  };
  auto records = parsed.records();
  records.push_back(records[0]);
  EXPECT_FALSE(encode(records).ok());
  records[1].tensor_name = "head.weight";
  records[1].role = DeepSeekTensorRole::kFinalHead;
  EXPECT_FALSE(encode(records).ok());  // Same shard byte range, different name.
  records.resize(1);
  records[0].target_logical_root = {};
  EXPECT_FALSE(encode(records).ok());
  records = parsed.records();
  records[0].shape[0] = UINT64_MAX;
  EXPECT_FALSE(encode(records).ok());
  records = parsed.records();
  records[0].storage_semantics = DeepSeekStorageSemantics::kDirectFp8E4m3Bits;
  EXPECT_FALSE(encode(records).ok());
  records = parsed.records();
  records[0].owner_rank = 1;
  EXPECT_FALSE(encode(records).ok());
  EXPECT_FALSE(DeepSeekRuntimeRecordsManifest::Encode(
      parsed.records(), {}, expected.disposition_root, pipeline).ok());
}

TEST(DeepSeekRuntimeRecordsManifestTest,
     RejectsObjectProfileCanonicalAndRecordRootDrift) {
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  auto wrong_object = authority();
  wrong_object.object_sha256 = Sha256Digest::ParseHex(
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
                                   .value();
  EXPECT_FALSE(DeepSeekRuntimeRecordsManifest::Parse(
                   kFixture, wrong_object, pipeline)
                   .ok());

  auto pp2 = DeepSeekPipelinePlan::Create(2, false).value();
  EXPECT_FALSE(
      DeepSeekRuntimeRecordsManifest::Parse(kFixture, authority(), pp2).ok());

  std::string noncanonical(kFixture);
  noncanonical.insert(1, " ");
  auto noncanonical_authority = authority();
  noncanonical_authority.object_bytes = noncanonical.size();
  noncanonical_authority.object_sha256 =
      sha256(std::as_bytes(std::span(noncanonical))).value();
  EXPECT_FALSE(DeepSeekRuntimeRecordsManifest::Parse(
                   noncanonical, noncanonical_authority, pipeline)
                   .ok());

  std::string changed(kFixture);
  const auto position = changed.find("direct_u8_bits");
  ASSERT_NE(position, std::string::npos);
  changed.replace(position, std::string_view("direct_u8_bits").size(),
                  "direct_f32_le_bits");
  auto changed_authority = authority();
  changed_authority.object_bytes = changed.size();
  changed_authority.object_sha256 =
      sha256(std::as_bytes(std::span(changed))).value();
  EXPECT_FALSE(DeepSeekRuntimeRecordsManifest::Parse(
                   changed, changed_authority, pipeline)
                   .ok());
}

}  // namespace
}  // namespace pih
