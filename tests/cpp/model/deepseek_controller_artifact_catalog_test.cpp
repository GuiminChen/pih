#include "pih/model/deepseek_controller_artifact_catalog.h"
#include "pih/model/deepseek_rank_artifact_handoff_plan.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace pih { namespace {

void write_safetensors(const std::filesystem::path& path,
                       std::string_view json, std::string_view payload) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  const auto size = static_cast<std::uint64_t>(json.size());
  for (std::size_t index = 0; index < 8; ++index)
    output.put(static_cast<char>((size >> (index * 8U)) & 0xffU));
  output.write(json.data(), static_cast<std::streamsize>(json.size()));
  output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

void write_text(const std::filesystem::path& path, std::string_view source) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(source.data(), static_cast<std::streamsize>(source.size()));
}

constexpr std::string_view kRuntimeRecordsFixture =
    R"({"abi":"deepseek_v4_runtime_records_v1","body_sha256":"d63ff1afd8bb1ccbdef15faa38ae95cd466ba9bdba95c9733fe3089263f674e5","disposition_root":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","dspark_enabled":false,"layout_root":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","model_family":"deepseek_v4_flash_0731","record_bytes":4,"record_count":1,"record_set_root":"fe2d1b6a1de52fec92bba347e1398a583de55e8ec7ecdca75e17846c61fffe1f","records":[{"disposition_record_root":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","dtype":"U8","file_begin":100,"file_end":104,"layout_record_root":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","logical_layer":4294967295,"name":"embed.weight","namespace":"endpoint","owner_rank":0,"role":"embedding","runtime_record_root":"b79ff076ebcd1d677c70db8e76909c337691a15a627dc73fb0119d8647d6deca","shape":[4],"shard":"model-endpoint.safetensors","storage_semantics":"direct_u8_bits","target_logical_root":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","tensor_bytes":4,"transform":"identity_bytes"}],"runtime_records_root":"abab692b33865e100964a1a6cff42997d4dee606ecfebf994de6c88bb4446ea7","schema":"pih.deepseek_v4_flash_0731_runtime_records.v1","support_state":"hardware_evidence_open","world_size":1})";

constexpr std::string_view kTargetManifestFixture =
    R"({"artifact_abi":"deepseek_v4_runtime_artifact_v1","artifact_root":"47c294f6ceddd1d7142d4b77e502026ecfedf61be503645f2a29d91a3056a40f","conversion_root":"2c567fb3d79a1e34686b581ae165fbf36e40b0970bb28ef7cd30ee2c8175522f","converter_abi":"deepseek_v4_streaming_identity_converter_v1","converter_identity_root":"4444444444444444444444444444444444444444444444444444444444444444","disposition_root":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","dspark_enabled":false,"index":{"file_bytes":88,"index_object_root":"838dcffaa835f6e5ab4fbbe9a641dea37c386cbb4382939953318be9183395d6","index_root":"2222222222222222222222222222222222222222222222222222222222222222","name":"model.safetensors.index.json","object_sha256":"6f2c3d8a02bbcce1c289920af84c96ea800af046d701732c5630d9d09daefc6c"},"layout_root":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","logical_layout_root":"8888888888888888888888888888888888888888888888888888888888888888","manifest_body_sha256":"e060e0369deb8b5d32efe27af91664035c54e1760d403211e54dd3e083133b6d","model_family":"deepseek_v4_flash_0731","resources":{"maximum_copy_chunk_bytes":1048576,"maximum_output_descriptors":1,"source_descriptor_count":50},"runtime_records":{"abi":"deepseek_v4_runtime_records_v1","body_sha256":"d63ff1afd8bb1ccbdef15faa38ae95cd466ba9bdba95c9733fe3089263f674e5","file_bytes":1335,"name":"pih.runtime-records.json","object_sha256":"7497c375ee3a6935185374e392a18e49abdab761ffcf325d79191057cea5cd6c","record_bytes":4,"record_count":1,"record_set_root":"fe2d1b6a1de52fec92bba347e1398a583de55e8ec7ecdca75e17846c61fffe1f","runtime_records_object_root":"19da446e5c407d889c0e307c8b668db6a37efa68d3790984f42024006800ede2","runtime_records_root":"abab692b33865e100964a1a6cff42997d4dee606ecfebf994de6c88bb4446ea7"},"schema":"pih.deepseek_v4_flash_0731_runtime_artifact.v1","shard_count":1,"shards":[{"converted_shard_root":"6d71a7b28575c412b0155ac34dadbcc728c8a2e2933291b01f00d02e88e192fd","file_bytes":104,"name":"model-endpoint.safetensors","namespace":"endpoint","object_sha256":"d8d6ecb59e7fc40b2127abb88797499b6225c8cf9bace128e14ce3433770c103","payload_bytes":4,"shard_layout_root":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","tensor_count":1}],"source_inventory_root":"5555555555555555555555555555555555555555555555555555555555555555","source_payload_closure_root":"6666666666666666666666666666666666666666666666666666666666666666","support_state":"hardware_evidence_open","tensor_bytes":4,"tensor_count":1,"world_size":1})";

void write_target_generation_fixture(const std::filesystem::path& root) {
  std::filesystem::create_directory(root);
  std::string header =
      R"({"embed.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";
  header.resize(92, ' ');
  write_safetensors(root / "model-endpoint.safetensors", header, "abcd");
  write_text(
      root / "model.safetensors.index.json",
      R"({"metadata":{"total_size":4},"weight_map":{"embed.weight":"model-endpoint.safetensors"}})");
  write_text(root / "pih.runtime-records.json",
             kRuntimeRecordsFixture);
  write_text(root / "pih.manifest.json", kTargetManifestFixture);
}

TEST(DeepSeekControllerArtifactCatalogTest,
     OpensMasterLeasesAndDispatchesOnlyRankShards) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-controller-catalog-test";
  std::filesystem::create_directory(root);
  write_safetensors(
      root / "a.safetensors",
      R"({"embed.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})",
      "abcd");
  write_safetensors(
      root / "b.safetensors",
      R"({"head.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})",
      "wxyz");
  auto index = SafetensorsShardIndex::Parse(
      R"({"metadata":{"total_size":8},"weight_map":{"embed.weight":"a.safetensors","head.weight":"b.safetensors"}})")
                   .value();
  auto pipeline = DeepSeekPipelinePlan::Create(2, false).value();
  auto catalog = DeepSeekControllerArtifactCatalog::Open(
      root, index, pipeline, 1024,
      ArtifactImmutabilityMode::kUncalibrated);
  ASSERT_TRUE(catalog.ok()) << catalog.status().message();
  EXPECT_EQ(catalog->master_shard_count(), 2U);
  EXPECT_EQ(catalog->immutability_mode(),
            ArtifactImmutabilityMode::kUncalibrated);
  EXPECT_EQ(catalog->integrity_owner_bytes(), 0U);
  EXPECT_EQ(catalog->mapping_plan().rank(0).shards.size(), 1U);
  EXPECT_EQ(catalog->mapping_plan().rank(1).shards.size(), 1U);
  auto rank_tensors = catalog->rank_tensor_records(0);
  ASSERT_TRUE(rank_tensors.ok());
  ASSERT_EQ(rank_tensors->size(), 1U);
  EXPECT_EQ(rank_tensors->front().tensor_name, "embed.weight");
  EXPECT_EQ(rank_tensors->front().shard_name, "a.safetensors");
  EXPECT_EQ(rank_tensors->front().dtype, DType::kUInt8);
  EXPECT_EQ(rank_tensors->front().file_end - rank_tensors->front().file_begin,
            4U);
  EXPECT_FALSE(catalog->rank_tensor_records(2).ok());
  auto rank_zero = catalog->duplicate_rank_descriptors(0);
  ASSERT_TRUE(rank_zero.ok());
  ASSERT_EQ(rank_zero->size(), 1U);
  EXPECT_EQ(rank_zero->front().shard_name, "a.safetensors");
  auto inventory = DeepSeekStageMappedInventory::Create(
      catalog->mapping_plan().rank(0), std::move(*rank_zero));
  ASSERT_TRUE(inventory.ok());
  const auto* shard_header = catalog->header("a.safetensors");
  ASSERT_NE(shard_header, nullptr);
  const auto* tensor = shard_header->tensor("embed.weight");
  ASSERT_NE(tensor, nullptr);
  auto bytes = inventory->bytes("a.safetensors", tensor->file_begin, 4);
  ASSERT_TRUE(bytes.ok());
  EXPECT_EQ(std::to_integer<char>((*bytes)[2]), 'c');
  EXPECT_TRUE(catalog->poll_master_leases().ok());
  write_safetensors(
      root / "b.safetensors",
      R"({"head.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
      "q");
  EXPECT_FALSE(catalog->poll_master_leases().ok());
  std::filesystem::remove_all(root);
}

TEST(DeepSeekControllerArtifactCatalogTest, RejectsIndexPayloadSizeDrift) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-controller-catalog-drift-test";
  std::filesystem::create_directory(root);
  write_safetensors(
      root / "a.safetensors",
      R"({"head.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})",
      "abcd");
  auto index = SafetensorsShardIndex::Parse(
      R"({"metadata":{"total_size":5},"weight_map":{"head.weight":"a.safetensors"}})")
                   .value();
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  EXPECT_FALSE(DeepSeekControllerArtifactCatalog::OpenFlash0731(
                   root, index, pipeline, 1024,
                   ArtifactImmutabilityMode::kUncalibrated)
                   .ok());
  EXPECT_FALSE(DeepSeekControllerArtifactCatalog::Open(
                   root, index, pipeline, 1024,
                   ArtifactImmutabilityMode::kUncalibrated)
                   .ok());
  std::filesystem::remove_all(root);
}

TEST(DeepSeekControllerArtifactCatalogTest,
     BindsCompleteRootAuthoritativeTargetGenerationBeforeMapping) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-target-generation-catalog-test";
  std::filesystem::remove_all(root);
  write_target_generation_fixture(root);
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  const auto artifact_root = Sha256Digest::ParseHex(
      "47c294f6ceddd1d7142d4b77e502026ecfedf61be503645f2a29d91a3056a40f")
                                 .value();
  EXPECT_FALSE(
      DeepSeekControllerArtifactCatalog::
          OpenFlash0731TargetGenerationFsVerity(
              root, artifact_root, pipeline, 1024, {})
              .ok());
  const std::array duplicated_verity{
      DeepSeekTargetMemberFsVerityDigest{"pih.manifest.json",
                                         artifact_root},
      DeepSeekTargetMemberFsVerityDigest{"pih.manifest.json",
                                         artifact_root}};
  EXPECT_FALSE(
      DeepSeekControllerArtifactCatalog::
          OpenFlash0731TargetGenerationFsVerity(
              root, artifact_root, pipeline, 1024, duplicated_verity)
              .ok());
  auto catalog = DeepSeekControllerArtifactCatalog::OpenTargetGeneration(
      root, artifact_root, pipeline, 1024,
      ArtifactImmutabilityMode::kUncalibrated);
  ASSERT_TRUE(catalog.ok()) << catalog.status().message();
  EXPECT_TRUE(catalog->target_authority_bound());
  EXPECT_FALSE(catalog->production_eligible());
  EXPECT_EQ(catalog->artifact_root(), artifact_root);
  EXPECT_EQ(catalog->master_shard_count(), 1U);
  auto handoff = DeepSeekRankArtifactHandoffPlan::Compile(*catalog);
  ASSERT_TRUE(handoff.ok()) << handoff.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactHandoffPlanAbi,
            "pih_deepseek_rank_artifact_handoff_plan_v1");
  EXPECT_EQ(handoff->world_size(), 1U);
  EXPECT_EQ(handoff->artifact_root(), artifact_root);
  EXPECT_EQ(handoff->immutability_mode(),
            ArtifactImmutabilityMode::kUncalibrated);
  EXPECT_FALSE(handoff->source_catalog_production_eligible());
  EXPECT_NE(handoff->mapping_root(), Sha256Digest{});
  EXPECT_NE(handoff->rank_root(0), Sha256Digest{});
  EXPECT_NE(handoff->plan_root(), Sha256Digest{});
  const auto& rank_manifest = handoff->rank_manifest(0);
  EXPECT_EQ(rank_manifest.rank(), 0U);
  EXPECT_EQ(rank_manifest.manifest_root(), handoff->rank_root(0));
  EXPECT_EQ(rank_manifest.mapping().rank, 0U);
  ASSERT_EQ(rank_manifest.tensor_records().size(), 1U);
  EXPECT_EQ(rank_manifest.tensor_records()[0].tensor_name, "embed.weight");
  ASSERT_EQ(rank_manifest.descriptor_expectations().size(), 1U);
  EXPECT_EQ(rank_manifest.descriptor_expectations()[0].ordinal, 0U);
  EXPECT_EQ(rank_manifest.descriptor_expectations()[0].shard_name,
            "model-endpoint.safetensors");
  ASSERT_EQ(handoff->descriptors(0).size(), 1U);
  EXPECT_EQ(handoff->descriptors(0)[0].shard_name,
            "model-endpoint.safetensors");
  EXPECT_EQ(rank_manifest.descriptor_expectations()[0].identity,
            handoff->descriptors(0)[0].descriptor.identity());
  EXPECT_EQ(rank_manifest.descriptor_expectations()[0].immutability_mode,
            ArtifactImmutabilityMode::kUncalibrated);
  EXPECT_EQ(rank_manifest.descriptor_expectations()[0].enforced_digest,
            Sha256Digest{});
  auto rank_mapping_root = compile_deepseek_rank_mapping_plan_root(
      rank_manifest.mapping());
  ASSERT_TRUE(rank_mapping_root.ok())
      << rank_mapping_root.status().message();
  EXPECT_NE(*rank_mapping_root, Sha256Digest{});
  auto descriptor_handoff_root =
      compile_deepseek_rank_artifact_descriptor_handoff_root(
          rank_manifest.descriptor_expectations());
  ASSERT_TRUE(descriptor_handoff_root.ok())
      << descriptor_handoff_root.status().message();
  EXPECT_NE(*descriptor_handoff_root, Sha256Digest{});
  auto tensor_handoff_root = compile_deepseek_rank_tensor_handoff_root(
      rank_manifest.tensor_records());
  ASSERT_TRUE(tensor_handoff_root.ok())
      << tensor_handoff_root.status().message();
  auto rebuilt_rank_root =
      compile_deepseek_rank_artifact_handoff_manifest_root(
          0, rank_manifest.mapping(),
          rank_manifest.descriptor_expectations(),
          rank_manifest.tensor_records());
  ASSERT_TRUE(rebuilt_rank_root.ok())
      << rebuilt_rank_root.status().message();
  EXPECT_EQ(*rebuilt_rank_root, rank_manifest.manifest_root());
  auto wrong_rank_mapping = rank_manifest.mapping();
  wrong_rank_mapping.rank = 1;
  EXPECT_FALSE(compile_deepseek_rank_artifact_handoff_manifest_root(
                   0, wrong_rank_mapping,
                   rank_manifest.descriptor_expectations(),
                   rank_manifest.tensor_records())
                   .ok());
  EXPECT_EQ(catalog->shard_enforced_digest("model-endpoint.safetensors")
                .value(),
            Sha256Digest{});
  EXPECT_FALSE(catalog->shard_enforced_digest("missing.safetensors").ok());
  EXPECT_THROW((void)handoff->rank_root(1), std::out_of_range);
  EXPECT_THROW((void)handoff->rank_manifest(1), std::out_of_range);
  EXPECT_THROW((void)handoff->descriptors(1), std::out_of_range);
  EXPECT_EQ(catalog->mapping_plan().rank(0).owned_tensor_count, 1U);
  auto records = catalog->rank_tensor_records(0);
  ASSERT_TRUE(records.ok());
  ASSERT_EQ(records->size(), 1U);
  const auto& record = records->front();
  EXPECT_EQ(record.tensor_name, "embed.weight");
  EXPECT_EQ(record.logical_layer, UINT32_MAX);
  EXPECT_EQ(record.tensor_bytes, 4U);
  EXPECT_EQ(record.storage_semantics,
            DeepSeekStorageSemantics::kDirectU8Bits);
  EXPECT_EQ(record.file_begin, 100U);
  EXPECT_EQ(record.file_end, 104U);
  EXPECT_EQ(record.runtime_record_root.hex(),
            "b79ff076ebcd1d677c70db8e76909c337691a15a627dc73fb0119d8647d6deca");
  EXPECT_TRUE(catalog->poll_master_leases().ok());
  EXPECT_FALSE(
      DeepSeekControllerArtifactCatalog::OpenFlash0731TargetGeneration(
          root, artifact_root, pipeline, 1024,
          ArtifactImmutabilityMode::kUncalibrated)
          .ok());

  write_text(root / "model.safetensors.index.json", "{}");
  EXPECT_FALSE(catalog->poll_master_leases().ok());
  EXPECT_FALSE(DeepSeekControllerArtifactCatalog::OpenTargetGeneration(
                   root, artifact_root, pipeline, 1024,
                   ArtifactImmutabilityMode::kUncalibrated)
                   .ok());
  std::filesystem::remove_all(root);
}

TEST(DeepSeekControllerArtifactCatalogTest,
     HandoffRejectsLegacyCatalogWithoutTargetAuthority) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-legacy-handoff-rejection-test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directory(root);
  write_safetensors(
      root / "a.safetensors",
      R"({"embed.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})",
      "abcd");
  auto index = SafetensorsShardIndex::Parse(
      R"({"metadata":{"total_size":4},"weight_map":{"embed.weight":"a.safetensors"}})")
                   .value();
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  auto catalog = DeepSeekControllerArtifactCatalog::Open(
      root, index, pipeline, 1024,
      ArtifactImmutabilityMode::kUncalibrated).value();
  EXPECT_FALSE(DeepSeekRankArtifactHandoffPlan::Compile(catalog).ok());
  std::filesystem::remove_all(root);
}

Sha256Digest indexed_handoff_root(std::uint32_t index,
                                  std::uint8_t domain) {
  Sha256Digest result{};
  result.bytes[0] = static_cast<std::byte>(domain);
  for (std::size_t byte = 0; byte < sizeof(index); ++byte) {
    result.bytes[byte + 1] =
        static_cast<std::byte>(index >> (byte * 8U));
  }
  return result;
}

TEST(DeepSeekControllerArtifactCatalogTest,
     RankTensorHandoffChunksTheMaximumOfficialRecordCount) {
  std::vector<DeepSeekRankTensorRecord> records;
  records.reserve(DeepSeekRuntimeRecordsManifest::kMaximumRecordCount);
  const auto common = indexed_handoff_root(1, 80);
  for (std::uint32_t index = 0;
       index < DeepSeekRuntimeRecordsManifest::kMaximumRecordCount; ++index) {
    DeepSeekRankTensorRecord record;
    auto ordinal = std::to_string(index);
    record.tensor_name =
        "tensor." + std::string(5U - ordinal.size(), '0') + ordinal;
    record.shard_name = "model.safetensors";
    record.role = DeepSeekTensorRole::kMainLayer;
    record.dtype = DType::kUInt8;
    record.shape = {1};
    record.file_begin = index;
    record.file_end = static_cast<std::uint64_t>(index) + 1U;
    record.logical_layer = index % 43U;
    record.tensor_bytes = 1;
    record.storage_semantics = DeepSeekStorageSemantics::kDirectU8Bits;
    record.artifact_root = common;
    record.layout_root = common;
    record.disposition_root = common;
    record.target_logical_root = common;
    record.disposition_record_root = common;
    record.layout_record_root = common;
    record.runtime_record_root = indexed_handoff_root(index, 81);
    records.push_back(std::move(record));
  }
  auto root = compile_deepseek_rank_tensor_handoff_root(records);
  ASSERT_TRUE(root.ok()) << root.status().message();
  EXPECT_NE(*root, Sha256Digest{});
  std::swap(records[0], records[1]);
  EXPECT_FALSE(compile_deepseek_rank_tensor_handoff_root(records).ok());
  std::swap(records[0], records[1]);
  records.front().role = static_cast<DeepSeekTensorRole>(255);
  EXPECT_FALSE(compile_deepseek_rank_tensor_handoff_root(records).ok());
  records.front().role = DeepSeekTensorRole::kMainLayer;
  records.back().runtime_record_root = records.front().runtime_record_root;
  EXPECT_FALSE(compile_deepseek_rank_tensor_handoff_root(records).ok());
}

TEST(DeepSeekControllerArtifactCatalogTest,
     RankTensorHandoffAcceptsEveryRuntimeStoragePair) {
  const std::array pairs{
      std::pair{DType::kFloat32,
                DeepSeekStorageSemantics::kDirectF32LittleEndianBits},
      std::pair{DType::kFloat16,
                DeepSeekStorageSemantics::kDirectF16LittleEndianBits},
      std::pair{DType::kBFloat16,
                DeepSeekStorageSemantics::kDirectBf16LittleEndianBits},
      std::pair{DType::kInt8,
                DeepSeekStorageSemantics::kDirectMxfp4E2m1PackedBits},
      std::pair{DType::kUInt8, DeepSeekStorageSemantics::kDirectU8Bits},
      std::pair{DType::kFloat8E4M3,
                DeepSeekStorageSemantics::kDirectFp8E4m3Bits},
      std::pair{DType::kFloat8E8M0,
                DeepSeekStorageSemantics::kDirectUe8m0ScaleBits},
      std::pair{DType::kInt32,
                DeepSeekStorageSemantics::kDirectI32LittleEndianBits},
      std::pair{DType::kInt64,
                DeepSeekStorageSemantics::kDirectI64LittleEndianBits},
      std::pair{DType::kBool,
                DeepSeekStorageSemantics::kDirectBoolBits}};
  const auto common = indexed_handoff_root(1, 82);
  std::vector<DeepSeekRankTensorRecord> records;
  for (std::uint32_t index = 0; index < pairs.size(); ++index) {
    DeepSeekRankTensorRecord record;
    record.tensor_name = "tensor.0" + std::to_string(index);
    record.shard_name = "model.safetensors";
    record.role = DeepSeekTensorRole::kMainLayer;
    record.dtype = pairs[index].first;
    record.shape = {1};
    record.file_begin = index;
    record.file_end = index + 1U;
    record.logical_layer = 0;
    record.tensor_bytes = 1;
    record.storage_semantics = pairs[index].second;
    record.artifact_root = common;
    record.layout_root = common;
    record.disposition_root = common;
    record.target_logical_root = common;
    record.disposition_record_root = common;
    record.layout_record_root = common;
    record.runtime_record_root = indexed_handoff_root(index, 83);
    records.push_back(std::move(record));
  }
  auto root = compile_deepseek_rank_tensor_handoff_root(records);
  ASSERT_TRUE(root.ok()) << root.status().message();
  records[3].dtype = DType::kUInt8;
  EXPECT_FALSE(compile_deepseek_rank_tensor_handoff_root(records).ok());
}

} }  // namespace pih
