#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "pih/core/sha256.h"

namespace pih::test_fixture {

inline void write_target_safetensors(const std::filesystem::path& path,
                                     std::string_view json,
                                     std::string_view payload) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  const auto size = static_cast<std::uint64_t>(json.size());
  for (std::size_t index = 0; index < 8; ++index) {
    output.put(static_cast<char>((size >> (index * 8U)) & 0xffU));
  }
  output.write(json.data(), static_cast<std::streamsize>(json.size()));
  output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

inline void write_target_text(const std::filesystem::path& path,
                              std::string_view source) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(source.data(), static_cast<std::streamsize>(source.size()));
}

inline constexpr std::string_view kReducedRuntimeRecordsFixture =
    R"({"abi":"deepseek_v4_runtime_records_v1","body_sha256":"d63ff1afd8bb1ccbdef15faa38ae95cd466ba9bdba95c9733fe3089263f674e5","disposition_root":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","dspark_enabled":false,"layout_root":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","model_family":"deepseek_v4_flash_0731","record_bytes":4,"record_count":1,"record_set_root":"fe2d1b6a1de52fec92bba347e1398a583de55e8ec7ecdca75e17846c61fffe1f","records":[{"disposition_record_root":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","dtype":"U8","file_begin":100,"file_end":104,"layout_record_root":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","logical_layer":4294967295,"name":"embed.weight","namespace":"endpoint","owner_rank":0,"role":"embedding","runtime_record_root":"b79ff076ebcd1d677c70db8e76909c337691a15a627dc73fb0119d8647d6deca","shape":[4],"shard":"model-endpoint.safetensors","storage_semantics":"direct_u8_bits","target_logical_root":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","tensor_bytes":4,"transform":"identity_bytes"}],"runtime_records_root":"abab692b33865e100964a1a6cff42997d4dee606ecfebf994de6c88bb4446ea7","schema":"pih.deepseek_v4_flash_0731_runtime_records.v1","support_state":"hardware_evidence_open","world_size":1})";

inline constexpr std::string_view kReducedTargetManifestFixture =
    R"({"artifact_abi":"deepseek_v4_runtime_artifact_v1","artifact_root":"47c294f6ceddd1d7142d4b77e502026ecfedf61be503645f2a29d91a3056a40f","conversion_root":"2c567fb3d79a1e34686b581ae165fbf36e40b0970bb28ef7cd30ee2c8175522f","converter_abi":"deepseek_v4_streaming_identity_converter_v1","converter_identity_root":"4444444444444444444444444444444444444444444444444444444444444444","disposition_root":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","dspark_enabled":false,"index":{"file_bytes":88,"index_object_root":"838dcffaa835f6e5ab4fbbe9a641dea37c386cbb4382939953318be9183395d6","index_root":"2222222222222222222222222222222222222222222222222222222222222222","name":"model.safetensors.index.json","object_sha256":"6f2c3d8a02bbcce1c289920af84c96ea800af046d701732c5630d9d09daefc6c"},"layout_root":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","logical_layout_root":"8888888888888888888888888888888888888888888888888888888888888888","manifest_body_sha256":"e060e0369deb8b5d32efe27af91664035c54e1760d403211e54dd3e083133b6d","model_family":"deepseek_v4_flash_0731","resources":{"maximum_copy_chunk_bytes":1048576,"maximum_output_descriptors":1,"source_descriptor_count":50},"runtime_records":{"abi":"deepseek_v4_runtime_records_v1","body_sha256":"d63ff1afd8bb1ccbdef15faa38ae95cd466ba9bdba95c9733fe3089263f674e5","file_bytes":1335,"name":"pih.runtime-records.json","object_sha256":"7497c375ee3a6935185374e392a18e49abdab761ffcf325d79191057cea5cd6c","record_bytes":4,"record_count":1,"record_set_root":"fe2d1b6a1de52fec92bba347e1398a583de55e8ec7ecdca75e17846c61fffe1f","runtime_records_object_root":"19da446e5c407d889c0e307c8b668db6a37efa68d3790984f42024006800ede2","runtime_records_root":"abab692b33865e100964a1a6cff42997d4dee606ecfebf994de6c88bb4446ea7"},"schema":"pih.deepseek_v4_flash_0731_runtime_artifact.v1","shard_count":1,"shards":[{"converted_shard_root":"6d71a7b28575c412b0155ac34dadbcc728c8a2e2933291b01f00d02e88e192fd","file_bytes":104,"name":"model-endpoint.safetensors","namespace":"endpoint","object_sha256":"d8d6ecb59e7fc40b2127abb88797499b6225c8cf9bace128e14ce3433770c103","payload_bytes":4,"shard_layout_root":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","tensor_count":1}],"source_inventory_root":"5555555555555555555555555555555555555555555555555555555555555555","source_payload_closure_root":"6666666666666666666666666666666666666666666666666666666666666666","support_state":"hardware_evidence_open","tensor_bytes":4,"tensor_count":1,"world_size":1})";

inline void write_reduced_target_generation_fixture(
    const std::filesystem::path& root) {
  std::filesystem::create_directory(root);
  std::string header =
      R"({"embed.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";
  header.resize(92, ' ');
  write_target_safetensors(root / "model-endpoint.safetensors", header,
                           "abcd");
  write_target_text(
      root / "model.safetensors.index.json",
      R"({"metadata":{"total_size":4},"weight_map":{"embed.weight":"model-endpoint.safetensors"}})");
  write_target_text(root / "pih.runtime-records.json",
                    kReducedRuntimeRecordsFixture);
  write_target_text(root / "pih.manifest.json",
                    kReducedTargetManifestFixture);
}

inline Sha256Digest reduced_target_artifact_root() {
  return Sha256Digest::ParseHex(
             "47c294f6ceddd1d7142d4b77e502026ecfedf61be503645f2a29d91a3056a40f")
      .value();
}

}  // namespace pih::test_fixture
