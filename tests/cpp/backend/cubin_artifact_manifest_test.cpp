#include "pih/backend/cuda/cubin_artifact_manifest.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/core/sha256.h"

namespace pih {
namespace {

std::vector<std::byte> fake_cubin() {
  std::vector<std::byte> bytes(64);
  bytes[0] = std::byte{0x7f};
  bytes[1] = std::byte{'E'};
  bytes[2] = std::byte{'L'};
  bytes[3] = std::byte{'F'};
  bytes[4] = std::byte{2};
  bytes[5] = std::byte{1};
  bytes[6] = std::byte{1};
  bytes[18] = std::byte{0xbe};
  bytes[20] = std::byte{1};
  return bytes;
}

std::string manifest_json(std::string_view cubin_digest,
                          std::uint64_t cubin_bytes = 64,
                          std::string_view target = "sm_89") {
  return "{\"schema\":\"pih.cubin_artifact.v1\","
         "\"target_sm\":\"" + std::string(target) +
         "\",\"producer_toolkit\":\"13.2\","
         "\"producer_flags_sha256\":\"" + std::string(64, 'a') +
         "\",\"cubin_sha256\":\"" + std::string(cubin_digest) +
         "\",\"cubin_bytes\":" + std::to_string(cubin_bytes) + "}";
}

TEST(CubinArtifactManifestTest, ParsesExactSupportedSchema) {
  auto parsed = CubinArtifactManifest::Parse(manifest_json(std::string(64, 'b')));
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  EXPECT_EQ(parsed->target_sm(), 89);
  EXPECT_EQ(parsed->producer_toolkit(), "13.2");
  EXPECT_EQ(parsed->cubin_bytes(), 64);
  EXPECT_EQ(parsed->cubin_digest().hex(), std::string(64, 'b'));
}

TEST(CubinArtifactManifestTest, RejectsSchemaTargetHashAndFieldDrift) {
  const std::string valid_digest(64, 'b');
  EXPECT_FALSE(CubinArtifactManifest::Parse(
                   manifest_json(valid_digest, 64, "sm_80"))
                   .ok());
  EXPECT_FALSE(CubinArtifactManifest::Parse(manifest_json("abc")).ok());
  auto unknown = manifest_json(valid_digest);
  unknown.insert(unknown.size() - 1, ",\"unexpected\":true");
  EXPECT_FALSE(CubinArtifactManifest::Parse(unknown).ok());
  EXPECT_FALSE(CubinArtifactManifest::Parse(manifest_json(valid_digest, 63)).ok());
}

TEST(CubinArtifactManifestTest, LoadsOnlyExactSizedDigestMatchedCubin) {
  const auto bytes = fake_cubin();
  const auto digest = sha256(bytes);
  ASSERT_TRUE(digest.ok());
  const auto base = std::filesystem::temp_directory_path() /
                    "pih-cubin-artifact-test";
  const auto cubin_path = base.string() + ".cubin";
  const auto manifest_path = base.string() + ".json";
  {
    std::ofstream cubin(cubin_path, std::ios::binary | std::ios::trunc);
    cubin.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    std::ofstream manifest(manifest_path, std::ios::binary | std::ios::trunc);
    manifest << manifest_json(digest->hex());
  }
  auto artifact = CubinArtifactManifest::Load(manifest_path);
  ASSERT_TRUE(artifact.ok()) << artifact.status().message();
  auto verified = artifact->load_cubin(cubin_path, 1024);
  ASSERT_TRUE(verified.ok()) << verified.status().message();
  EXPECT_EQ(verified->bytes().size(), 64);

  auto wrong_size = CubinArtifactManifest::Parse(
      manifest_json(digest->hex(), bytes.size() + 1));
  ASSERT_TRUE(wrong_size.ok());
  EXPECT_FALSE(wrong_size->load_cubin(cubin_path, 1024).ok());
  EXPECT_FALSE(artifact->load_cubin(cubin_path, 63).ok());
  std::filesystem::remove(cubin_path);
  std::filesystem::remove(manifest_path);
}

}  // namespace
}  // namespace pih
