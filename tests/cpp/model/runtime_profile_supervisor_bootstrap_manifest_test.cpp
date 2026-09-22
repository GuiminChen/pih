#include "pih/model/runtime_profile_supervisor_bootstrap_manifest.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest bootstrap_digest(std::byte value) {
  Sha256Digest digest{};
  digest.bytes.fill(value);
  return digest;
}

TEST(RuntimeProfileSupervisorBootstrapManifestTest,
     BindsAuthorityFdsGenerationAndSecretNonce) {
  auto fds = RuntimeProfileInheritedFdManifest::Create(
      {3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14}).value();
  std::array<std::byte, kRuntimeReadinessNonceBytes> nonce{};
  nonce.fill(std::byte{0x5a});
  auto manifest = RuntimeProfileSupervisorBootstrapManifest::Create(
      std::move(fds), 9, bootstrap_digest(std::byte{1}),
      bootstrap_digest(std::byte{2}), nonce);
  ASSERT_TRUE(manifest.ok());
  EXPECT_EQ(manifest->deployment_generation(), 9U);
  EXPECT_TRUE(manifest->matches_readiness_nonce(nonce));
  nonce[0] = std::byte{0x5b};
  EXPECT_FALSE(manifest->matches_readiness_nonce(nonce));
  EXPECT_FALSE(manifest->matches_readiness_nonce(
      std::span<const std::byte>(nonce).first(31)));
  EXPECT_NE(manifest->readiness_nonce_digest(), manifest->manifest_root());
}

TEST(RuntimeProfileSupervisorBootstrapManifestTest, RejectsEmptyIdentities) {
  auto make_fds = [] {
    return RuntimeProfileInheritedFdManifest::Create(
        {3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14}).value();
  };
  std::array<std::byte, kRuntimeReadinessNonceBytes> nonce{};
  EXPECT_FALSE(RuntimeProfileSupervisorBootstrapManifest::Create(
      make_fds(), 1, bootstrap_digest(std::byte{1}),
      bootstrap_digest(std::byte{2}), nonce).ok());
  nonce.fill(std::byte{1});
  EXPECT_FALSE(RuntimeProfileSupervisorBootstrapManifest::Create(
      make_fds(), 0, bootstrap_digest(std::byte{1}),
      bootstrap_digest(std::byte{2}), nonce).ok());
  EXPECT_FALSE(RuntimeProfileSupervisorBootstrapManifest::Create(
      make_fds(), 1, Sha256Digest{}, bootstrap_digest(std::byte{2}), nonce).ok());
}

}  // namespace
}  // namespace pih
