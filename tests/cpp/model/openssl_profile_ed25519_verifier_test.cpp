#include "pih/model/openssl_profile_ed25519_verifier.h"

#include <gtest/gtest.h>

#include <array>
#include <string_view>
#include <vector>

namespace pih {
namespace {

std::vector<std::byte> hex(std::string_view value) {
  std::vector<std::byte> result;
  result.reserve(value.size() / 2);
  const auto nibble = [](char character) -> std::uint8_t {
    return character >= 'a'
               ? static_cast<std::uint8_t>(character - 'a' + 10)
               : static_cast<std::uint8_t>(character - '0');
  };
  for (std::size_t index = 0; index < value.size(); index += 2) {
    result.push_back(static_cast<std::byte>(
        (nibble(value[index]) << 4U) | nibble(value[index + 1])));
  }
  return result;
}

TEST(OpenSslProfileEd25519VerifierTest, VerifiesRfc8032EmptyMessageVector) {
  const auto public_key = hex(
      "d75a980182b10ab7d54bfed3c964073a"
      "0ee172f3daa62325af021a68f707511a");
  const auto signature = hex(
      "e5564300c360ac729086e2cc806e828a"
      "84877f1eb8e5d974d873e06522490155"
      "5fb8821590a33bacc61e39701cf9b46b"
      "d25bf5f0595bbe24655141438e7a100b");
  OpenSslProfileEd25519Verifier verifier;

  EXPECT_TRUE(verifier.verify("release-key-1", public_key, {}, signature).ok());
}

TEST(OpenSslProfileEd25519VerifierTest, RejectsMutationAndInvalidBounds) {
  const auto public_key = hex(
      "d75a980182b10ab7d54bfed3c964073a"
      "0ee172f3daa62325af021a68f707511a");
  auto signature = hex(
      "e5564300c360ac729086e2cc806e828a"
      "84877f1eb8e5d974d873e06522490155"
      "5fb8821590a33bacc61e39701cf9b46b"
      "d25bf5f0595bbe24655141438e7a100b");
  OpenSslProfileEd25519Verifier verifier;

  signature[0] ^= std::byte{1};
  EXPECT_FALSE(verifier.verify("release-key-1", public_key, {}, signature).ok());
  EXPECT_FALSE(verifier.verify("release-key-1", public_key,
                               std::array{std::byte{1}}, signature).ok());
  EXPECT_FALSE(verifier.verify("", public_key, {}, signature).ok());
  EXPECT_FALSE(verifier.verify("release-key-1",
                               std::span(public_key).first(31), {}, signature)
                   .ok());
  EXPECT_FALSE(verifier.verify("release-key-1", public_key, {},
                               std::span(signature).first(63))
                   .ok());
}

}  // namespace
}  // namespace pih
