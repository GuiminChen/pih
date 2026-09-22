#include "pih/io/canonical_extent_verifier.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest bytes_digest(std::string_view value) {
  return sha256(std::as_bytes(std::span(value))).value();
}

TEST(CanonicalExtentVerifierTest, IndependentlyVerifiesPayloadAndPadding) {
  const std::string wire{"meta" "abc" "\0" "12345" "\0\0\0", 16};
  const std::vector<CanonicalExtentVerificationPlan> extents{
      {"a", 4, 3, 4, bytes_digest("abc")},
      {"b", 8, 5, 8, bytes_digest("12345")}};
  auto receipt = verify_canonical_extent_file(
      std::as_bytes(std::span(wire.data(), wire.size())), 4, extents);
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(receipt->file_bytes, 16U);
  EXPECT_EQ(receipt->payload_record_count, 2U);
}

TEST(CanonicalExtentVerifierTest, RejectsPayloadPaddingAndTrailingMutation) {
  std::vector<std::byte> wire(8, std::byte{0});
  std::copy_n(reinterpret_cast<const std::byte*>("abc"), 3, wire.begin() + 4);
  const std::vector<CanonicalExtentVerificationPlan> extents{
      {"a", 4, 3, 4, bytes_digest("abc")}};
  auto payload = wire; payload[4] ^= std::byte{1};
  EXPECT_FALSE(verify_canonical_extent_file(payload, 4, extents).ok());
  auto padding = wire; padding[7] = std::byte{1};
  EXPECT_FALSE(verify_canonical_extent_file(padding, 4, extents).ok());
  wire.push_back(std::byte{0});
  EXPECT_FALSE(verify_canonical_extent_file(wire, 4, extents).ok());
}

TEST(CanonicalExtentVerifierTest, RejectsNoncanonicalPlan) {
  const std::vector<std::byte> wire(8, std::byte{0});
  const std::vector<CanonicalExtentVerificationPlan> gap{
      {"a", 5, 3, 4, bytes_digest("abc")}};
  EXPECT_FALSE(verify_canonical_extent_file(wire, 4, gap).ok());
}

}  // namespace
}  // namespace pih
