#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "pih/core/sha256.h"

namespace pih {
namespace {

std::span<const std::byte> bytes(std::string_view value) {
  return {reinterpret_cast<const std::byte*>(value.data()), value.size()};
}

TEST(Sha256Test, MatchesPublishedShortVectors) {
  EXPECT_EQ(sha256(bytes("")).value().hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(sha256(bytes("abc")).value().hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(
      sha256(bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))
          .value()
          .hex(),
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(Sha256Test, StreamingChunksMatchOneShotAcrossBlockBoundaries) {
  const std::string input(1000, 'a');
  const auto expected = sha256(bytes(input));
  ASSERT_TRUE(expected.ok());
  Sha256 streaming;
  for (std::size_t offset = 0; offset < input.size(); offset += 7) {
    const std::size_t count = std::min<std::size_t>(7, input.size() - offset);
    ASSERT_TRUE(streaming.update(bytes(std::string_view(input).substr(offset, count))).ok());
  }
  auto actual = streaming.finalize();
  ASSERT_TRUE(actual.ok());
  EXPECT_EQ(actual.value(), expected.value());
  EXPECT_FALSE(streaming.update(bytes("late")).ok());
  EXPECT_FALSE(streaming.finalize().ok());
}

TEST(Sha256Test, HexIsFixedWidthLowercase) {
  const auto digest = sha256(bytes("PIH"));
  ASSERT_TRUE(digest.ok());
  EXPECT_EQ(digest->hex().size(), 64);
  EXPECT_EQ(digest->hex(),
            "f61e8a71b8af36242efdbbc2b5b10649f9d0b44c4578083e2a0e4ffe1f590e02");
}

TEST(Sha256Test, ParsesOnlyCanonicalLowercaseHex) {
  constexpr std::string_view canonical =
      "f61e8a71b8af36242efdbbc2b5b10649f9d0b44c4578083e2a0e4ffe1f590e02";
  auto parsed = Sha256Digest::ParseHex(canonical);
  ASSERT_TRUE(parsed.ok());
  EXPECT_EQ(parsed->hex(), canonical);
  EXPECT_FALSE(Sha256Digest::ParseHex(canonical.substr(1)).ok());
  EXPECT_FALSE(Sha256Digest::ParseHex(
                   "F61e8a71b8af36242efdbbc2b5b10649f9d0b44c4578083e2a0e4ffe1f590e02")
                   .ok());
  EXPECT_FALSE(Sha256Digest::ParseHex(
                   "g61e8a71b8af36242efdbbc2b5b10649f9d0b44c4578083e2a0e4ffe1f590e02")
                   .ok());
}

}  // namespace
}  // namespace pih
