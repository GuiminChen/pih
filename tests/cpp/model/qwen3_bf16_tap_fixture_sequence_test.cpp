#include <algorithm>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_bf16_tap_fixture_sequence.h"

namespace pih {
namespace {

std::vector<std::int64_t> tokens() {
  std::vector<std::int64_t> result(
      QwenBf16TapFixtureSequence::kRequiredTokenCount);
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<std::int64_t>(index % 151936);
  }
  return result;
}

TEST(QwenBf16TapFixtureSequenceTest, PartitionsOneManifestWithoutGaps) {
  const auto manifest = tokens();
  const auto suite = QwenBf16TapSuitePlan::Create().value();
  auto sequence = QwenBf16TapFixtureSequence::Create(manifest, suite);

  ASSERT_TRUE(sequence.ok()) << sequence.status().message();
  ASSERT_EQ(sequence->size(), 5);
  constexpr std::uint32_t first_positions[]{0, 1, 3, 18, 130};
  constexpr std::size_t token_counts[]{1, 2, 15, 112, 3968};
  constexpr std::uint32_t capture_positions[]{0, 2, 17, 129, 4097};
  std::size_t consumed = 0;
  for (std::size_t index = 0; index < sequence->size(); ++index) {
    const auto fixture = (*sequence)[index];
    EXPECT_EQ(fixture.first_position, first_positions[index]);
    EXPECT_EQ(fixture.tokens.size(), token_counts[index]);
    EXPECT_EQ(fixture.capture_position, capture_positions[index]);
    EXPECT_EQ(fixture.tokens.data(), sequence->tokens().data() + consumed);
    EXPECT_EQ(fixture.tokens.back(), manifest[capture_positions[index]]);
    consumed += fixture.tokens.size();
  }
  EXPECT_EQ(consumed, manifest.size());
  EXPECT_TRUE(sequence->semantic_digest().ok());
}

TEST(QwenBf16TapFixtureSequenceTest, RejectsUnsealedOrInvalidManifest) {
  const auto suite = QwenBf16TapSuitePlan::Create().value();
  auto manifest = tokens();
  manifest.pop_back();
  EXPECT_FALSE(QwenBf16TapFixtureSequence::Create(manifest, suite).ok());

  manifest = tokens();
  manifest[17] = 151936;
  EXPECT_FALSE(QwenBf16TapFixtureSequence::Create(manifest, suite).ok());
  manifest[17] = -1;
  EXPECT_FALSE(QwenBf16TapFixtureSequence::Create(manifest, suite).ok());
}

TEST(QwenBf16TapFixtureSequenceTest, DigestBindsEveryToken) {
  const auto suite = QwenBf16TapSuitePlan::Create().value();
  auto first = tokens();
  auto second = first;
  ++second[4096];
  const auto a = QwenBf16TapFixtureSequence::Create(first, suite).value();
  const auto b = QwenBf16TapFixtureSequence::Create(second, suite).value();
  EXPECT_NE(a.semantic_digest().value(), b.semantic_digest().value());
}

}  // namespace
}  // namespace pih
