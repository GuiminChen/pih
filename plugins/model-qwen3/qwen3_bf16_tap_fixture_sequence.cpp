#include "pih/model/qwen3_bf16_tap_fixture_sequence.h"

#include <array>
#include <string_view>

namespace pih {
namespace {

constexpr std::array<std::uint32_t, 5> kFirstPositions{0, 1, 3, 18, 130};
constexpr std::array<std::uint32_t, 5> kCapturePositions{0, 2, 17, 129,
                                                        4097};
constexpr std::array<std::size_t, 5> kTokenCounts{1, 2, 15, 112, 3968};

Status update_u64(Sha256& digest, std::uint64_t value) {
  std::array<std::byte, 8> wire{};
  for (std::size_t index = 0; index < wire.size(); ++index) {
    wire[index] = static_cast<std::byte>(value >> (index * 8U));
  }
  return digest.update(wire);
}

}  // namespace

Result<QwenBf16TapFixtureSequence> QwenBf16TapFixtureSequence::Create(
    std::span<const std::int64_t> tokens,
    const QwenBf16TapSuitePlan& suite) {
  if (tokens.size() != kRequiredTokenCount ||
      suite.size() != QwenBf16TapSuitePlan::kFixtureCount) {
    return Status::InvalidArgument(
        "Qwen BF16 tap fixture sequence has an invalid manifest shape");
  }
  for (const auto token : tokens) {
    if (token < 0 || token >= kVocabularySize) {
      return Status::InvalidArgument(
          "Qwen BF16 tap fixture sequence contains an invalid token");
    }
  }
  for (std::size_t index = 0; index < suite.size(); ++index) {
    if (suite[index].first_position != kCapturePositions[index] ||
        kFirstPositions[index] + kTokenCounts[index] - 1 !=
            kCapturePositions[index]) {
      return Status::InvalidArgument(
          "Qwen BF16 tap suite does not match the fixture sequence");
    }
  }
  auto suite_digest = suite.semantic_digest();
  if (!suite_digest.ok()) return suite_digest.status();
  return QwenBf16TapFixtureSequence(
      std::vector<std::int64_t>(tokens.begin(), tokens.end()), *suite_digest);
}

QwenBf16TapFixtureInput QwenBf16TapFixtureSequence::operator[](
    std::size_t index) const noexcept {
  const auto first = kFirstPositions[index];
  return {static_cast<QwenBf16TapFixtureKind>(index), first,
          kCapturePositions[index],
          std::span<const std::int64_t>(tokens_).subspan(
              first, kTokenCounts[index])};
}

Result<Sha256Digest> QwenBf16TapFixtureSequence::semantic_digest() const {
  Sha256 digest;
  constexpr std::string_view domain =
      "pih.qwen_bf16_tap_fixture_sequence.v1";
  Status status = digest.update(std::as_bytes(std::span(domain)));
  if (status.ok()) status = digest.update(suite_digest_.bytes);
  if (status.ok()) status = update_u64(digest, tokens_.size());
  for (const auto token : tokens_) {
    if (status.ok()) status = update_u64(digest, token);
  }
  if (!status.ok()) return status;
  return digest.finalize();
}

}  // namespace pih
