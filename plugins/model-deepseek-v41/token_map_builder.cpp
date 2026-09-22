#include "token_map_builder.h"
#include "../common/bytelevel_tokenizer.h"
#include <unicode/normalizer2.h>
#include <unicode/locid.h>
#include <unicode/uchar.h>
#include <new>
#include <unordered_map>

namespace pih::deepseek_v41 {
namespace {
std::string Normalize(std::string_view text, const icu::Normalizer2& nfkc, const icu::Normalizer2& nfd) {
  UErrorCode error = U_ZERO_ERROR;
  icu::UnicodeString first, decomposed, stripped, lower, collapsed;
  nfkc.normalize(icu::UnicodeString::fromUTF8(text), first, error);
  nfd.normalize(first, decomposed, error);
  if (U_FAILURE(error)) throw std::runtime_error("Engram Unicode normalization failed");
  for (int32_t i = 0; i < decomposed.length();) {
    const auto c = decomposed.char32At(i); i += U16_LENGTH(c);
    const auto category = u_charType(c);
    if (category != U_NON_SPACING_MARK && category != U_COMBINING_SPACING_MARK && category != U_ENCLOSING_MARK)
      stripped.append(c);
  }
  // Rust's normalizer lowercases each scalar independently. Do not apply
  // whole-string contextual casing (e.g. Greek final sigma).
  for (int32_t i = 0; i < stripped.length();) {
    const auto c = stripped.char32At(i); i += U16_LENGTH(c);
    icu::UnicodeString scalar; scalar.append(c); scalar.toLower(icu::Locale::getRoot()); lower.append(scalar);
  }
  bool previous_space = false;
  for (int32_t i = 0; i < lower.length();) {
    const auto c = lower.char32At(i); i += U16_LENGTH(c);
    const bool space = c == ' ' || c == '\t' || c == '\r' || c == '\n';
    if (!space || !previous_space) collapsed.append(space ? UChar32{' '} : c);
    previous_space = space;
  }
  if (collapsed.length() == 1 && collapsed.charAt(0) == ' ') collapsed.setCharAt(0, 0xe000);
  int32_t begin = 0, end = collapsed.length();
  while (begin < end && u_isUWhiteSpace(collapsed.char32At(begin))) begin += U16_LENGTH(collapsed.char32At(begin));
  while (begin < end) {
    const auto previous = collapsed.moveIndex32(end, -1);
    if (!u_isUWhiteSpace(collapsed.char32At(previous))) break;
    end = previous;
  }
  icu::UnicodeString restored;
  for (int32_t i = begin; i < end;) {
    const auto c = collapsed.char32At(i); i += U16_LENGTH(c);
    restored.append(c == 0xe000 ? UChar32{' '} : c);
  }
  if (first.isBogus() || decomposed.isBogus() || stripped.isBogus() || lower.isBogus() ||
      collapsed.isBogus() || restored.isBogus()) throw std::bad_alloc();
  std::string output; restored.toUTF8String(output);
  return output.empty() ? std::string(text) : output;
}
}
Result<std::vector<std::byte>> BuildEngramTokenMap(std::string_view json, const Sha256Digest& expected) {
  if (json.empty() || json.size() > 16 * 1024 * 1024 || expected == Sha256Digest{})
    return Status::InvalidArgument("Engram map build requires bounded tokenizer and independent map digest");
  try {
    auto digest = sha256(std::as_bytes(std::span(json.data(), json.size())));
    if (!digest.ok()) return digest.status();
    if (digest->hex() != plugin_text::kV41TokenizerSha256)
      return Status::FailedPrecondition("Engram tokenizer differs from frozen V4.1 artifact");
    auto tokenizer = plugin_text::Tokenizer::FromJson(json, plugin_text::TokenizerFamily::DeepSeekV41Flash);
    UErrorCode error = U_ZERO_ERROR;
    const auto* nfkc = icu::Normalizer2::getNFKCInstance(error);
    const auto* nfd = icu::Normalizer2::getNFDInstance(error);
    if (U_FAILURE(error) || !nfkc || !nfd) return Status::Unavailable("Engram ICU normalization data unavailable");
    std::unordered_map<std::string, std::uint32_t> keys;
    keys.reserve(99092);
    std::uint64_t key_bytes = 0;
    std::vector<std::byte> output(FlashConfig::kVocabularySize * 4);
    for (std::int64_t token = 0; token < FlashConfig::kVocabularySize; ++token) {
      const auto text = tokenizer.Decode(std::span(&token, 1));
      if (text.size() > 8192) return Status::InvalidArgument("Engram decoded token exceeds 8 KiB");
      auto key = text.find("\xef\xbf\xbd") != text.npos ? tokenizer.OriginalTokenSpelling(token) : Normalize(text, *nfkc, *nfd);
      if (key.size() > 65536) return Status::InvalidArgument("Engram normalized key exceeds 64 KiB");
      auto found = keys.find(key);
      if (found == keys.end()) {
        if (keys.size() >= 99092 || key.size() > (64ULL << 20) - key_bytes)
          return Status::FailedPrecondition("Engram compressed vocabulary or key budget differs");
        key_bytes += key.size();
        found = keys.emplace(std::move(key), static_cast<std::uint32_t>(keys.size())).first;
      }
      for (unsigned byte = 0; byte < 4; ++byte)
        output[static_cast<std::size_t>(token) * 4 + byte] = static_cast<std::byte>((found->second >> (8 * byte)) & 255U);
    }
    if (keys.size() != 99092) return Status::FailedPrecondition("Engram compressed vocabulary size differs from reference");
    auto map_digest = sha256(output); if (!map_digest.ok()) return map_digest.status();
    if (*map_digest != expected) return Status::FailedPrecondition("Generated Engram map differs from independently trusted map digest");
    return output;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Engram map generation allocation failed"); }
  catch (const std::exception&) { return Status::InvalidArgument("Engram tokenizer/normalizer generation failed"); }
}
}  // namespace pih::deepseek_v41
