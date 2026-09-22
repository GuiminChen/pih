#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <unicode/regex.h>

namespace pih::plugin_text {
enum class TokenizerFamily { Qwen3, DeepSeekV4Flash0731, DeepSeekV41Flash };
inline constexpr std::string_view kV41TokenizerSha256 = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b";
class Tokenizer final {
 public:
  explicit Tokenizer(const std::filesystem::path& path,
      TokenizerFamily family = TokenizerFamily::Qwen3);
  // Parse the caller's already-read bytes without reopening a mutable path.
  // This method does not itself authenticate those bytes.
  static Tokenizer FromJson(std::string_view bytes, TokenizerFamily family);
  std::vector<int64_t> Encode(std::string_view text) const;
  // Literal caller text: never recognize registered added-token spellings.
  // Encode remains available for explicitly rendered/trusted model prompts.
  std::vector<int64_t> EncodeText(std::string_view text) const;
  std::string Decode(std::span<const int64_t> tokens) const;
  // Exact token bytes before UTF-8 replacement. Individual byte-level tokens
  // need not be valid UTF-8; do not decode each token separately for stop matching.
  std::string RawTokenBytes(int64_t token) const;
  // Original vocabulary spelling, before byte-alphabet decoding or special-token
  // filtering. Required when Engram keys partial UTF-8 tokens by raw spelling.
  std::string OriginalTokenSpelling(int64_t token) const;
  // Owned dense ID-ordered byte table for the controller's raw-byte stop ledger.
  // Bounds aggregate payload and each entry; special-token policy is unchanged.
  std::vector<std::string> RawVocabulary(std::uint64_t payload_budget) const;
  // Retain incomplete UTF-8 across token boundaries; final flush uses the same
  // replacement decoding as Decode(). The caller owns one pending buffer/request.
  std::string DecodeIncremental(std::span<const int64_t> tokens,
      std::string& pending, bool final = false) const;
 private:
  struct JsonBytes {};
  Tokenizer(std::string_view bytes, TokenizerFamily family, JsonBytes);
  std::string DecodeBytes(std::span<const int64_t> tokens) const;
  std::vector<int64_t> EncodePlain(std::string_view text) const;
  std::vector<std::pair<std::string, int64_t>> specials_;
  std::unordered_map<int64_t, std::string> added_decode_;
  std::unordered_map<int64_t, std::string> added_spelling_;
  TokenizerFamily family_;
  std::vector<std::unique_ptr<icu::RegexPattern>> splits_;
  std::unordered_map<std::string, int64_t> vocab_;
  std::unordered_map<int64_t, std::string> decode_;
  std::unordered_map<std::string, size_t> ranks_;
  std::string bytes_[256];
  std::unordered_map<int32_t, unsigned char> inverse_;
};
}
