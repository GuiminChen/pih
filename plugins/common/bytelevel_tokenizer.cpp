#include "bytelevel_tokenizer.h"
#include "pih/core/bounded_json.h"
#include "pih/core/sha256.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace pih::plugin_text {
namespace {
const JsonValue& Field(const JsonValue& object, std::string_view name) {
  auto* value = object.at(name);
  if (!value) throw std::invalid_argument("tokenizer field missing");
  return *value;
}
std::string Pair(const std::string& a, const std::string& b) {
  return a + '\0' + b;
}
std::string ReadTokenizer(const std::filesystem::path& path) {
  const auto size = std::filesystem::file_size(path);
  if (size == 0 || size > 64 * 1024 * 1024) throw std::invalid_argument("tokenizer size invalid");
  std::ifstream input(path, std::ios::binary);
  std::string bytes(static_cast<size_t>(size), '\0');
  if (!input.read(bytes.data(), bytes.size())) throw std::invalid_argument("tokenizer read failed");
  if (input.peek() != std::char_traits<char>::eof()) throw std::invalid_argument("tokenizer grew during read");
  return bytes;
}
}

Tokenizer::Tokenizer(const std::filesystem::path& path, TokenizerFamily family)
    : Tokenizer(ReadTokenizer(path), family, JsonBytes{}) {}

Tokenizer Tokenizer::FromJson(std::string_view bytes, TokenizerFamily family) {
  return Tokenizer(bytes, family, JsonBytes{});
}

Tokenizer::Tokenizer(std::string_view bytes, TokenizerFamily family, JsonBytes)
    : family_(family) {
  if (family != TokenizerFamily::Qwen3 && family != TokenizerFamily::DeepSeekV4Flash0731 &&
      family != TokenizerFamily::DeepSeekV41Flash) throw std::invalid_argument("tokenizer family invalid");
  const bool deepseek = family != TokenizerFamily::Qwen3;
  const int64_t vocabulary_size = deepseek ? 129280 : 151936;
  if (bytes.empty() || bytes.size() > 64 * 1024 * 1024) throw std::invalid_argument("tokenizer size invalid");
  if (family == TokenizerFamily::DeepSeekV41Flash) {
    const auto digest = sha256(std::as_bytes(std::span(bytes.data(), bytes.size())));
    if (!digest.ok() || digest->hex() != kV41TokenizerSha256)
      throw std::invalid_argument("DeepSeek V4.1 tokenizer differs from frozen official snapshot");
  }
  auto parsed = JsonValue::Parse(bytes, {64 * 1024 * 1024, 32, 2000000, 1024 * 1024});
  if (!parsed.ok()) throw std::invalid_argument("tokenizer JSON invalid");
  const auto& model = Field(*parsed, "model");
  if (Field(model, "type").string() != "BPE") throw std::invalid_argument("native tokenizer requires BPE");
  for (auto key : {"continuing_subword_prefix", "end_of_word_suffix", "unk_token", "dropout"}) {
    const auto* option = model.at(key);
    if (option && !option->is_null() && !(option->is_string() && option->string().empty()))
      throw std::invalid_argument("unsupported BPE option");
  }
  for (auto key : {"byte_fallback", "ignore_merges", "fuse_unk"}) {
    const auto* option = model.at(key);
    if (option && (!option->is_boolean() || option->boolean())) throw std::invalid_argument("unsupported BPE mode");
  }
  const auto* normalizer = parsed->at("normalizer");
  if (normalizer && !normalizer->is_null() &&
      !(deepseek && Field(*normalizer, "type").string() == "Sequence" &&
        Field(*normalizer, "normalizers").array().empty()))
    throw std::invalid_argument("unsupported tokenizer normalizer");
  if (Field(Field(*parsed, "decoder"), "type").string() != "ByteLevel")
    throw std::invalid_argument("native tokenizer requires ByteLevel decoding");
  for (const auto& token : Field(*parsed, "added_tokens").array()) {
    const auto& text = Field(token, "content").string();
    const auto id = Field(token, "id").integer();
    const bool special = Field(token, "special").boolean();
    if (text.empty() || id < 0 || id >= vocabulary_size || Field(token, "lstrip").boolean() ||
        Field(token, "rstrip").boolean() || Field(token, "single_word").boolean() ||
        Field(token, "normalized").boolean()) throw std::invalid_argument("unsupported added token");
    if (std::any_of(specials_.begin(), specials_.end(), [&](const auto& entry) { return entry.first == text; }))
      throw std::invalid_argument("duplicate added token content");
    specials_.emplace_back(text, id);
    // DeepSeek's output parser consumes structural special tokens. Qwen's
    // non-thinking text output instead skips special tokens.
    if (!added_decode_.emplace(id, !deepseek && special ? "" : text).second)
      throw std::invalid_argument("duplicate added token id");
    added_spelling_.emplace(id, text);
  }
  const std::vector<std::pair<std::string, int64_t>> expected_specials = deepseek ?
      std::vector<std::pair<std::string, int64_t>>{{"<｜begin▁of▁sentence｜>", 0},
        {"<｜end▁of▁sentence｜>", 1}, {"<｜▁pad▁｜>", 2}} :
      std::vector<std::pair<std::string, int64_t>>{{"<|endoftext|>", 151643},
        {"<|im_start|>", 151644}, {"<|im_end|>", 151645}};
  for (const auto& expected : expected_specials)
    if (std::find(specials_.begin(), specials_.end(), expected) == specials_.end())
      throw std::invalid_argument("special token identity mismatch");
  for (const auto& [token, id] : Field(model, "vocab").object()) {
    if (!id.is_integer() || id.integer() < 0 || id.integer() >= vocabulary_size ||
        !vocab_.emplace(token, id.integer()).second || !decode_.emplace(id.integer(), token).second)
      throw std::invalid_argument("tokenizer vocabulary invalid");
  }
  size_t rank = 0;
  if (deepseek) {
    if (decode_.size() != 128000 || added_decode_.size() != 1283)
      throw std::invalid_argument("DeepSeek vocabulary inventory mismatch");
    for (int64_t id = 0; id < vocabulary_size; ++id) {
      if ((id < 128000 && !decode_.contains(id)) ||
          ((id < 3 || id >= 128000) != added_decode_.contains(id)))
        throw std::invalid_argument("DeepSeek vocabulary is not dense");
    }
  }
  for (const auto& merge : Field(model, "merges").array()) {
    std::string first, second;
    if (merge.is_array() && merge.array().size() == 2) {
      first = merge.array()[0].string(); second = merge.array()[1].string();
    } else if (merge.is_string()) {
      auto pos = merge.string().find(' ');
      if (pos == std::string::npos) throw std::invalid_argument("invalid merge");
      first = merge.string().substr(0, pos); second = merge.string().substr(pos + 1);
    } else throw std::invalid_argument("invalid BPE merges");
    if (!vocab_.contains(first) || !vocab_.contains(second) || !vocab_.contains(first + second) ||
        !ranks_.emplace(Pair(first, second), rank++).second) throw std::invalid_argument("invalid merge inventory");
  }
  const auto& pre = Field(*parsed, "pre_tokenizer");
  if (Field(pre, "type").string() != "Sequence") throw std::invalid_argument("unexpected pre-tokenizer");
  const auto& sequence = Field(pre, "pretokenizers").array();
  if (sequence.size() != (deepseek ? 4 : 2) ||
      Field(sequence.back(), "type").string() != "ByteLevel" ||
      Field(sequence.back(), "add_prefix_space").boolean() ||
      Field(sequence.back(), "use_regex").boolean())
    throw std::invalid_argument("unsupported pre-tokenizer options");
  for (size_t i = 0; i + 1 < sequence.size(); ++i) {
    const auto& stage = sequence[i];
    if (Field(stage, "type").string() != "Split" ||
        Field(stage, "behavior").string() != "Isolated" || Field(stage, "invert").boolean())
      throw std::invalid_argument("unsupported split pre-tokenizer options");
    UErrorCode error = U_ZERO_ERROR;
    splits_.emplace_back(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(
        Field(Field(stage, "pattern"), "Regex").string()), 0, error));
    if (U_FAILURE(error) || !splits_.back()) throw std::invalid_argument("pre-tokenizer regex invalid");
  }
  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    int codepoint = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 ? b : 256 + extra++;
    icu::UnicodeString character(codepoint);
    character.toUTF8String(bytes_[b]);
    inverse_.emplace(codepoint, static_cast<unsigned char>(b));
    if (!vocab_.contains(bytes_[b])) throw std::invalid_argument("byte vocabulary incomplete");
  }
}

std::vector<int64_t> Tokenizer::Encode(std::string_view text) const {
  if (text.size() > 1024 * 1024) throw std::invalid_argument("tokenizer input exceeds 1 MiB");
  std::vector<int64_t> result;
  while (!text.empty()) {
    size_t nearest = std::string_view::npos, length = 0; int64_t id = 0;
    for (const auto& [special, token] : specials_) {
      const auto at = text.find(special);
      if (at != std::string_view::npos && (at < nearest || (at == nearest && special.size() > length))) {
        nearest = at; length = special.size(); id = token;
      }
    }
    auto prefix = EncodePlain(text.substr(0, nearest)); result.insert(result.end(), prefix.begin(), prefix.end());
    if (nearest == std::string_view::npos) break;
    result.push_back(id); text.remove_prefix(nearest + length);
  }
  return result;
}

std::vector<int64_t> Tokenizer::EncodeText(std::string_view text) const {
  if (text.size() > 1024 * 1024) throw std::invalid_argument("tokenizer input exceeds 1 MiB");
  return EncodePlain(text);
}

std::vector<int64_t> Tokenizer::EncodePlain(std::string_view text) const {
  auto unicode = icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(), static_cast<int32_t>(text.size())));
  std::string roundtrip; unicode.toUTF8String(roundtrip);
  if (roundtrip != text) throw std::invalid_argument("input is not valid UTF-8");
  std::vector<icu::UnicodeString> pieces{unicode};
  for (const auto& split : splits_) {
    std::vector<icu::UnicodeString> next;
    for (const auto& input : pieces) {
      UErrorCode error = U_ZERO_ERROR;
      std::unique_ptr<icu::RegexMatcher> matcher(split->matcher(input, error));
      if (U_FAILURE(error) || !matcher) throw std::runtime_error("regex matcher failed");
      matcher->setTimeLimit(1000, error);
      matcher->setStackLimit(4 * 1024 * 1024, error);
      int32_t consumed = 0;
      // Isolated splitting preserves unmatched gaps. Later rules apply to
      // each earlier piece independently, never across a split boundary.
      while (matcher->find(error)) {
        const auto start = matcher->start(error), end = matcher->end(error);
        if (U_FAILURE(error) || start < consumed || end <= start)
          throw std::invalid_argument("invalid or empty pre-tokenizer match");
        if (start > consumed) next.push_back(icu::UnicodeString(input, consumed, start - consumed));
        next.push_back(icu::UnicodeString(input, start, end - start));
        consumed = end;
      }
      if (U_FAILURE(error)) throw std::invalid_argument("pre-tokenizer failed");
      if (consumed < input.length()) next.push_back(icu::UnicodeString(input, consumed));
    }
    pieces = std::move(next);
  }
  std::vector<int64_t> result;
  for (const auto& part : pieces) {
    std::string piece; part.toUTF8String(piece);
    // Bound the quadratic reference BPE merge loop against pathological input.
    if (piece.size() > 8192) throw std::invalid_argument("pre-token exceeds 8192 bytes");
    std::vector<std::string> symbols;
    for (unsigned char byte : piece) symbols.push_back(bytes_[byte]);
    while (symbols.size() > 1) {
      size_t best = std::numeric_limits<size_t>::max(), at = 0;
      for (size_t i = 0; i + 1 < symbols.size(); ++i) {
        auto found = ranks_.find(Pair(symbols[i], symbols[i + 1]));
        if (found != ranks_.end() && found->second < best) { best = found->second; at = i; }
      }
      if (best == std::numeric_limits<size_t>::max()) break;
      symbols[at] += symbols[at + 1];
      symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(at + 1));
    }
    for (const auto& symbol : symbols) result.push_back(vocab_.at(symbol));
  }
  return result;
}

std::string Tokenizer::RawTokenBytes(int64_t id) const {
  if (auto added = added_decode_.find(id); added != added_decode_.end()) return added->second;
  const auto it = decode_.find(id);
  if (it == decode_.end()) throw std::runtime_error("generated token has no tokenizer mapping");
  const auto value = icu::UnicodeString::fromUTF8(it->second);
  std::string result;
  for (int32_t i = 0; i < value.length();) {
    auto codepoint = value.char32At(i); i += U16_LENGTH(codepoint);
    auto byte = inverse_.find(codepoint);
    if (byte == inverse_.end()) throw std::runtime_error("token contains non-byte alphabet");
    if (result.size() == 8 * 1024 * 1024) throw std::invalid_argument("decoded token exceeds 8 MiB");
    result.push_back(static_cast<char>(byte->second));
  }
  return result;
}

std::string Tokenizer::OriginalTokenSpelling(int64_t id) const {
  if (const auto added = added_spelling_.find(id); added != added_spelling_.end()) return added->second;
  const auto found = decode_.find(id);
  if (found == decode_.end()) throw std::invalid_argument("token id has no original spelling");
  return found->second;
}

std::vector<std::string> Tokenizer::RawVocabulary(std::uint64_t budget) const {
  const int64_t size = family_ == TokenizerFamily::Qwen3 ? 151936 : 129280;
  std::vector<std::string> result;
  result.reserve(static_cast<std::size_t>(size));
  for (int64_t id = 0; id < size; ++id) {
    auto bytes = RawTokenBytes(id);
    if (bytes.size() > 8192 || bytes.size() > budget)
      throw std::invalid_argument("raw tokenizer vocabulary exceeds byte budget");
    budget -= bytes.size();
    result.push_back(std::move(bytes));
  }
  return result;
}

std::string Tokenizer::DecodeBytes(std::span<const int64_t> tokens) const {
  std::string result;
  for (auto id : tokens) {
    if (family_ == TokenizerFamily::Qwen3 && (id == 151643 || id == 151645)) break;
    auto bytes = RawTokenBytes(id);
    if (bytes.size() > 8 * 1024 * 1024 - result.size())
      throw std::invalid_argument("decoded text exceeds 8 MiB");
    result += bytes;
  }
  return result;
}

std::string Tokenizer::Decode(std::span<const int64_t> tokens) const {
  auto result = DecodeBytes(tokens);
  // Match replacement-character decoding for generation cut inside a UTF-8 codepoint.
  auto repaired = icu::UnicodeString::fromUTF8(result);
  result.clear(); repaired.toUTF8String(result);
  return result;
}

std::string Tokenizer::DecodeIncremental(std::span<const int64_t> tokens,
    std::string& pending, bool final) const {
  pending += DecodeBytes(tokens);
  std::size_t end = pending.size();
  if (!final && end != 0) {
    auto start = end - 1;
    while (start > 0 && (static_cast<unsigned char>(pending[start]) & 0xc0) == 0x80) --start;
    const auto lead = static_cast<unsigned char>(pending[start]);
    const std::size_t expected = lead >= 0xc2 && lead <= 0xdf ? 2 :
        lead >= 0xe0 && lead <= 0xef ? 3 : lead >= 0xf0 && lead <= 0xf4 ? 4 : 1;
    if (end - start < expected) end = start;
  }
  auto unicode = icu::UnicodeString::fromUTF8(icu::StringPiece(pending.data(), static_cast<int32_t>(end)));
  std::string result;
  unicode.toUTF8String(result);
  pending.erase(0, end);
  return result;
}
}
