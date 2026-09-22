#include "tokenizer.h"
#include "pih/core/sha256.h"
#include <fstream>
#include <stdexcept>

namespace pih::qwen_plugin {

Tokenizer LoadPinnedTokenizer(std::span<const std::byte> authenticated_bytes) {
  constexpr uint64_t kMaximumBytes = 12 * 1024 * 1024;
  if (authenticated_bytes.empty() || authenticated_bytes.size() > kMaximumBytes)
    throw std::invalid_argument("Qwen tokenizer snapshot extent invalid");
  const auto observed = pih::sha256(authenticated_bytes);
  if (!observed.ok() || observed->hex() != kTokenizerSha256)
    throw std::invalid_argument("Qwen tokenizer differs from the pinned artifact");
  const auto bytes = std::string_view(
      reinterpret_cast<const char*>(authenticated_bytes.data()), authenticated_bytes.size());
  return Tokenizer::FromJson(bytes, pih::plugin_text::TokenizerFamily::Qwen3);
}

Tokenizer LoadPinnedTokenizer(const std::filesystem::path& path) {
  constexpr uint64_t kMaximumBytes = 12 * 1024 * 1024;
  const auto size = std::filesystem::file_size(path);
  if (!size || size > kMaximumBytes)
    throw std::invalid_argument("Qwen tokenizer byte count exceeds its bound");
  std::ifstream input(path, std::ios::binary);
  std::string bytes(static_cast<size_t>(size), '\0');
  if (!input.read(bytes.data(), bytes.size()) || input.peek() != std::char_traits<char>::eof())
    throw std::invalid_argument("Qwen tokenizer read failed or size changed");
  const auto observed = pih::sha256(std::as_bytes(std::span(bytes.data(), bytes.size())));
  if (!observed.ok()) throw std::runtime_error("Qwen tokenizer digest calculation failed");
  if (observed->hex() != kTokenizerSha256)
    throw std::invalid_argument("Qwen tokenizer differs from the pinned Qwen3 artifact");
  return Tokenizer::FromJson(bytes, pih::plugin_text::TokenizerFamily::Qwen3);
}

}  // namespace pih::qwen_plugin
