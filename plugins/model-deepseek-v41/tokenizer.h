#pragma once
#include "../common/bytelevel_tokenizer.h"
#include "weight_files.h"

namespace pih::deepseek_v41 {
// Supervisor CPU adapter. Authenticated tokenizer only; chat rendering and
// tool/DSML classification are separate semantic admission responsibilities.
class V41Tokenizer final {
 public:
  static Result<std::unique_ptr<V41Tokenizer>> Open(const std::filesystem::path& artifact_directory,
      std::uint64_t raw_vocabulary_budget);
  V41Tokenizer(const V41Tokenizer&) = delete;
  V41Tokenizer& operator=(const V41Tokenizer&) = delete;
  // Caller supplies the correctly rendered model prompt, including any special
  // tokens. This method does not invent a chat template or prepend BOS.
  Result<std::vector<std::uint32_t>> EncodeRendered(std::string_view prompt) const;
  std::span<const std::string_view> token_bytes() const noexcept { return views_; }
  const plugin_text::Tokenizer& tokenizer() const noexcept { return tokenizer_; }
 private:
  explicit V41Tokenizer(plugin_text::Tokenizer tokenizer) : tokenizer_(std::move(tokenizer)) {}
  plugin_text::Tokenizer tokenizer_;
  std::vector<std::string> bytes_;
  std::vector<std::string_view> views_;
};
}  // namespace pih::deepseek_v41
