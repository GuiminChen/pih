#pragma once
#include "bytelevel_tokenizer.h"
#include <utility>

namespace pih::plugin_text {
// Immutable native value constructed only after the complete pinned semantic
// closure has been copied and authenticated. No Python source is executed.
class DeepSeekSemanticArtifacts final {
 public:
  static constexpr std::string_view kRevision = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb";
  static constexpr std::string_view kClosureRoot = "f77784ede96fb3bc9520622850f94d545ee729bca01bfaecc6d14a29a62ef03c";
  static DeepSeekSemanticArtifacts Load(const std::filesystem::path& snapshot_root);
  const Tokenizer& tokenizer() const noexcept { return tokenizer_; }
 private:
  explicit DeepSeekSemanticArtifacts(Tokenizer tokenizer) : tokenizer_(std::move(tokenizer)) {}
  Tokenizer tokenizer_;
};
}
