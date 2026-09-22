#pragma once
#include "../common/bytelevel_tokenizer.h"
#include <span>

namespace pih::qwen_plugin {
using Tokenizer = pih::plugin_text::Tokenizer;
inline constexpr std::string_view kTokenizerSha256 =
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4";
// Verify the bounded byte snapshot before parsing, without reopening the file.
// This pins tokenizer semantics, not the model weights or filesystem identity.
Tokenizer LoadPinnedTokenizer(const std::filesystem::path& path);
Tokenizer LoadPinnedTokenizer(std::span<const std::byte> authenticated_bytes);
}
