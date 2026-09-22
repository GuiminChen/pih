#include "tokenizer.h"
#include <new>
#include <stdexcept>

namespace pih::deepseek_v41 {
Result<std::unique_ptr<V41Tokenizer>> V41Tokenizer::Open(const std::filesystem::path& directory, std::uint64_t budget) {
  if (!budget || budget > (64U << 20)) return Status::InvalidArgument("V4.1 raw vocabulary budget invalid");
  try {
    const auto expected = Sha256Digest::ParseHex(plugin_text::kV41TokenizerSha256);
    if (!expected.ok()) return expected.status();
    auto bytes = ReadAuthenticatedMetadata(directory / "tokenizer.json", *expected, 16U << 20);
    if (!bytes.ok()) return bytes.status();
    auto tokenizer = plugin_text::Tokenizer::FromJson(
        {reinterpret_cast<const char*>(bytes->data()), bytes->size()}, plugin_text::TokenizerFamily::DeepSeekV41Flash);
    auto owner = std::unique_ptr<V41Tokenizer>(new V41Tokenizer(std::move(tokenizer)));
    owner->bytes_ = owner->tokenizer_.RawVocabulary(budget);
    if (owner->bytes_.size() != FlashConfig::kVocabularySize) return Status::FailedPrecondition("V4.1 vocabulary dimension differs");
    owner->views_.reserve(owner->bytes_.size());
    for (const auto& value : owner->bytes_) owner->views_.push_back(value);
    return owner;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("V4.1 tokenizer allocation failed"); }
  catch (const std::exception&) { return Status::FailedPrecondition("V4.1 tokenizer snapshot or semantics invalid"); }
}
Result<std::vector<std::uint32_t>> V41Tokenizer::EncodeRendered(std::string_view prompt) const {
  try {
    const auto encoded = tokenizer_.Encode(prompt);
    if (encoded.empty() || encoded.size() > 4096) return Status::InvalidArgument("V4.1 prompt token count outside native prefill bounds");
    std::vector<std::uint32_t> output; output.reserve(encoded.size());
    for (const auto id : encoded) {
      if (id < 0 || id >= FlashConfig::kVocabularySize) return Status::FailedPrecondition("V4.1 prompt token identity invalid");
      output.push_back(static_cast<std::uint32_t>(id));
    }
    return output;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("V4.1 prompt tokenization allocation failed"); }
  catch (const std::exception&) { return Status::InvalidArgument("V4.1 rendered prompt tokenization failed"); }
}
}  // namespace pih::deepseek_v41
