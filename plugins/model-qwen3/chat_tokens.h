#pragma once

#include "chat_format.h"
#include "tokenizer.h"

namespace pih::qwen_plugin {

// The service and offline token inspection use this exact assembly path.
inline std::vector<int64_t> PlainChatTokens(
    const JsonValue& messages, const Tokenizer& tokenizer, size_t maximum_tokens) {
  if (!maximum_tokens || maximum_tokens > 65536)
    throw std::invalid_argument("invalid chat token limit");
  std::vector<int64_t> result;
  for (const auto& segment : PlainChatPlan(messages)) {
    if (segment.control_token >= 0) {
      if (result.size() == maximum_tokens)
        throw std::invalid_argument("chat exceeds token limit");
      result.push_back(segment.control_token);
    } else {
      auto part = segment.recognize_specials ? tokenizer.Encode(segment.text)
                                            : tokenizer.EncodeText(segment.text);
      if (part.size() > maximum_tokens - result.size())
        throw std::invalid_argument("chat exceeds token limit");
      result.insert(result.end(), part.begin(), part.end());
    }
  }
  return result;
}

}  // namespace pih::qwen_plugin
