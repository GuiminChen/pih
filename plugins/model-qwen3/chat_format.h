#pragma once

#include "pih/core/bounded_json.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pih::qwen_plugin {

// Shared by the model service and CPU format tool. Control tokens must stay
// separate from caller text; literal control-token spellings in content are
// ordinary text, not template authority. This is the plain-text subset only.
struct ChatSegment {
  std::string text;
  int64_t control_token{-1};
  bool recognize_specials{false};
};

inline std::vector<ChatSegment> PlainChatPlan(const JsonValue& messages) {
  constexpr size_t kMaximumRenderedBytes = 2 * 1024 * 1024;
  if (!messages.is_array() || messages.array().empty() || messages.array().size() > 128)
    throw std::invalid_argument("invalid message count");
  std::vector<ChatSegment> result;
  size_t bytes = 0;
  auto append = [&](std::string_view text, int64_t control = -1, bool recognize_specials = false) {
    if (text.size() > kMaximumRenderedBytes - bytes)
      throw std::invalid_argument("rendered chat exceeds byte limit");
    bytes += text.size();
    // BPE must see contiguous text as one span: splitting a role/newline/content
    // arbitrarily can change regex pieces and merge opportunities.
    if (control < 0 && !result.empty() && result.back().control_token < 0 &&
        result.back().recognize_specials == recognize_specials) {
      result.back().text.append(text);
    } else {
      result.push_back({std::string(text), control, recognize_specials});
    }
  };
  const auto* last_role = messages.array().back().at("role");
  if (!last_role || !last_role->is_string() || last_role->string() != "user")
    throw std::invalid_argument("plain-text chat must end with a user message");
  for (const auto& message : messages.array()) {
    const auto* role = message.at("role");
    const auto* content = message.at("content");
    if (!message.is_object() || message.object().size() != 2 || !role || !content ||
        !role->is_string() || !content->is_string() ||
        (role->string() != "system" && role->string() != "user" && role->string() != "assistant"))
      throw std::invalid_argument("only text system/user/assistant messages are supported");
    if (role->string() == "assistant" && content->string().find("</think>") != std::string::npos)
      throw std::invalid_argument("reasoning histories are not supported by the plain-text template");
    append("<|im_start|>", 151644);
    append(role->string());
    append("\n");
    append(content->string());
    append("<|im_end|>", 151645);
    append("\n");
  }
  append("<|im_start|>", 151644);
  // Constant template text may contain model-defined thinking delimiters.
  // No caller-controlled byte enters this trusted segment.
  append("assistant\n<think>\n\n</think>\n\n", -1, true);
  return result;
}

}  // namespace pih::qwen_plugin
