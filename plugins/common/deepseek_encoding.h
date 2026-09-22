#pragma once
#include "pih/core/bounded_json.h"

namespace pih::plugin_text {
struct DeepSeekEncodingOptions final {
  bool thinking = false;
  bool drop_thinking = true;
  bool add_bos = true;
  std::string reasoning_effort = "low";
};
// Full-conversation rendering: messages include per-message tools/response_format.
// Prefix-cache context is deliberately not accepted by this stateless interface.
std::string EncodeDeepSeekMessages(const JsonValue& messages,
    const DeepSeekEncodingOptions& options = {});
// Strict complete-turn decoder; requires EOS, never invents a tool call from a
// length-truncated or malformed DSML fragment. Returns OpenAI message fields.
JsonValue ParseDeepSeekCompletion(std::string_view text, bool thinking);
}
