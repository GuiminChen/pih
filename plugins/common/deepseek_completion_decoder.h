#pragma once
#include "pih/core/bounded_json.h"

namespace pih::plugin_text {
// A request-owned UTF-8 text state machine. The tokenizer must retain incomplete
// UTF-8 bytes before Feed. No engine, transport or Python objects are involved.
// Discard the decoder after any exception; failed streams cannot be resumed.
class DeepSeekCompletionDecoder final {
 public:
  explicit DeepSeekCompletionDecoder(bool thinking) : thinking_(thinking), phase_(thinking ? Phase::Reasoning : Phase::Content) {}
  // Returns an OpenAI-style delta object. Text/reasoning is incremental; DSML
  // calls are withheld until Finish validates the entire completed turn.
  JsonValue Feed(std::string_view decoded);
  JsonValue Finish(bool stopped);
  JsonValue Message() const;
 private:
  enum class Phase { Reasoning, Content, Tools, End };
  bool thinking_, finished_ = false;
  Phase phase_;
  std::string raw_, pending_, content_, reasoning_;
  JsonValue::Array calls_;
};
}
