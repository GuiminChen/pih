#include "deepseek_completion_decoder.h"
#include "deepseek_encoding.h"
#include "text_json.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace pih::plugin_text {
namespace {
constexpr std::string_view bos = "<｜begin▁of▁sentence｜>", eos = "<｜end▁of▁sentence｜>";
constexpr std::string_view think = "<think>", end_think = "</think>", dsml = "｜DSML｜";
constexpr std::string_view tool_start = "\n\n<｜DSML｜tool_calls";
constexpr std::array markers{tool_start, bos, eos, think, end_think, dsml};
void Append(std::string& result, std::string_view text) {
  if (text.size() > 8 * 1024 * 1024 - result.size()) throw std::invalid_argument("DeepSeek completion exceeds 8 MiB");
  result += text;
}
void Delta(JsonValue::Object& delta, std::string_view name, std::string_view text) {
  if (text.empty()) return;
  for (auto& [key, value] : delta) if (key == name) {
    auto combined = value.string(); Append(combined, text); value = JsonValue(std::move(combined)); return;
  }
  delta.emplace_back(name, JsonValue(std::string(text)));
}
}

JsonValue DeepSeekCompletionDecoder::Feed(std::string_view decoded) {
  if (finished_) throw std::logic_error("DeepSeek decoder already finalized");
  RequireUtf8(decoded);
  if (phase_ == Phase::End && !decoded.empty()) throw std::invalid_argument("data after DeepSeek EOS");
  Append(raw_, decoded);
  if (phase_ == Phase::Tools) return JsonValue(JsonValue::Object{});
  Append(pending_, decoded);
  JsonValue::Object delta;
  auto emit = [&](size_t count) {
    const std::string_view part(pending_.data(), count);
    if (phase_ == Phase::Reasoning) { Append(reasoning_, part); Delta(delta, "reasoning_content", part); }
    else { Append(content_, part); Delta(delta, "content", part); }
    pending_.erase(0, count);
  };
  while (!pending_.empty()) {
    size_t at = std::string::npos; std::string_view marker;
    for (auto candidate : markers) {
      const auto position = pending_.find(candidate);
      if (position < at) { at = position; marker = candidate; }
    }
    // A partial outer tool marker can precede a complete inner DSML token.
    // Hold the earlier prefix rather than misclassifying that inner token.
    size_t retained = 0;
    for (auto candidate : markers) {
      const auto maximum = std::min(pending_.size(), candidate.size() - 1);
      for (size_t count = maximum; count > retained; --count)
        if (std::string_view(pending_).ends_with(candidate.substr(0, count))) { retained = count; break; }
    }
    if (retained && pending_.size() - retained <= at) {
      emit(pending_.size() - retained); break;
    }
    if (at == std::string::npos) { emit(pending_.size()); break; }
    emit(at);
    pending_.erase(0, marker.size());
    if (phase_ == Phase::Reasoning && marker == end_think) phase_ = Phase::Content;
    else if (phase_ == Phase::Content && marker == tool_start) {
      phase_ = Phase::Tools; pending_.clear(); break;
    } else if (phase_ == Phase::Content && marker == eos) {
      phase_ = Phase::End;
      if (!pending_.empty()) throw std::invalid_argument("data after DeepSeek EOS");
    } else throw std::invalid_argument("unexpected DeepSeek structural token");
  }
  return JsonValue(std::move(delta));
}

JsonValue DeepSeekCompletionDecoder::Finish(bool stopped) {
  if (finished_) throw std::logic_error("DeepSeek decoder already finalized");
  JsonValue::Object delta;
  if (stopped || phase_ == Phase::End) {
    const auto parsed = ParseDeepSeekCompletion(raw_, thinking_);
    // Earlier emitted plain text must exactly equal the strict final parse.
    if (parsed.at("content")->string() != content_ || parsed.at("reasoning_content")->string() != reasoning_)
      throw std::logic_error("DeepSeek incremental/final parse mismatch");
    calls_ = parsed.at("tool_calls")->array();
    if (!calls_.empty()) delta.emplace_back("tool_calls", JsonValue(calls_));
  } else if (phase_ == Phase::Reasoning || phase_ == Phase::Content) {
    // Length exhaustion may leave ordinary text ending like a marker prefix.
    // Preserve it, except an already identifiable unfinished DSML block.
    if (pending_.find(dsml) == std::string::npos) {
      if (phase_ == Phase::Reasoning) { Append(reasoning_, pending_); Delta(delta, "reasoning_content", pending_); }
      else { Append(content_, pending_); Delta(delta, "content", pending_); }
    }
  }
  // Length-truncated tool blocks never become executable calls or plain text.
  pending_.clear(); finished_ = true;
  return JsonValue(std::move(delta));
}

JsonValue DeepSeekCompletionDecoder::Message() const {
  if (!finished_) throw std::logic_error("DeepSeek decoder has not been finalized");
  return JsonValue(JsonValue::Object{{"role", JsonValue(std::string("assistant"))},
      {"content", JsonValue(content_)}, {"reasoning_content", JsonValue(reasoning_)},
      {"tool_calls", JsonValue(calls_)}});
}
}
