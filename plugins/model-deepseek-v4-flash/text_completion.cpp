#include "text_completion.h"
#include "../common/deepseek_encoding.h"
#include "../common/deepseek_completion_decoder.h"
#include "../common/text_json.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace pih::deepseek_plugin {
namespace {
using plugin_text::QuoteText;
using plugin_text::TextJson;
const JsonValue& Field(const JsonValue& value, std::string_view key) {
  const auto* field = value.at(key);
  if (!field) throw std::invalid_argument("required text request field missing");
  return *field;
}
bool Boolean(const JsonValue& value, std::string_view key, bool fallback) {
  const auto* field = value.at(key); return field ? field->boolean() : fallback;
}
void Set(JsonValue& value, std::string key, JsonValue item) {
  auto object = value.object();
  for (auto& entry : object) if (entry.first == key) { entry.second = std::move(item); value = JsonValue(std::move(object)); return; }
  object.emplace_back(std::move(key), std::move(item)); value = JsonValue(std::move(object));
}
JsonValue WithCallIds(JsonValue value, uint64_t request_id, bool streaming) {
  const auto* calls = value.at("tool_calls");
  if (!calls || calls->array().empty()) return value;
  auto updated = calls->array();
  for (size_t i = 0; i < updated.size(); ++i) {
    Set(updated[i], "id", JsonValue("call_pih_ds_" + std::to_string(request_id) + "_" + std::to_string(i)));
    if (streaming) Set(updated[i], "index", JsonValue(static_cast<int64_t>(i)));
  }
  Set(value, "tool_calls", JsonValue(std::move(updated))); return value;
}
}

Result<std::string> CompleteText(DeepSeekEngine& engine,
    const plugin_text::DeepSeekSemanticArtifacts& semantic,
    uint32_t context_capacity, uint32_t prefill_chunk, uint32_t generation_timeout_ms,
    uint64_t& next_request, uint64_t& next_generation, uint64_t& next_plan,
    bool chat, std::string_view request, const pih_text_output_sink_v2& sink) {
  if (generation_timeout_ms < 1 || generation_timeout_ms > 86400000)
    return Status::InvalidArgument("DeepSeek generation timeout must be 1..86400000 ms");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(generation_timeout_ms);
  auto parsed = JsonValue::Parse(request, {1024 * 1024, 64, 16384, 1024 * 1024});
  if (!parsed.ok()) return parsed.status();
  const auto& body = *parsed;
  for (const auto& [key, value] : body.object()) {
    if (key == "model" || key == "max_tokens" || key == "temperature" || key == "top_p" ||
        key == "stream" || key == "seed" || key == "n" || key == (chat ? "messages" : "prompt")) continue;
    if (chat && (key == "thinking_mode" || key == "reasoning_effort" || key == "drop_thinking" ||
        key == "tools" || key == "response_format")) continue;
    throw std::invalid_argument("unsupported DeepSeek text request field");
  }
  constexpr std::string_view model = "deepseek-ai/DeepSeek-V4-Flash-0731";
  if (Field(body, "model").string() != model) throw std::invalid_argument("DeepSeek model identity mismatch");
  const bool streaming = Boolean(body, "stream", false);
  if (const auto* n = body.at("n"); n && (!n->is_integer() || n->integer() != 1))
    throw std::invalid_argument("only n=1 is supported");
  const auto maximum = body.at("max_tokens") ? body.at("max_tokens")->integer() : 128;
  if (maximum < 1 || static_cast<uint64_t>(maximum) >= context_capacity)
    throw std::invalid_argument("invalid max_tokens");
  DeepSeekRequestSamplingConfig sampling{};
  sampling.temperature = body.at("temperature") ? body.at("temperature")->number() : 0;
  sampling.top_p = body.at("top_p") ? body.at("top_p")->number() : 1;
  if (!std::isfinite(sampling.temperature) || sampling.temperature < 0 || sampling.temperature > 2 ||
      !std::isfinite(sampling.top_p) || sampling.top_p <= 0 || sampling.top_p > 1 ||
      (sampling.temperature == 0 && sampling.top_p != 1))
    throw std::invalid_argument("invalid temperature/top_p");
  sampling.mode = sampling.temperature == 0 ? DeepSeekSamplingMode::kGreedy : DeepSeekSamplingMode::kStochastic;
  const auto seed = body.at("seed") ? body.at("seed")->integer() : 7;
  if (seed < 0) throw std::invalid_argument("seed must be nonnegative");
  sampling.effective_seed = static_cast<uint64_t>(seed);
  plugin_text::DeepSeekEncodingOptions options;
  std::string prompt;
  if (chat) {
    if (const auto* tools = body.at("tools"); tools && !tools->is_array())
      throw std::invalid_argument("tools must be an array");
    if (const auto* format = body.at("response_format"); format && !format->is_object())
      throw std::invalid_argument("response_format must be an object");
    if (const auto* mode = body.at("thinking_mode")) {
      if (mode->string() != "chat" && mode->string() != "thinking") throw std::invalid_argument("invalid thinking_mode");
      options.thinking = mode->string() == "thinking";
    }
    options.drop_thinking = Boolean(body, "drop_thinking", true);
    if (const auto* effort = body.at("reasoning_effort")) options.reasoning_effort = effort->string();
    auto messages = Field(body, "messages").array();
    if (messages.empty() || messages.size() > 128) throw std::invalid_argument("invalid message count");
    const auto role = Field(messages.back(), "role").string();
    if (role != "user" && role != "developer" && role != "tool")
      throw std::invalid_argument("chat must end with user/developer/tool input");
    if (body.at("tools") || body.at("response_format")) {
      if (Field(messages.front(), "role").string() != "system")
        messages.insert(messages.begin(), JsonValue(JsonValue::Object{{"role", JsonValue(std::string("system"))}, {"content", JsonValue(std::string{})}}));
      for (auto key : {"tools", "response_format"}) if (body.at(key)) {
        if (messages.front().at(key)) throw std::invalid_argument("conflicting message/request tools or response_format");
        Set(messages.front(), key, *body.at(key));
      }
    }
    prompt = plugin_text::EncodeDeepSeekMessages(JsonValue(std::move(messages)), options);
  } else prompt = Field(body, "prompt").string();
  const auto encoded = semantic.tokenizer().Encode(prompt);
  if (encoded.empty() || encoded.size() + static_cast<uint64_t>(maximum) > context_capacity)
    throw std::invalid_argument("prompt plus completion exceeds sealed context");
  std::vector<uint32_t> tokens(encoded.begin(), encoded.end());
  if (next_request == std::numeric_limits<uint64_t>::max() || next_generation == std::numeric_limits<uint64_t>::max())
    return Status::ResourceExhausted("DeepSeek request identity exhausted");
  const auto id = next_request++, generation = next_generation++;
  const auto created = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  const auto identity = "{\"id\":\"pih-deepseek-" + std::to_string(id) + "\",\"created\":" + std::to_string(created) +
      ",\"model\":" + QuoteText(model) + ",\"object\":";
  auto chunk = [&](const JsonValue& delta, std::string_view raw, const char* finish, std::string_view usage = "") {
    const auto json = identity + QuoteText(chat ? "chat.completion.chunk" : "text_completion") +
        ",\"choices\":[{\"index\":0,\"finish_reason\":" + (finish ? QuoteText(finish) : "null") +
        (chat ? ",\"delta\":" + TextJson(WithCallIds(delta, id, true)) : ",\"text\":" + QuoteText(raw)) +
        "}]" + std::string(usage) + "}";
    return sink.write(sink.context, json.data(), json.size()) == 1;
  };
  if (std::chrono::steady_clock::now() >= deadline)
    return Status::DeadlineExceeded("DeepSeek request preparation deadline elapsed");
  if (sink.cancelled(sink.context) || sink.start(sink.context, streaming ? 1U : 0U) != 1)
    return Status::Unavailable("connection cancelled before generation");
  if (streaming && chat && !chunk(JsonValue(JsonValue::Object{{"role", JsonValue(std::string("assistant"))}}), "", nullptr))
    return Status::Unavailable("connection cancelled before generation");
  plugin_text::DeepSeekCompletionDecoder decoder(options.thinking);
  std::string pending_utf8, raw_result;
  const JsonValue empty(JsonValue::Object{});
  auto emit = [&](std::string_view text) {
    if (chat) {
      auto delta = decoder.Feed(text);
      return !streaming || delta.object().empty() || chunk(delta, "", nullptr);
    }
    if (text.size() > 8 * 1024 * 1024 - raw_result.size()) throw std::runtime_error("text output limit exceeded");
    raw_result += text;
    return !streaming || text.empty() || chunk(empty, text, nullptr);
  };
  GenerationCallbacks callbacks;
  callbacks.cancelled = [&] { return sink.cancelled(sink.context) != 0; };
  callbacks.on_token = [&](const DeepSeekAcceptedTokenSnapshot& snapshot) {
    const int64_t token = snapshot.token_ids.back();
    if (!chat && token == 1) return true;
    return emit(semantic.tokenizer().DecodeIncremental({&token, 1}, pending_utf8));
  };
  try {
  auto generated = Generate(engine, tokens, static_cast<uint32_t>(maximum), 0, sampling,
      prefill_chunk, context_capacity, id, generation, next_plan,
      deadline, callbacks);
  if (!generated.ok()) return generated.status();
  if (!emit(semantic.tokenizer().DecodeIncremental({}, pending_utf8, true)))
    return Status::Unavailable("connection cancelled after generation");
  const bool stopped = generated->finish_reason == DeepSeekFinishReason::kStop;
  JsonValue message = empty;
  if (chat) {
    auto final_delta = decoder.Finish(stopped);
    if (streaming && !final_delta.object().empty() && !chunk(final_delta, "", nullptr))
      return Status::Unavailable("connection cancelled after generation");
    message = WithCallIds(decoder.Message(), id, false);
  }
  const auto* finish = !stopped ? "length" : chat && !Field(message, "tool_calls").array().empty() ? "tool_calls" : "stop";
  const auto usage = ",\"usage\":{\"prompt_tokens\":" + std::to_string(tokens.size()) + ",\"completion_tokens\":" +
      std::to_string(generated->accepted_completion_count) + ",\"total_tokens\":" +
      std::to_string(tokens.size() + generated->accepted_completion_count) + "}";
  if (streaming) {
    if (!chunk(empty, "", finish, usage)) return Status::Unavailable("connection cancelled after generation");
    return std::string{};
  }
  auto response = identity + QuoteText(chat ? "chat.completion" : "text_completion") +
      ",\"choices\":[{\"index\":0,\"finish_reason\":" + QuoteText(finish) +
      (chat ? ",\"message\":" + TextJson(message) : ",\"text\":" + QuoteText(raw_result)) + "}]" + usage + "}";
  if (response.size() > 8 * 1024 * 1024) return Status::ResourceExhausted("text response limit exceeded");
  return response;
  } catch (const std::invalid_argument&) {
    throw std::runtime_error("native DeepSeek generated malformed text or DSML");
  } catch (const std::bad_variant_access&) {
    throw std::runtime_error("native DeepSeek generated invalid output types");
  }
}
}
