#include "completion_request.h"

#include <algorithm>
#include <cmath>
#include <set>

#include "pih/core/bounded_json.h"

namespace pih::surface_openai_http {
namespace {

constexpr std::size_t kMaximumPromptBytes = 4 * 1024 * 1024;
constexpr std::size_t kMaximumPromptCount = 256;
constexpr std::uint32_t kMaximumCompletionTokens = 40'960;
constexpr std::size_t kMaximumStopCount = 16;
constexpr std::size_t kMaximumStopBytes = 4096;

Status validate_fields(const JsonValue::Object& object) {
  static const std::set<std::string_view> kSupported{
      "model", "prompt", "stream", "stream_options", "max_tokens",
      "max_completion_tokens", "temperature", "top_p", "stop", "seed",
      "logprobs", "top_logprobs", "user", "n", "echo"};
  for (const auto& [name, _] : object) {
    if (!kSupported.contains(name)) {
      return Status::InvalidArgument("openai_completion_field_unsupported");
    }
  }
  return Status::Ok();
}

Status append_prompt(const JsonValue& value,
                     std::vector<std::string>& prompts) {
  if (!value.is_string() || value.string().empty() ||
      value.string().size() > kMaximumPromptBytes) {
    return Status::InvalidArgument("openai_completion_prompt_invalid");
  }
  prompts.push_back(value.string());
  return Status::Ok();
}

Status append_stop(const JsonValue& value, std::vector<std::string>& stop,
                   std::size_t& bytes) {
  if (!value.is_string() || value.string().empty() ||
      value.string().size() > kMaximumStopBytes) {
    return Status::InvalidArgument("openai_completion_stop_invalid");
  }
  if (std::ranges::find(stop, value.string()) == stop.end()) {
    if (bytes > kMaximumStopBytes - value.string().size()) {
      return Status::InvalidArgument("openai_completion_stop_invalid");
    }
    bytes += value.string().size();
    stop.push_back(value.string());
  }
  return Status::Ok();
}

}  // namespace

Result<CompletionRequest> ParseCompletionRequest(
    std::string_view body, std::string_view expected_model,
    std::uint32_t default_maximum_completion_tokens) {
  if (expected_model.empty() || default_maximum_completion_tokens == 0 ||
      default_maximum_completion_tokens > kMaximumCompletionTokens) {
    return Status::InvalidArgument("openai_completion_policy_invalid");
  }
  JsonLimits limits;
  limits.max_input_bytes = 4 * 1024 * 1024;
  limits.max_depth = 8;
  limits.max_nodes = 2048;
  limits.max_string_bytes = kMaximumPromptBytes;
  auto root = JsonValue::Parse(body, limits);
  if (!root.ok() || !root->is_object()) {
    return Status::InvalidArgument("openai_completion_json_invalid");
  }
  auto status = validate_fields(root->object());
  if (!status.ok()) return status;
  const auto* model = root->at("model");
  const auto* prompt = root->at("prompt");
  if (model == nullptr || !model->is_string() ||
      model->string() != expected_model || prompt == nullptr) {
    return Status::InvalidArgument("openai_completion_identity_invalid");
  }
  CompletionRequest result;
  result.model = model->string();
  if (prompt->is_array()) {
    if (prompt->array().empty() ||
        prompt->array().size() > kMaximumPromptCount) {
      return Status::InvalidArgument("openai_completion_prompt_invalid");
    }
    result.prompts.reserve(prompt->array().size());
    for (const auto& item : prompt->array()) {
      status = append_prompt(item, result.prompts);
      if (!status.ok()) return status;
    }
  } else {
    status = append_prompt(*prompt, result.prompts);
    if (!status.ok()) return status;
  }
  const auto* max_tokens = root->at("max_tokens");
  const auto* max_completion_tokens = root->at("max_completion_tokens");
  if (max_tokens != nullptr && max_completion_tokens != nullptr) {
    return Status::InvalidArgument("openai_completion_token_limit_duplicated");
  }
  const auto* token_limit = max_completion_tokens != nullptr
      ? max_completion_tokens
      : max_tokens;
  result.maximum_completion_tokens = default_maximum_completion_tokens;
  if (token_limit != nullptr) {
    if (!token_limit->is_integer() || token_limit->integer() < 1 ||
        static_cast<std::uint64_t>(token_limit->integer()) >
            kMaximumCompletionTokens) {
      return Status::InvalidArgument("openai_completion_token_limit_invalid");
    }
    result.maximum_completion_tokens =
        static_cast<std::uint32_t>(token_limit->integer());
  }
  const auto* stream = root->at("stream");
  if (stream != nullptr) {
    if (!stream->is_boolean()) {
      return Status::InvalidArgument("openai_completion_stream_invalid");
    }
    result.stream = stream->boolean();
  }
  const auto* temperature = root->at("temperature");
  if (temperature != nullptr) {
    if (!temperature->is_number() || !std::isfinite(temperature->number()) ||
        temperature->number() < 0.0 || temperature->number() > 2.0) {
      return Status::InvalidArgument("openai_completion_temperature_invalid");
    }
    result.temperature = temperature->number();
  }
  const auto* top_p = root->at("top_p");
  if (top_p != nullptr) {
    if (!top_p->is_number() || !std::isfinite(top_p->number()) ||
        top_p->number() <= 0.0 || top_p->number() > 1.0) {
      return Status::InvalidArgument("openai_completion_top_p_invalid");
    }
    result.top_p = top_p->number();
  }
  if (result.temperature == 0.0 && result.top_p != 1.0) {
    return Status::InvalidArgument("openai_completion_greedy_filter_invalid");
  }
  const auto* stop = root->at("stop");
  if (stop != nullptr && !stop->is_null()) {
    std::size_t stop_bytes = 0;
    if (stop->is_array()) {
      if (stop->array().size() > kMaximumStopCount) {
        return Status::InvalidArgument("openai_completion_stop_invalid");
      }
      for (const auto& item : stop->array()) {
        status = append_stop(item, result.stop, stop_bytes);
        if (!status.ok()) return status;
      }
    } else {
      status = append_stop(*stop, result.stop, stop_bytes);
      if (!status.ok()) return status;
    }
  }
  const auto* seed = root->at("seed");
  if (seed != nullptr && !seed->is_null()) {
    if (!seed->is_integer() || seed->integer() < 0) {
      return Status::InvalidArgument("openai_completion_seed_invalid");
    }
    result.seed = static_cast<std::uint64_t>(seed->integer());
  }
  const auto* logprobs = root->at("logprobs");
  if (logprobs != nullptr && !logprobs->is_null()) {
    if (!logprobs->is_boolean()) {
      return Status::InvalidArgument("openai_completion_logprobs_invalid");
    }
    result.logprobs = logprobs->boolean();
  }
  const auto* top_logprobs = root->at("top_logprobs");
  if (top_logprobs != nullptr && !top_logprobs->is_null()) {
    if (!top_logprobs->is_integer() || top_logprobs->integer() < 0 ||
        top_logprobs->integer() > 20) {
      return Status::InvalidArgument("openai_completion_top_logprobs_invalid");
    }
    result.top_logprobs =
        static_cast<std::uint32_t>(top_logprobs->integer());
  }
  if (result.top_logprobs != 0 && !result.logprobs) {
    return Status::InvalidArgument("openai_completion_top_logprobs_unbound");
  }
  const auto* stream_options = root->at("stream_options");
  if (stream_options != nullptr && !stream_options->is_null()) {
    if (!result.stream || !stream_options->is_object() ||
        stream_options->object().size() != 1 ||
        stream_options->object().front().first != "include_usage" ||
        !stream_options->object().front().second.is_boolean()) {
      return Status::InvalidArgument("openai_completion_stream_options_invalid");
    }
    result.include_usage =
        stream_options->object().front().second.boolean();
  }
  const auto* n = root->at("n");
  if (n != nullptr && (!n->is_integer() || n->integer() != 1)) {
    return Status::InvalidArgument("openai_completion_n_invalid");
  }
  const auto* echo = root->at("echo");
  if (echo != nullptr && (!echo->is_boolean() || echo->boolean())) {
    return Status::InvalidArgument("openai_completion_echo_invalid");
  }
  const auto* user = root->at("user");
  if (user != nullptr && !user->is_null()) {
    if (!user->is_string() || user->string().empty() ||
        user->string().size() > 128) {
      return Status::InvalidArgument("openai_completion_user_invalid");
    }
    result.user = user->string();
  }
  return result;
}

}  // namespace pih::surface_openai_http
