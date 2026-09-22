// C++ adaptation of encoding/encoding_dsv4.py, DeepSeek-V4-Flash-0731
// revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb.
// Copyright (c) 2023 DeepSeek. MIT: see DEEPSEEK_LICENSE.
#include "deepseek_encoding.h"
#include "text_json.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <unicode/uchar.h>
#include <unicode/utf8.h>

namespace pih::plugin_text {
namespace {
constexpr std::string_view bos = "<｜begin▁of▁sentence｜>", eos = "<｜end▁of▁sentence｜>";
constexpr std::string_view user = "<｜User｜>", assistant = "<｜Assistant｜>";
constexpr std::string_view think = "<think>", end_think = "</think>", dsml = "｜DSML｜";
constexpr std::string_view calls_start = "\n\n<｜DSML｜tool_calls";
constexpr std::string_view effort_high =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
constexpr std::string_view effort_max =
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";
constexpr std::string_view tools_header = R"(## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$TOOL_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

)";
constexpr std::string_view tools_footer =
    "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

void Append(std::string& output, std::string_view part, size_t limit = 1024 * 1024) {
  if (output.size() > limit || part.size() > limit - output.size())
    throw std::invalid_argument("DeepSeek encoding byte budget exceeded");
  output += part;
}
const JsonValue& Field(const JsonValue& object, std::string_view key) {
  const auto* value = object.at(key);
  if (!value) throw std::invalid_argument("DeepSeek encoding field missing: " + std::string(key));
  return *value;
}
std::string String(const JsonValue& object, std::string_view key) {
  const auto* value = object.at(key);
  return !value || value->is_null() ? "" : value->string();
}
bool Truth(const JsonValue* value) {
  if (!value || value->is_null()) return false;
  if (value->is_boolean()) return value->boolean();
  if (value->is_string()) return !value->string().empty();
  if (value->is_array()) return !value->array().empty();
  if (value->is_object()) return !value->object().empty();
  return value->number() != 0;
}
void Set(JsonValue& object, std::string key, JsonValue value) {
  auto fields = object.object();
  for (auto& entry : fields) if (entry.first == key) { entry.second = std::move(value); object = JsonValue(std::move(fields)); return; }
  fields.emplace_back(std::move(key), std::move(value)); object = JsonValue(std::move(fields));
}
JsonValue Text(std::string_view value) { return JsonValue(std::string(value)); }
std::string Role(const JsonValue& message) { return Field(message, "role").string(); }

std::vector<JsonValue> Preprocess(const JsonValue& source) {
  if (!source.is_array() || source.array().empty() || source.array().size() > 128)
    throw std::invalid_argument("DeepSeek messages must contain 1 to 128 entries");
  std::vector<JsonValue> merged;
  for (const auto& message : source.array()) {
    const auto role = Role(message);
    if (role != "tool" && role != "user") { merged.push_back(message); continue; }
    JsonValue block = role == "tool" ? JsonValue(JsonValue::Object{
        {"type", Text("tool_result")}, {"tool_use_id", Text(String(message, "tool_call_id"))},
        {"content", message.at("content") ? *message.at("content") : Text("")}}) :
        JsonValue(JsonValue::Object{{"type", Text("text")}, {"text", Text(String(message, "content"))}});
    if (!merged.empty() && Role(merged.back()) == "user" && merged.back().at("content_blocks") &&
        (role == "tool" || !merged.back().at("task") || merged.back().at("task")->is_null())) {
      auto blocks = Field(merged.back(), "content_blocks").array();
      blocks.push_back(std::move(block)); Set(merged.back(), "content_blocks", JsonValue(std::move(blocks)));
    } else {
      JsonValue next(JsonValue::Object{{"role", Text("user")}});
      if (role == "user") Set(next, "content", Text(String(message, "content")));
      Set(next, "content_blocks", JsonValue(JsonValue::Array{std::move(block)}));
      if (role == "user") for (const auto key : {"task", "wo_eos", "mask"})
        if (message.at(key)) Set(next, key, *message.at(key));
      merged.push_back(std::move(next));
    }
  }
  std::unordered_map<std::string, size_t> order;
  for (auto& message : merged) {
    if (Role(message) == "assistant" && Truth(message.at("tool_calls"))) {
      order.clear(); size_t index = 0;
      for (const auto& call : Field(message, "tool_calls").array()) {
        auto id = String(call, "id");
        if (id.empty() && call.at("function")) id = String(*call.at("function"), "id");
        if (!id.empty()) order[id] = index;
        ++index;
      }
    } else if (Role(message) == "user" && message.at("content_blocks") && !order.empty()) {
      auto blocks = Field(message, "content_blocks").array();
      std::vector<JsonValue> results;
      for (const auto& block : blocks) if (String(block, "type") == "tool_result") results.push_back(block);
      auto rank = [&](const JsonValue& block) { auto it = order.find(String(block, "tool_use_id")); return it == order.end() ? size_t{0} : it->second; };
      std::stable_sort(results.begin(), results.end(), [&](const auto& a, const auto& b) { return rank(a) < rank(b); });
      size_t index = 0;
      for (auto& block : blocks) if (String(block, "type") == "tool_result") block = std::move(results[index++]);
      Set(message, "content_blocks", JsonValue(std::move(blocks)));
    }
  }
  return merged;
}

std::string RenderTools(const JsonValue& tools) {
  if (tools.array().size() > 128) throw std::invalid_argument("too many tools");
  std::string result(tools_header); bool first = true;
  for (const auto& tool : tools.array()) {
    if (!first) Append(result, "\n"); first = false;
    Append(result, TextJson(Field(tool, "function"), 1024 * 1024));
  }
  Append(result, tools_footer); return result;
}
std::string RenderCalls(const JsonValue& calls) {
  if (calls.array().size() > 128) throw std::invalid_argument("too many tool calls");
  std::string result = "\n\n<｜DSML｜tool_calls>\n"; bool first = true;
  for (const auto& call : calls.array()) {
    if (!first) Append(result, "\n"); first = false;
    const auto& function = Field(call, "function");
    Append(result, "<｜DSML｜invoke name=\""); Append(result, Field(function, "name").string()); Append(result, "\">\n");
    const auto arguments = Field(function, "arguments").string();
    auto parsed = JsonValue::Parse(arguments, {1024 * 1024, 64, 16384, 1024 * 1024});
    // Upstream wraps syntactically invalid JSON as a string-valued argument.
    JsonValue values = parsed.ok() ? std::move(*parsed) :
        JsonValue(JsonValue::Object{{"arguments", Text(arguments)}});
    if (!values.is_object()) throw std::invalid_argument("tool arguments must be an object");
    bool first_parameter = true;
    for (const auto& [key, value] : values.object()) {
      if (!first_parameter) Append(result, "\n"); first_parameter = false;
      Append(result, "<｜DSML｜parameter name=\""); Append(result, key);
      Append(result, value.is_string() ? "\" string=\"true\">" : "\" string=\"false\">");
      Append(result, value.is_string() ? value.string() : TextJson(value, 1024 * 1024));
      Append(result, "</｜DSML｜parameter>");
    }
    Append(result, "\n</｜DSML｜invoke>");
  }
  Append(result, "\n</｜DSML｜tool_calls>"); return result;
}
std::string ToolContent(const JsonValue& block) {
  const auto* content = block.at("content");
  if (!content || content->is_null()) return "";
  if (content->is_string()) return content->string();
  std::string result; bool first = true;
  for (const auto& item : content->array()) {
    if (!first) Append(result, "\n\n"); first = false;
    Append(result, String(item, "type") == "text" ? String(item, "text") : "[Unsupported " + String(item, "type") + "]");
  }
  return result;
}
}

std::string EncodeDeepSeekMessages(const JsonValue& messages, const DeepSeekEncodingOptions& options) {
  // Validate size and UTF-8 before building nested copies or prompt fragments.
  (void)TextJson(messages, 1024 * 1024);
  if (options.reasoning_effort != "low" && options.reasoning_effort != "high" && options.reasoning_effort != "max")
    throw std::invalid_argument("invalid reasoning effort");
  auto processed = Preprocess(messages);
  bool drop = options.drop_thinking;
  for (const auto& message : processed) if (Truth(message.at("tools"))) drop = false;
  auto last_user = [](const auto& list) {
    int index = -1;
    for (size_t i = 0; i < list.size(); ++i) if (Role(list[i]) == "user" || Role(list[i]) == "developer") index = static_cast<int>(i);
    return index;
  };
  if (options.thinking && drop) {
    const int last = last_user(processed);
    std::vector<JsonValue> kept;
    for (size_t i = 0; i < processed.size(); ++i) {
      const auto role = Role(processed[i]);
      if (static_cast<int>(i) >= last || role == "user" || role == "system" || role == "tool" ||
          role == "latest_reminder" || role == "direct_search_results" || role == "assistant")
        kept.push_back(processed[i]);
    }
    processed = std::move(kept);
  }
  const int last = last_user(processed);
  std::string result = options.add_bos ? std::string(bos) : "";
  for (size_t i = 0; i < processed.size(); ++i) {
    const auto& message = processed[i]; const auto role = Role(message);
    if (i == 0 && options.thinking) {
      if (options.reasoning_effort == "high") Append(result, effort_high);
      else if (options.reasoning_effort == "max") Append(result, effort_max);
    }
    if (role == "system" || role == "developer") {
      const auto content = String(message, "content");
      if (role == "developer") {
        if (content.empty()) throw std::invalid_argument("empty developer message");
        Append(result, user);
      }
      Append(result, content);
      if (Truth(message.at("tools"))) { Append(result, "\n\n"); Append(result, RenderTools(*message.at("tools"))); }
      if (Truth(message.at("response_format"))) {
        Append(result, "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
        Append(result, TextJson(*message.at("response_format"), 1024 * 1024));
      }
    } else if (role == "user") {
      Append(result, user); bool first = true;
      for (const auto& block : Field(message, "content_blocks").array()) {
        if (!first) Append(result, "\n\n"); first = false;
        if (String(block, "type") == "text") Append(result, String(block, "text"));
        else { Append(result, "<tool_result>"); Append(result, ToolContent(block)); Append(result, "</tool_result>"); }
      }
    } else if (role == "latest_reminder") {
      Append(result, "<｜latest_reminder｜>"); Append(result, String(message, "content"));
    } else if (role == "assistant") {
      const bool previous_task = i != 0 && processed[i - 1].at("task") && !processed[i - 1].at("task")->is_null();
      if (options.thinking && !previous_task && (!drop || static_cast<int>(i) > last)) {
        Append(result, String(message, "reasoning_content")); Append(result, end_think);
      }
      Append(result, String(message, "content"));
      if (Truth(message.at("tool_calls"))) Append(result, RenderCalls(*message.at("tool_calls")));
      if (!Truth(message.at("wo_eos"))) Append(result, eos);
    } else throw std::invalid_argument("unsupported DeepSeek message role");
    if (i + 1 < processed.size() && Role(processed[i + 1]) != "assistant" && Role(processed[i + 1]) != "latest_reminder") continue;
    if (const auto* task = message.at("task"); task && !task->is_null()) {
      const auto name = task->string();
      if (name != "action" && name != "query" && name != "authority" && name != "domain" && name != "title" && name != "read_url")
        throw std::invalid_argument("unsupported DeepSeek task");
      if (name == "action") { Append(result, assistant); Append(result, options.thinking ? think : end_think); }
      Append(result, "<｜" + name + "｜>");
    } else if (role == "user" || role == "developer") {
      Append(result, assistant); Append(result, options.thinking && (!drop || static_cast<int>(i) >= last) ? think : end_think);
    }
  }
  return result;
}

namespace {
struct Segment { std::string_view text, stop; };
Segment Until(std::string_view text, size_t& index, std::initializer_list<std::string_view> stops) {
  size_t end = text.size(); std::string_view stop;
  for (auto candidate : stops) { const auto at = text.find(candidate, index); if (at < end) { end = at; stop = candidate; } }
  auto value = text.substr(index, end - index); index = end + stop.size(); return {value, stop};
}
void Require(bool condition, const char* error) { if (!condition) throw std::invalid_argument(error); }
std::string_view TrimLeadingSpace(std::string_view text) {
  int32_t index = 0;
  while (index < static_cast<int32_t>(text.size())) {
    const int32_t previous = index; UChar32 c;
    U8_NEXT(text.data(), index, static_cast<int32_t>(text.size()), c);
    if (!u_isUWhiteSpace(c) && !(c >= 0x1c && c <= 0x1f)) return text.substr(previous);
  }
  return {};
}
JsonValue::Array ParseCalls(std::string_view text, size_t& index) {
  constexpr std::string_view invoke = "<｜DSML｜invoke", end_invoke = "</｜DSML｜invoke";
  constexpr std::string_view parameter = "<｜DSML｜parameter", end_parameter = "/｜DSML｜parameter";
  constexpr std::string_view end_calls = "</｜DSML｜tool_calls>";
  JsonValue::Array calls;
  while (index < text.size()) {
    auto before = Until(text, index, {invoke, end_calls});
    Require(before.text == ">\n", "invalid DSML call separator");
    if (before.stop == end_calls) return calls;
    Require(before.stop == invoke && calls.size() < 128, "missing DSML invoke or too many calls");
    auto header = Until(text, index, {parameter, end_invoke});
    auto name = TrimLeadingSpace(header.text);
    // The official renderer emits an empty arguments line for a zero-parameter
    // call. Accept it explicitly instead of rejecting our own encoded output.
    if (header.stop == end_invoke && name.ends_with("\">\n\n")) name.remove_suffix(1);
    Require(name.starts_with("name=\"") && name.ends_with("\">\n") && name.size() >= 9, "invalid DSML tool name");
    name.remove_prefix(6); name.remove_suffix(3);
    std::unordered_set<std::string> keys;
    std::string arguments = "{"; bool first = true;
    auto stop = header.stop;
    while (stop == parameter) {
      auto value = Until(text, index, {end_parameter});
      Require(value.stop == end_parameter && value.text.starts_with(" name=\"") && value.text.ends_with("<"), "invalid DSML parameter");
      auto body = value.text.substr(7, value.text.size() - 8);
      constexpr std::string_view string_header = "\" string=\"true\">";
      constexpr std::string_view json_header = "\" string=\"false\">";
      size_t separator = body.find(string_header);
      bool string_value = true;
      const auto other = body.find(json_header);
      if (other < separator) { separator = other; string_value = false; }
      Require(separator != std::string_view::npos && keys.size() < 1024, "invalid DSML parameter header");
      const auto key = body.substr(0, separator);
      Require(keys.emplace(key).second, "duplicate DSML parameter");
      const auto raw = body.substr(separator + (string_value ? string_header.size() : json_header.size()));
      if (!string_value) {
        auto parsed = JsonValue::Parse(raw, {8 * 1024 * 1024, 64, 16384, 8 * 1024 * 1024});
        Require(parsed.ok(), "non-string DSML parameter must contain valid JSON");
      }
      if (!first) Append(arguments, ", ", 8 * 1024 * 1024); first = false;
      Append(arguments, QuoteText(key), 8 * 1024 * 1024); Append(arguments, ": ", 8 * 1024 * 1024);
      Append(arguments, string_value ? QuoteText(raw) : std::string(raw), 8 * 1024 * 1024);
      auto between = Until(text, index, {parameter, end_invoke});
      Require(between.text == ">\n", "invalid DSML parameter separator"); stop = between.stop;
    }
    Require(stop == end_invoke, "missing DSML invoke terminator");
    Append(arguments, "}", 8 * 1024 * 1024);
    calls.emplace_back(JsonValue::Object{{"type", Text("function")}, {"function", JsonValue(JsonValue::Object{
        {"name", Text(name)}, {"arguments", Text(arguments)}})}});
  }
  throw std::invalid_argument("missing DSML tool_calls terminator");
}
}

JsonValue ParseDeepSeekCompletion(std::string_view text, bool thinking) {
  RequireUtf8(text);
  size_t index = 0; std::string reasoning;
  if (thinking) {
    auto thought = Until(text, index, {end_think, calls_start});
    Require(thought.stop == end_think, "missing thinking terminator"); reasoning = thought.text;
  }
  auto summary = Until(text, index, {eos, calls_start});
  JsonValue::Array calls;
  if (summary.stop == calls_start) {
    calls = ParseCalls(text, index);
    auto ending = Until(text, index, {eos});
    Require(ending.text.empty() && ending.stop == eos, "missing EOS or trailing text after tool calls");
  } else Require(summary.stop == eos, "missing completion EOS");
  Require(index == text.size(), "unexpected trailing completion text");
  for (auto token : {bos, eos, think, end_think, dsml})
    Require(summary.text.find(token) == std::string_view::npos && reasoning.find(token) == std::string::npos,
        "unexpected structural token in completion content");
  return JsonValue(JsonValue::Object{{"role", Text("assistant")}, {"content", Text(summary.text)},
      {"reasoning_content", Text(reasoning)}, {"tool_calls", JsonValue(std::move(calls))}});
}
}
