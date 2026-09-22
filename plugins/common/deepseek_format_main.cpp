#include "deepseek_encoding.h"
#include "deepseek_completion_decoder.h"
#include "text_json.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: pih-deepseek-format REQUEST.json\n"
      "operation encode: messages, thinking_mode, drop_thinking, add_bos, reasoning_effort\n"
      "operation parse: text (complete output including EOS), thinking_mode\n"
      "operation decode: chunks (UTF-8 text fragments), stopped, thinking_mode\n";
    return 2;
  }
  try {
    using namespace pih::plugin_text;
    const auto path = std::filesystem::canonical(argv[1]);
    const auto size = std::filesystem::file_size(path);
    if (!size || size > 8 * 1024 * 1024) throw std::invalid_argument("request size invalid");
    std::ifstream input(path, std::ios::binary);
    std::string bytes(static_cast<size_t>(size), '\0');
    if (!input.read(bytes.data(), bytes.size())) throw std::invalid_argument("request read failed");
    auto parsed = pih::JsonValue::Parse(bytes, {8 * 1024 * 1024, 64, 65536, 8 * 1024 * 1024});
    if (!parsed.ok() || !parsed->is_object()) throw std::invalid_argument("request JSON invalid");
    const auto* operation = parsed->at("operation");
    if (!operation || !operation->is_string() ||
        (operation->string() != "encode" && operation->string() != "parse" && operation->string() != "decode"))
      throw std::invalid_argument("operation must be encode, parse or decode");
    const bool encode = operation->string() == "encode";
    const bool decode = operation->string() == "decode";
    for (const auto& [key, value] : parsed->object()) {
      if (key == "operation" || key == "thinking_mode") continue;
      if (encode && (key == "messages" || key == "drop_thinking" || key == "add_bos" || key == "reasoning_effort")) continue;
      if (!encode && !decode && key == "text") continue;
      if (decode && (key == "chunks" || key == "stopped")) continue;
      throw std::invalid_argument("unknown request field: " + key);
    }
    std::string mode = "chat";
    if (const auto* field = parsed->at("thinking_mode")) mode = field->string();
    if (mode != "chat" && mode != "thinking") throw std::invalid_argument("thinking_mode must be chat or thinking");
    std::string output;
    if (encode) {
      DeepSeekEncodingOptions options;
      options.thinking = mode == "thinking";
      if (const auto* field = parsed->at("drop_thinking")) options.drop_thinking = field->boolean();
      if (const auto* field = parsed->at("add_bos")) options.add_bos = field->boolean();
      if (const auto* field = parsed->at("reasoning_effort")) options.reasoning_effort = field->string();
      const auto* messages = parsed->at("messages");
      if (!messages) throw std::invalid_argument("encode requires messages");
      output = "{\"text\":" + QuoteText(EncodeDeepSeekMessages(*messages, options)) + "}";
    } else if (decode) {
      const auto* chunks = parsed->at("chunks");
      const auto* stopped = parsed->at("stopped");
      if (!chunks || !chunks->is_array() || chunks->array().size() > 65536 || !stopped || !stopped->is_boolean())
        throw std::invalid_argument("decode requires chunks (at most 65536 UTF-8 strings) and boolean stopped");
      DeepSeekCompletionDecoder decoder(mode == "thinking");
      pih::JsonValue::Array deltas;
      for (const auto& chunk : chunks->array()) {
        auto delta = decoder.Feed(chunk.string());
        if (!delta.object().empty()) deltas.push_back(std::move(delta));
      }
      auto last = decoder.Finish(stopped->boolean());
      if (!last.object().empty()) deltas.push_back(std::move(last));
      output = TextJson(pih::JsonValue(pih::JsonValue::Object{
          {"deltas", pih::JsonValue(std::move(deltas))}, {"message", decoder.Message()}}));
    } else {
      const auto* text = parsed->at("text");
      if (!text || !text->is_string()) throw std::invalid_argument("parse requires text");
      output = TextJson(ParseDeepSeekCompletion(text->string(), mode == "thinking"));
    }
    std::cout << output << '\n';
    if (!std::cout) throw std::runtime_error("output write failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native DeepSeek format failed: " << error.what() << '\n'; return 2;
  }
}
