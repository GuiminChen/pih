// CPU-only rendering observation, not model/artifact admission or generation.
#include "chat_tokens.h"
#include "../common/text_json.h"
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2 && argc != 4) {
    std::cerr << "usage: pih-qwen-format REQUEST.json\n"
                 "   or: pih-qwen-format REQUEST.json --tokens TOKENIZER.json\n"
                 "request: {\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}\n";
    return 2;
  }
  try {
    if (argc == 4 && std::string_view(argv[2]) != "--tokens")
      throw std::invalid_argument("expected --tokens TOKENIZER.json");
    const auto path = std::filesystem::canonical(argv[1]);
    const auto size = std::filesystem::file_size(path);
    if (!size || size > 1024 * 1024)
      throw std::invalid_argument("request must be 1 byte to 1 MiB");
    std::ifstream input(path, std::ios::binary);
    std::string bytes(static_cast<size_t>(size), '\0');
    if (!input.read(bytes.data(), bytes.size()) || input.peek() != std::char_traits<char>::eof())
      throw std::invalid_argument("request read failed or size changed");
    auto request = pih::JsonValue::Parse(bytes, {1024 * 1024, 8, 2048, 1024 * 1024});
    if (!request.ok() || !request->is_object() || request->object().size() != 1 ||
        !request->at("messages"))
      throw std::invalid_argument("request requires only messages");
    const auto plan = pih::qwen_plugin::PlainChatPlan(*request->at("messages"));
    std::string rendered;
    for (const auto& segment : plan) rendered += segment.text;
    std::string output = "{\"text\":" + pih::plugin_text::QuoteText(rendered);
    if (argc == 4) {
      const auto tokenizer = pih::qwen_plugin::LoadPinnedTokenizer(argv[3]);
      const auto tokens = pih::qwen_plugin::PlainChatTokens(*request->at("messages"), tokenizer, 65536);
      output += ",\"tokens\":[";
      for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) output += ',';
        output += std::to_string(tokens[i]);
      }
      output += ']';
    }
    output += "}\n";
    if (output.size() > 8 * 1024 * 1024)
      throw std::invalid_argument("format output exceeds byte bound");
    std::cout << output;
    if (!std::cout) throw std::runtime_error("format output write failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native Qwen format failed: " << error.what() << '\n';
    return 2;
  }
}
