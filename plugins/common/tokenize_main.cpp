// Offline format conversion only; no artifact admission or model execution.
#include "bytelevel_tokenizer.h"
#include "deepseek_semantic_artifact.h"
#include "pih/core/bounded_json.h"
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace {
std::string Quote(std::string_view input) {
  constexpr char hex[] = "0123456789abcdef";
  std::string output = "\"";
  for (const unsigned char c : input) {
    if (c == '"' || c == '\\') { output += '\\'; output += static_cast<char>(c); }
    else if (c < 32) { output += "\\u00"; output += hex[c >> 4]; output += hex[c & 15]; }
    else output += static_cast<char>(c);
  }
  output += '"';
  return output;
}
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: pih-tokenize qwen3|deepseek-v4-flash-0731|deepseek-v41-flash TOKENIZER.json REQUEST.json\n"
      "   or: pih-tokenize deepseek-v4-flash-0731-verified SNAPSHOT_DIR REQUEST.json\n"
      "request: {\"text\":\"raw prompt\"} or {\"tokens\":[0,1,2]}\n";
    return 2;
  }
  try {
    using pih::plugin_text::TokenizerFamily;
    const std::string family = argv[1];
    const bool verified = family == "deepseek-v4-flash-0731-verified";
    if (family != "qwen3" && family != "deepseek-v4-flash-0731" && family != "deepseek-v41-flash" && !verified)
      throw std::invalid_argument("unsupported tokenizer family");
    const auto path = std::filesystem::canonical(argv[3]);
    const auto size = std::filesystem::file_size(path);
    if (!size || size > 1024 * 1024) throw std::invalid_argument("request must be 1 byte to 1 MiB");
    std::ifstream input(path, std::ios::binary);
    std::string bytes(static_cast<size_t>(size), '\0');
    if (!input.read(bytes.data(), bytes.size())) throw std::invalid_argument("request read failed");
    auto request = pih::JsonValue::Parse(bytes, {1024 * 1024, 4, 65540, 1024 * 1024});
    if (!request.ok() || !request->is_object() || request->object().size() != 1)
      throw std::invalid_argument("request must be a one-field JSON object");
    const auto* text = request->at("text");
    const auto* tokens = request->at("tokens");
    if ((!text || !text->is_string()) && (!tokens || !tokens->is_array()))
      throw std::invalid_argument("request requires string text or integer array tokens");
    std::optional<pih::plugin_text::DeepSeekSemanticArtifacts> admitted;
    std::optional<pih::plugin_text::Tokenizer> raw;
    if (verified) admitted.emplace(pih::plugin_text::DeepSeekSemanticArtifacts::Load(argv[2]));
    else raw.emplace(std::filesystem::canonical(argv[2]),
        family == "qwen3" ? TokenizerFamily::Qwen3 :
        family == "deepseek-v41-flash" ? TokenizerFamily::DeepSeekV41Flash : TokenizerFamily::DeepSeekV4Flash0731);
    const auto& tokenizer = admitted ? admitted->tokenizer() : *raw;
    std::string output;
    if (text) {
      auto ids = tokenizer.Encode(text->string());
      if (ids.size() > 65536) throw std::invalid_argument("encoded token count exceeds 65536");
      output = "{\"tokens\":[";
      for (size_t i = 0; i < ids.size(); ++i) {
        if (i) output += ',';
        output += std::to_string(ids[i]);
      }
      output += "]}";
    } else {
      if (tokens->array().size() > 65536) throw std::invalid_argument("token count exceeds 65536");
      std::vector<int64_t> ids;
      for (const auto& id : tokens->array()) {
        if (!id.is_integer() || id.integer() < 0 ||
            id.integer() >= (family == "qwen3" ? 151936 : 129280))
          throw std::invalid_argument("token id out of range");
        ids.push_back(id.integer());
      }
      output = "{\"text\":" + Quote(tokenizer.Decode(ids)) + "}";
    }
    if (output.size() > 8 * 1024 * 1024) throw std::invalid_argument("output exceeds 8 MiB");
    std::cout << output << '\n';
    if (!std::cout) throw std::runtime_error("output write failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native tokenizer failed: " << error.what() << '\n';
    return 2;
  }
}
