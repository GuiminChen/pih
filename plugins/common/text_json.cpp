#include "text_json.h"
#include <charconv>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <unicode/unistr.h>

namespace pih::plugin_text {
void RequireUtf8(std::string_view text) {
  if (text.size() > 8 * 1024 * 1024) throw std::invalid_argument("text exceeds 8 MiB");
  std::string copy;
  icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(), static_cast<int32_t>(text.size()))).toUTF8String(copy);
  if (copy != text) throw std::invalid_argument("text is not valid UTF-8");
}

std::string QuoteText(std::string_view text, size_t maximum_bytes) {
  RequireUtf8(text);
  std::string output;
  auto append = [&](std::string_view part) {
    if (part.size() > maximum_bytes - output.size()) throw std::invalid_argument("JSON byte budget exceeded");
    output += part;
  };
  append("\"");
  for (const unsigned char c : text) {
    switch (c) {
      case '"': append("\\\""); break;
      case '\\': append("\\\\"); break;
      case '\b': append("\\b"); break;
      case '\f': append("\\f"); break;
      case '\n': append("\\n"); break;
      case '\r': append("\\r"); break;
      case '\t': append("\\t"); break;
      default:
        if (c < 32) {
          constexpr char hex[] = "0123456789abcdef";
          const char escaped[]{'\\', 'u', '0', '0', hex[c >> 4], hex[c & 15]};
          append(std::string_view(escaped, sizeof(escaped)));
        } else { const char byte = static_cast<char>(c); append(std::string_view(&byte, 1)); }
    }
  }
  append("\"");
  return output;
}

namespace {
std::string FloatText(double value) {
  if (!std::isfinite(value)) throw std::invalid_argument("non-finite JSON number");
  char buffer[128];
  const auto converted = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific);
  if (converted.ec != std::errc{}) throw std::runtime_error("JSON float conversion failed");
  std::string scientific(buffer, converted.ptr);
  const auto e = scientific.find('e');
  const auto exponent = std::stoi(scientific.substr(e + 1));
  // Python repr uses fixed notation for decimal exponents -4 through 15.
  if (exponent < -4 || exponent >= 16) return scientific;
  const bool negative = scientific[0] == '-';
  auto digits = scientific.substr(negative ? 1 : 0, e - (negative ? 1 : 0));
  const auto dot = digits.find('.');
  if (dot != std::string::npos) digits.erase(dot, 1);
  const int position = exponent + 1;
  std::string result = negative ? "-" : "";
  if (position <= 0) result += "0." + std::string(static_cast<size_t>(-position), '0') + digits;
  else if (static_cast<size_t>(position) >= digits.size())
    result += digits + std::string(static_cast<size_t>(position) - digits.size(), '0') + ".0";
  else result += digits.substr(0, position) + "." + digits.substr(position);
  return result;
}
}

std::string TextJson(const JsonValue& value, size_t maximum_bytes) {
  std::string output;
  size_t nodes = 0;
  auto append = [&](std::string_view part) {
    if (part.size() > maximum_bytes - output.size()) throw std::invalid_argument("JSON byte budget exceeded");
    output += part;
  };
  std::function<void(const JsonValue&, size_t)> write = [&](const JsonValue& item, size_t depth) {
    if (depth > 64 || ++nodes > 65536) throw std::invalid_argument("JSON structure budget exceeded");
    if (item.is_null()) append("null");
    else if (item.is_boolean()) append(item.boolean() ? "true" : "false");
    else if (item.is_integer()) append(std::to_string(item.integer()));
    else if (item.is_number()) append(FloatText(item.number()));
    else if (item.is_string()) append(QuoteText(item.string(), maximum_bytes - output.size()));
    else if (item.is_array()) {
      append("["); bool first = true;
      for (const auto& child : item.array()) { if (!first) append(", "); first = false; write(child, depth + 1); }
      append("]");
    } else {
      append("{"); bool first = true;
      for (const auto& [key, child] : item.object()) {
        if (!first) append(", "); first = false;
        append(QuoteText(key, maximum_bytes - output.size())); append(": "); write(child, depth + 1);
      }
      append("}");
    }
  };
  write(value, 1);
  return output;
}
}
