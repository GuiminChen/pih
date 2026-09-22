#include "pih/core/bounded_json.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <system_error>

namespace pih {
namespace {

class Parser final {
 public:
  Parser(std::string_view input, JsonLimits limits)
      : input_(input), limits_(limits) {}

  Result<JsonValue> parse() {
    if (input_.size() > limits_.max_input_bytes) {
      return Status::ResourceExhausted("JSON input exceeds byte budget");
    }
    skip_space();
    auto value = parse_value(1);
    if (!value.ok()) return value.status();
    skip_space();
    if (position_ != input_.size()) {
      return Status::InvalidArgument("JSON has trailing data");
    }
    return value;
  }

 private:
  Result<JsonValue> parse_value(std::size_t depth) {
    if (depth > limits_.max_depth) {
      return Status::ResourceExhausted("JSON nesting exceeds depth budget");
    }
    if (++nodes_ > limits_.max_nodes) {
      return Status::ResourceExhausted("JSON exceeds node budget");
    }
    if (position_ == input_.size()) {
      return Status::InvalidArgument("unexpected end of JSON");
    }
    switch (input_[position_]) {
      case 'n': return literal("null", JsonValue(nullptr));
      case 't': return literal("true", JsonValue(true));
      case 'f': return literal("false", JsonValue(false));
      case '"': {
        auto text = parse_string();
        if (!text.ok()) return text.status();
        return JsonValue(std::move(text).value());
      }
      case '[': return parse_array(depth);
      case '{': return parse_object(depth);
      default: return parse_number();
    }
  }

  Result<JsonValue> literal(std::string_view token, JsonValue value) {
    if (input_.substr(position_, token.size()) != token) {
      return Status::InvalidArgument("invalid JSON literal");
    }
    position_ += token.size();
    return value;
  }

  Result<JsonValue> parse_array(std::size_t depth) {
    ++position_;
    skip_space();
    JsonValue::Array values;
    if (consume(']')) return JsonValue(std::move(values));
    while (true) {
      auto value = parse_value(depth + 1);
      if (!value.ok()) return value.status();
      values.push_back(std::move(value).value());
      skip_space();
      if (consume(']')) return JsonValue(std::move(values));
      if (!consume(',')) return Status::InvalidArgument("JSON array requires comma");
      skip_space();
    }
  }

  Result<JsonValue> parse_object(std::size_t depth) {
    ++position_;
    skip_space();
    JsonValue::Object values;
    if (consume('}')) return finish_object(std::move(values));
    while (true) {
      if (position_ == input_.size() || input_[position_] != '"') {
        return Status::InvalidArgument("JSON object key must be a string");
      }
      auto key = parse_string();
      if (!key.ok()) return key.status();
      skip_space();
      if (!consume(':')) return Status::InvalidArgument("JSON object requires colon");
      skip_space();
      auto value = parse_value(depth + 1);
      if (!value.ok()) return value.status();
      values.emplace_back(std::move(key).value(), std::move(value).value());
      skip_space();
      if (consume('}')) return finish_object(std::move(values));
      if (!consume(',')) return Status::InvalidArgument("JSON object requires comma");
      skip_space();
    }
  }

  Result<JsonValue> finish_object(JsonValue::Object values) {
    std::sort(values.begin(), values.end(),
              [](const auto& left, const auto& right) {
                return left.first < right.first;
              });
    for (std::size_t index = 1; index < values.size(); ++index) {
      if (values[index - 1].first == values[index].first) {
        return Status::InvalidArgument("duplicate JSON object key");
      }
    }
    return JsonValue(std::move(values));
  }

  Result<JsonValue> parse_number() {
    const std::size_t begin = position_;
    if (consume('-') && position_ == input_.size()) {
      return Status::InvalidArgument("incomplete JSON number");
    }
    if (consume('0')) {
      if (position_ < input_.size() && input_[position_] >= '0' &&
          input_[position_] <= '9') {
        return Status::InvalidArgument("JSON number has a leading zero");
      }
    } else {
      if (position_ == input_.size() || input_[position_] < '1' ||
          input_[position_] > '9') {
        return Status::InvalidArgument("invalid JSON number");
      }
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') ++position_;
    }
    bool integral = true;
    if (consume('.')) {
      integral = false;
      const auto fraction = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') ++position_;
      if (fraction == position_) return Status::InvalidArgument("empty JSON fraction");
    }
    if (position_ < input_.size() &&
        (input_[position_] == 'e' || input_[position_] == 'E')) {
      integral = false;
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-')) ++position_;
      const auto exponent = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') ++position_;
      if (exponent == position_) return Status::InvalidArgument("empty JSON exponent");
    }
    const auto token = input_.substr(begin, position_ - begin);
    if (integral) {
      std::int64_t value = 0;
      const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
      if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) {
        return Status::InvalidArgument("JSON integer is out of range");
      }
      return JsonValue(value);
    }
    double value = 0;
    const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size() ||
        !std::isfinite(value)) {
      return Status::InvalidArgument("JSON number is out of range");
    }
    return JsonValue(value);
  }

  Result<std::string> parse_string() {
    ++position_;
    std::string output;
    while (position_ < input_.size()) {
      const unsigned char byte = static_cast<unsigned char>(input_[position_++]);
      if (byte == '"') return output;
      if (byte < 0x20) return Status::InvalidArgument("control byte in JSON string");
      if (byte != '\\') {
        if (byte < 0x80) {
          output.push_back(static_cast<char>(byte));
        } else {
          const auto status = append_raw_utf8(byte, output);
          if (!status.ok()) return status;
        }
      } else {
        if (position_ == input_.size()) return Status::InvalidArgument("truncated escape");
        const char escape = input_[position_++];
        switch (escape) {
          case '"': case '\\': case '/': output.push_back(escape); break;
          case 'b': output.push_back('\b'); break;
          case 'f': output.push_back('\f'); break;
          case 'n': output.push_back('\n'); break;
          case 'r': output.push_back('\r'); break;
          case 't': output.push_back('\t'); break;
          case 'u': {
            auto codepoint = parse_codepoint();
            if (!codepoint.ok()) return codepoint.status();
            append_utf8(codepoint.value(), output);
            break;
          }
          default: return Status::InvalidArgument("invalid JSON escape");
        }
      }
      if (output.size() > limits_.max_string_bytes) {
        return Status::ResourceExhausted("JSON string exceeds byte budget");
      }
    }
    return Status::InvalidArgument("unterminated JSON string");
  }

  Status append_raw_utf8(unsigned char lead, std::string& output) {
    std::size_t continuation_bytes = 0;
    unsigned char second_minimum = 0x80;
    unsigned char second_maximum = 0xBF;
    if (lead >= 0xC2 && lead <= 0xDF) {
      continuation_bytes = 1;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
      continuation_bytes = 2;
      if (lead == 0xE0) second_minimum = 0xA0;
      if (lead == 0xED) second_maximum = 0x9F;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
      continuation_bytes = 3;
      if (lead == 0xF0) second_minimum = 0x90;
      if (lead == 0xF4) second_maximum = 0x8F;
    } else {
      return Status::InvalidArgument("invalid UTF-8 leading byte in JSON string");
    }
    if (input_.size() - position_ < continuation_bytes) {
      return Status::InvalidArgument("truncated UTF-8 sequence in JSON string");
    }
    const auto second = static_cast<unsigned char>(input_[position_]);
    if (second < second_minimum || second > second_maximum) {
      return Status::InvalidArgument("invalid UTF-8 scalar in JSON string");
    }
    for (std::size_t offset = 1; offset < continuation_bytes; ++offset) {
      const auto continuation =
          static_cast<unsigned char>(input_[position_ + offset]);
      if (continuation < 0x80 || continuation > 0xBF) {
        return Status::InvalidArgument(
            "invalid UTF-8 continuation byte in JSON string");
      }
    }
    output.push_back(static_cast<char>(lead));
    output.append(input_.data() + position_, continuation_bytes);
    position_ += continuation_bytes;
    return Status::Ok();
  }

  Result<std::uint32_t> parse_codepoint() {
    auto first = parse_hex4();
    if (!first.ok()) return first.status();
    std::uint32_t value = first.value();
    if (value >= 0xD800 && value <= 0xDBFF) {
      if (position_ + 2 > input_.size() || input_[position_] != '\\' ||
          input_[position_ + 1] != 'u') {
        return Status::InvalidArgument("unpaired high surrogate");
      }
      position_ += 2;
      auto second = parse_hex4();
      if (!second.ok() || second.value() < 0xDC00 || second.value() > 0xDFFF) {
        return Status::InvalidArgument("invalid low surrogate");
      }
      value = 0x10000 + ((value - 0xD800) << 10) + (second.value() - 0xDC00);
    } else if (value >= 0xDC00 && value <= 0xDFFF) {
      return Status::InvalidArgument("unpaired low surrogate");
    }
    return value;
  }

  Result<std::uint32_t> parse_hex4() {
    if (position_ + 4 > input_.size()) return Status::InvalidArgument("truncated unicode escape");
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = input_[position_++];
      value <<= 4;
      if (c >= '0' && c <= '9') value += c - '0';
      else if (c >= 'a' && c <= 'f') value += c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') value += c - 'A' + 10;
      else return Status::InvalidArgument("invalid unicode escape");
    }
    return value;
  }

  static void append_utf8(std::uint32_t value, std::string& output) {
    if (value <= 0x7F) output.push_back(static_cast<char>(value));
    else if (value <= 0x7FF) {
      output.push_back(static_cast<char>(0xC0 | (value >> 6)));
      output.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    } else if (value <= 0xFFFF) {
      output.push_back(static_cast<char>(0xE0 | (value >> 12)));
      output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
      output.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    } else {
      output.push_back(static_cast<char>(0xF0 | (value >> 18)));
      output.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3F)));
      output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
      output.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    }
  }

  void skip_space() {
    while (position_ < input_.size() &&
           (input_[position_] == ' ' || input_[position_] == '\n' ||
            input_[position_] == '\r' || input_[position_] == '\t')) ++position_;
  }
  bool consume(char value) {
    if (position_ < input_.size() && input_[position_] == value) {
      ++position_;
      return true;
    }
    return false;
  }

  std::string_view input_;
  JsonLimits limits_;
  std::size_t position_ = 0;
  std::size_t nodes_ = 0;
};

}  // namespace

Result<JsonValue> JsonValue::Parse(std::string_view input, JsonLimits limits) {
  if (limits.max_depth == 0 || limits.max_nodes == 0) {
    return Status::InvalidArgument("JSON budgets must be nonzero");
  }
  return Parser(input, limits).parse();
}

bool JsonValue::is_null() const noexcept { return std::holds_alternative<std::nullptr_t>(storage_); }
bool JsonValue::is_boolean() const noexcept { return std::holds_alternative<bool>(storage_); }
bool JsonValue::is_integer() const noexcept { return std::holds_alternative<std::int64_t>(storage_); }
bool JsonValue::is_number() const noexcept { return is_integer() || std::holds_alternative<double>(storage_); }
bool JsonValue::is_string() const noexcept { return std::holds_alternative<std::string>(storage_); }
bool JsonValue::is_array() const noexcept { return std::holds_alternative<Array>(storage_); }
bool JsonValue::is_object() const noexcept { return std::holds_alternative<Object>(storage_); }
bool JsonValue::boolean() const { return std::get<bool>(storage_); }
std::int64_t JsonValue::integer() const { return std::get<std::int64_t>(storage_); }
double JsonValue::number() const { return is_integer() ? static_cast<double>(integer()) : std::get<double>(storage_); }
const std::string& JsonValue::string() const { return std::get<std::string>(storage_); }
const JsonValue::Array& JsonValue::array() const { return std::get<Array>(storage_); }
const JsonValue::Object& JsonValue::object() const { return std::get<Object>(storage_); }
const JsonValue* JsonValue::at(std::string_view key) const noexcept {
  if (!is_object()) return nullptr;
  for (const auto& entry : object()) if (entry.first == key) return &entry.second;
  return nullptr;
}

}  // namespace pih
