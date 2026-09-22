#include "pih/core/canonical_json.h"

#include <algorithm>
#include <charconv>

namespace pih {
namespace {

class Writer final {
 public:
  Writer(std::size_t maximum_bytes,
         std::span<const std::string_view> omitted_root_fields)
      : maximum_bytes_(maximum_bytes),
        omitted_root_fields_(omitted_root_fields) {}

  Result<std::string> write(const JsonValue& value) {
    if (maximum_bytes_ == 0) {
      return Status::InvalidArgument(
          "canonical JSON byte budget must be nonzero");
    }
    auto status = append_value(value, true);
    if (!status.ok()) return status;
    if (output_.empty()) {
      return Status::InvalidArgument("canonical JSON output is empty");
    }
    return std::move(output_);
  }

 private:
  Status append(std::string_view source) {
    if (source.size() > maximum_bytes_ - output_.size()) {
      return Status::ResourceExhausted(
          "canonical JSON exceeds byte budget");
    }
    output_.append(source);
    return Status::Ok();
  }

  Status append_character(char value) {
    return append(std::string_view(&value, 1));
  }

  Status append_string(std::string_view value) {
    auto status = append_character('"');
    if (!status.ok()) return status;
    static constexpr char kHex[] = "0123456789abcdef";
    for (const unsigned char byte : value) {
      if (byte > 0x7f) {
        return Status::InvalidArgument(
            "canonical authority JSON must be ASCII");
      }
      switch (byte) {
        case '"': status = append("\\\""); break;
        case '\\': status = append("\\\\"); break;
        case '\b': status = append("\\b"); break;
        case '\f': status = append("\\f"); break;
        case '\n': status = append("\\n"); break;
        case '\r': status = append("\\r"); break;
        case '\t': status = append("\\t"); break;
        default:
          if (byte < 0x20) {
            char escaped[6]{'\\', 'u', '0', '0', kHex[byte >> 4],
                            kHex[byte & 0x0f]};
            status = append(std::string_view(escaped, sizeof(escaped)));
          } else {
            status = append_character(static_cast<char>(byte));
          }
      }
      if (!status.ok()) return status;
    }
    return append_character('"');
  }

  bool omitted(std::string_view key) const noexcept {
    return std::ranges::find(omitted_root_fields_, key) !=
           omitted_root_fields_.end();
  }

  Status append_value(const JsonValue& value, bool root) {
    if (value.is_null()) return append("null");
    if (value.is_boolean()) return append(value.boolean() ? "true" : "false");
    if (value.is_integer()) {
      char buffer[32]{};
      const auto converted = std::to_chars(
          buffer, buffer + sizeof(buffer), value.integer());
      if (converted.ec != std::errc{}) {
        return Status::Internal("cannot encode canonical JSON integer");
      }
      return append(std::string_view(
          buffer, static_cast<std::size_t>(converted.ptr - buffer)));
    }
    if (value.is_number()) {
      return Status::InvalidArgument(
          "canonical authority JSON forbids floating-point numbers");
    }
    if (value.is_string()) return append_string(value.string());
    if (value.is_array()) {
      auto status = append_character('[');
      if (!status.ok()) return status;
      for (std::size_t index = 0; index < value.array().size(); ++index) {
        if (index != 0) {
          status = append_character(',');
          if (!status.ok()) return status;
        }
        status = append_value(value.array()[index], false);
        if (!status.ok()) return status;
      }
      return append_character(']');
    }
    if (!value.is_object()) {
      return Status::InvalidArgument("unknown canonical JSON value");
    }
    auto status = append_character('{');
    if (!status.ok()) return status;
    bool first = true;
    for (const auto& [key, member] : value.object()) {
      if (root && omitted(key)) continue;
      if (!first) {
        status = append_character(',');
        if (!status.ok()) return status;
      }
      first = false;
      status = append_string(key);
      if (status.ok()) status = append_character(':');
      if (status.ok()) status = append_value(member, false);
      if (!status.ok()) return status;
    }
    return append_character('}');
  }

  std::size_t maximum_bytes_ = 0;
  std::span<const std::string_view> omitted_root_fields_;
  std::string output_;
};

}  // namespace

Result<std::string> canonical_ascii_json(
    const JsonValue& value, std::size_t maximum_bytes,
    std::span<const std::string_view> omitted_root_fields) {
  if (!value.is_object() && !omitted_root_fields.empty()) {
    return Status::InvalidArgument(
        "canonical JSON root omissions require an object");
  }
  for (std::size_t index = 0; index < omitted_root_fields.size(); ++index) {
    if (omitted_root_fields[index].empty()) {
      return Status::InvalidArgument(
          "canonical JSON omitted field is empty");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (omitted_root_fields[prior] == omitted_root_fields[index]) {
        return Status::InvalidArgument(
            "canonical JSON omitted field is duplicated");
      }
    }
  }
  return Writer(maximum_bytes, omitted_root_fields).write(value);
}

}  // namespace pih
