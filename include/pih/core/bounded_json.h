#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "pih/core/result.h"

namespace pih {

struct JsonLimits final {
  std::size_t max_input_bytes = 16 * 1024 * 1024;
  std::size_t max_depth = 64;
  std::size_t max_nodes = 65'536;
  std::size_t max_string_bytes = 1024 * 1024;
};

class JsonValue final {
 public:
  using Array = std::vector<JsonValue>;
  using Object = std::vector<std::pair<std::string, JsonValue>>;

  static Result<JsonValue> Parse(std::string_view input,
                                 JsonLimits limits = {});

  [[nodiscard]] bool is_null() const noexcept;
  [[nodiscard]] bool is_boolean() const noexcept;
  [[nodiscard]] bool is_integer() const noexcept;
  [[nodiscard]] bool is_number() const noexcept;
  [[nodiscard]] bool is_string() const noexcept;
  [[nodiscard]] bool is_array() const noexcept;
  [[nodiscard]] bool is_object() const noexcept;

  [[nodiscard]] bool boolean() const;
  [[nodiscard]] std::int64_t integer() const;
  [[nodiscard]] double number() const;
  [[nodiscard]] const std::string& string() const;
  [[nodiscard]] const Array& array() const;
  [[nodiscard]] const Object& object() const;
  [[nodiscard]] const JsonValue* at(std::string_view key) const noexcept;

  explicit JsonValue(std::nullptr_t) : storage_(nullptr) {}
  explicit JsonValue(bool value) : storage_(value) {}
  explicit JsonValue(std::int64_t value) : storage_(value) {}
  explicit JsonValue(double value) : storage_(value) {}
  explicit JsonValue(std::string value) : storage_(std::move(value)) {}
  explicit JsonValue(Array value) : storage_(std::move(value)) {}
  explicit JsonValue(Object value) : storage_(std::move(value)) {}

 private:
  std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Array,
               Object>
      storage_;
};

}  // namespace pih
