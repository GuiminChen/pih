#pragma once

#include "pih/core/bounded_json.h"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <utility>

namespace pih::qwen_plugin {
// Feed complete UTF-8 decoder fragments. Retain potential stop prefixes so
// neither output mode publishes a partial stop marker before disambiguation.
class TextStop final {
 public:
  explicit TextStop(const JsonValue* field) {
    if (!field || field->is_null()) return;
    auto add = [&](const JsonValue& value) {
      if (!value.is_string() || value.string().empty() || value.string().size() > 256)
        throw std::invalid_argument("stop strings must contain 1..256 UTF-8 bytes");
      if (std::find(stops_.begin(), stops_.end(), value.string()) != stops_.end())
        throw std::invalid_argument("duplicate stop string");
      stops_.push_back(value.string());
    };
    if (field->is_string()) add(*field);
    else {
      if (!field->is_array() || field->array().empty() || field->array().size() > 4)
        throw std::invalid_argument("stop must be null, a string or 1..4 strings");
      for (const auto& item : field->array()) add(item);
    }
  }
  std::string Feed(std::string_view text) {
    if (finished_) throw std::invalid_argument("stop filter already finalized");
    if (matched_) return {};
    if (text.size() > 8 * 1024 * 1024 - pending_.size())
      throw std::invalid_argument("stop-filter input exceeds byte bound");
    pending_.append(text);
    size_t first = std::string::npos, earliest_end = std::string::npos;
    // Earliest completed byte match, then earliest start, is independent of
    // decoder chunk boundaries even when one stop overlaps another's prefix.
    for (const auto& stop : stops_) {
      const auto start = pending_.find(stop);
      if (start != std::string::npos &&
          (start + stop.size() < earliest_end ||
           (start + stop.size() == earliest_end && start < first))) {
        first = start;
        earliest_end = start + stop.size();
      }
    }
    if (first != std::string::npos) {
      auto output = pending_.substr(0, first);
      pending_.clear();
      matched_ = true;
      return output;
    }
    size_t retained = 0;
    for (const auto& stop : stops_) {
      const size_t maximum = std::min(pending_.size(), stop.size() - 1);
      for (size_t length = maximum; length > retained; --length) {
        if (std::string_view(pending_).substr(pending_.size() - length) ==
            std::string_view(stop).substr(0, length)) {
          retained = length;
          break;
        }
      }
    }
    auto output = pending_.substr(0, pending_.size() - retained);
    pending_.erase(0, pending_.size() - retained);
    return output;
  }
  std::string Finish() {
    if (finished_) throw std::invalid_argument("stop filter already finalized");
    finished_ = true;
    return std::exchange(pending_, {});
  }
  bool matched() const noexcept { return matched_; }
 private:
  std::vector<std::string> stops_;
  std::string pending_;
  bool matched_{false};
  bool finished_{false};
};
}  // namespace pih::qwen_plugin
