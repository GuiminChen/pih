#include "pih/model/cgroup_v2_events.h"

namespace pih {

Result<bool> parse_cgroup_v2_populated(std::string_view events) {
  if (events.empty() || events.size() > 4096 ||
      events.find('\0') != std::string_view::npos)
    return Status::InvalidArgument("cgroup.events payload is invalid");
  bool found = false;
  bool populated = false;
  std::size_t offset = 0;
  while (offset < events.size()) {
    const auto newline = events.find('\n', offset);
    const auto end = newline == std::string_view::npos ? events.size() : newline;
    const auto line = events.substr(offset, end - offset);
    if (line.empty())
      return Status::InvalidArgument("cgroup.events contains an empty record");
    const auto separator = line.find(' ');
    if (separator == 0 || separator == std::string_view::npos ||
        line.find(' ', separator + 1) != std::string_view::npos ||
        separator + 1 == line.size())
      return Status::InvalidArgument("cgroup.events record is malformed");
    const auto key = line.substr(0, separator);
    const auto value = line.substr(separator + 1);
    if (key == "populated") {
      if (found || (value != "0" && value != "1"))
        return Status::InvalidArgument(
            "cgroup.events populated record is invalid");
      found = true;
      populated = value == "1";
    }
    if (newline == std::string_view::npos) break;
    offset = newline + 1;
    if (offset == events.size()) break;
  }
  if (!found)
    return Status::FailedPrecondition(
        "cgroup.events has no populated observation");
  return populated;
}

}  // namespace pih
