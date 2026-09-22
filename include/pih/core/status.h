#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace pih {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kResourceExhausted,
  kFailedPrecondition,
  kInternal,
  kUnavailable,
  kDeadlineExceeded,
};

class Status final {
 public:
  static constexpr std::size_t kMaxMessageBytes = 4096;

  static Status Ok();
  static Status InvalidArgument(std::string message);
  static Status ResourceExhausted(std::string message);
  static Status FailedPrecondition(std::string message);
  static Status Internal(std::string message);
  static Status Unavailable(std::string message);
  static Status DeadlineExceeded(std::string message);

  [[nodiscard]] bool ok() const noexcept { return code_ == StatusCode::kOk; }
  [[nodiscard]] StatusCode code() const noexcept { return code_; }
  [[nodiscard]] std::string_view message() const noexcept { return message_; }

 private:
  Status(StatusCode code, std::string message);

  StatusCode code_;
  std::string message_;
};

}  // namespace pih
