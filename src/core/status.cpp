#include "pih/core/status.h"

#include <utility>

namespace pih {

Status::Status(StatusCode code, std::string message)
    : code_(code), message_(std::move(message)) {
  if (code_ == StatusCode::kOk) {
    message_.clear();
  } else if (message_.size() > kMaxMessageBytes) {
    message_.resize(kMaxMessageBytes);
  }
}

Status Status::Ok() { return Status(StatusCode::kOk, {}); }

Status Status::InvalidArgument(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status Status::ResourceExhausted(std::string message) {
  return Status(StatusCode::kResourceExhausted, std::move(message));
}

Status Status::FailedPrecondition(std::string message) {
  return Status(StatusCode::kFailedPrecondition, std::move(message));
}

Status Status::Internal(std::string message) {
  return Status(StatusCode::kInternal, std::move(message));
}

Status Status::Unavailable(std::string message) {
  return Status(StatusCode::kUnavailable, std::move(message));
}

Status Status::DeadlineExceeded(std::string message) {
  return Status(StatusCode::kDeadlineExceeded, std::move(message));
}

}  // namespace pih
