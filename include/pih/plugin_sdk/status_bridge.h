#pragma once

#include <algorithm>
#include <string>

#include "pih/core/status.h"
#include "pih/plugin_sdk/status.h"

namespace pih {

inline Status plugin_status(const pih_status_v1& result) {
  if (!pih_status_is_valid_v1(&result))
    return Status::Internal("Plugin returned an invalid status");
  const std::string message(
      result.message,
      std::find(result.message, result.message + sizeof(result.message), '\0'));
  switch (result.code) {
    case PIH_STATUS_OK_V1: return Status::Ok();
    case PIH_STATUS_INVALID_ARGUMENT_V1: return Status::InvalidArgument(message);
    case PIH_STATUS_FAILED_PRECONDITION_V1:
      return Status::FailedPrecondition(message);
    case PIH_STATUS_RESOURCE_EXHAUSTED_V1:
      return Status::ResourceExhausted(message);
    case PIH_STATUS_UNAVAILABLE_V1: return Status::Unavailable(message);
    case PIH_STATUS_DEADLINE_EXCEEDED_V1:
      return Status::DeadlineExceeded(message);
    default: return Status::Internal(message);
  }
}

}  // namespace pih
