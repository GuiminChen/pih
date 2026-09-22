#include "pih/model/engine_inet_diag_multipart_decoder.h"

namespace pih {
namespace {

constexpr std::size_t kMaximumInetDiagRows = 1024U * 1024U;

}  // namespace

Result<EngineInetDiagMultipartDecoder>
EngineInetDiagMultipartDecoder::Create(std::uint32_t request_sequence) {
  if (request_sequence == 0) {
    return Status::InvalidArgument("inet_diag request sequence is zero");
  }
  return EngineInetDiagMultipartDecoder(request_sequence);
}

Status EngineInetDiagMultipartDecoder::accept(
    const EngineInetDiagMultipartMessage& message) {
  if (poisoned_ || done_) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "inet_diag multipart dump already terminated");
  }
  if (message.sequence != sequence_ || message.truncated ||
      !message.multipart) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "inet_diag multipart envelope is invalid");
  }

  switch (message.kind) {
    case EngineInetDiagMultipartMessageKind::kRow:
      if (message.error_code != 0 || message.row.local_port == 0 ||
          static_cast<std::uint32_t>(message.row.lifecycle) >= 2) {
        poisoned_ = true;
        return Status::FailedPrecondition(
            "inet_diag multipart row is invalid");
      }
      if (rows_.size() >= kMaximumInetDiagRows) {
        poisoned_ = true;
        return Status::ResourceExhausted(
            "inet_diag multipart dump has too many rows");
      }
      rows_.push_back(message.row);
      return Status::Ok();
    case EngineInetDiagMultipartMessageKind::kDone:
      if (message.error_code != 0) {
        poisoned_ = true;
        return Status::FailedPrecondition(
            "inet_diag DONE carried an error code");
      }
      done_ = true;
      return Status::Ok();
    case EngineInetDiagMultipartMessageKind::kError:
      poisoned_ = true;
      return message.error_code == 0
                 ? Status::FailedPrecondition(
                       "inet_diag error message has no error code")
                 : Status::Unavailable("inet_diag multipart dump failed");
  }
  poisoned_ = true;
  return Status::FailedPrecondition(
      "inet_diag multipart message kind is invalid");
}

Result<std::vector<EngineNetworkPortCensusRow>>
EngineInetDiagMultipartDecoder::finish() const {
  if (poisoned_ || !done_) {
    return Status::FailedPrecondition(
        "inet_diag multipart dump is incomplete or poisoned");
  }
  return rows_;
}

}  // namespace pih
