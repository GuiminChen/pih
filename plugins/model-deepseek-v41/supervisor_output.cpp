#include "supervisor_output.h"
#include <algorithm>

namespace pih::deepseek_v41 {
Status SupervisorOutput::Adopt(TokenOutputLease lease) {
  if (lease_) return Status::FailedPrecondition("Supervisor output lease still pending");
  auto publication = queue_->Read(lease);
  if (!publication.ok()) { failed_ = true; return publication.status(); }
  // Retain the valid lease even when its record is invalid: the caller must be
  // able to discard it during fault retirement, without losing queue credit.
  lease_ = lease; publication_ = *publication; sent_ = 0;
  if (publication_.visible_size > publication_.visible.size() ||
      publication_.record.accepted_count != released_ + 1) {
    failed_ = true;
    return Status::FailedPrecondition("Supervisor output publication order or extent invalid");
  }
  return Status::Ok();
}
Status SupervisorOutput::Release(bool delivered) {
  const auto released = queue_->Release(*lease_);
  if (!released.ok()) { failed_ = true; return released; }
  ++released_; if (delivered) ++delivered_;
  lease_.reset(); return Status::Ok();
}
Result<bool> SupervisorOutput::Write(void* context, Writer writer) {
  if (failed_ || !lease_ || !writer)
    return Status::FailedPrecondition("Supervisor output writer or custody invalid");
  if (sent_ < publication_.visible_size) {
    const auto size = std::min<std::size_t>(4096, publication_.visible_size - sent_);
    try {
      auto count = writer(context, {publication_.visible.data() + sent_, size});
      if (!count.ok()) { failed_ = true; return count.status(); }
      if (*count > size) {
        failed_ = true;
        return Status::FailedPrecondition("Supervisor writer consumed beyond offered bytes");
      }
      sent_ += *count; visible_bytes_ += *count;
    } catch (...) {
      failed_ = true;
      return Status::Internal("Supervisor output writer threw an exception");
    }
  }
  if (sent_ != publication_.visible_size) return false;
  const auto released = Release(true); if (!released.ok()) return released;
  return true;
}
Status SupervisorOutput::Discard() {
  failed_ = true;
  return lease_ ? Release(false) : Status::Ok();
}
Status SupervisorOutput::Finish(const TokenLedger& ledger) const {
  if (failed_ || lease_ || ledger.failed() || ledger.pending_output() || ledger.finish() == TokenFinish::kNone ||
      released_ != delivered_ || delivered_ != ledger.records().size())
    return Status::FailedPrecondition("Supervisor completion output ledger is not terminal");
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
