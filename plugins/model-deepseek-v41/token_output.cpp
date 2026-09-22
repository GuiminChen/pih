#include "token_output.h"
#include <atomic>
#include <limits>
#include <new>
#include <type_traits>

namespace pih::deepseek_v41 {
Result<std::unique_ptr<TokenOutputQueue>> TokenOutputQueue::Create(std::uint32_t slots, std::uint64_t budget) {
  if (!slots || slots > 4096) return Status::InvalidArgument("Token output slot count must be 1..4096");
  if (slots > budget / sizeof(TokenPublication)) return Status::ResourceExhausted("Publication slots exceed byte budget");
  auto credits = OutputBurstCreditPool::Create({slots, 1, 5, 1, sizeof(TokenPublication)});
  if (!credits.ok()) return credits.status();
  static std::atomic<std::uint64_t> next_instance{1};
  auto instance = next_instance.load(std::memory_order_relaxed);
  do {
    if (instance == std::numeric_limits<std::uint64_t>::max()) return Status::FailedPrecondition("Output queue identity space exhausted");
  } while (!next_instance.compare_exchange_weak(instance, instance + 1, std::memory_order_relaxed));
  try {
    auto queue = std::unique_ptr<TokenOutputQueue>(new TokenOutputQueue(std::move(*credits)));
    queue->slots_.resize(slots); queue->instance_ = instance; return queue;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Publication slot allocation failed"); }
}
Status TokenOutputQueue::Identity(const TokenOutputLease& lease) const {
  const auto& c = lease.credit;
  if (lease.queue_instance != instance_ || c.credit_index >= slots_.size() ||
      c.records != 1 || c.slots != 1 || c.bytes != sizeof(TokenPublication))
    return Status::FailedPrecondition("Output lease belongs to a different queue or envelope");
  return Status::Ok();
}
Result<TokenOutputLease> TokenOutputQueue::Reserve(std::uint64_t plan) {
  auto credit = credits_.acquire(plan, OutputPlanKind::kDeepSeek, 1, 1, sizeof(TokenPublication));
  if (!credit.ok()) return credit.status();
  TokenOutputLease lease{instance_, *credit}; slots_[credit->credit_index].lease = lease; return lease;
}
Status TokenOutputQueue::MarkInFlight(const TokenOutputLease& lease) {
  const auto valid = Identity(lease); if (!valid.ok()) return valid;
  return credits_.commit(lease.credit);
}
Status TokenOutputQueue::ValidateCommitted(const TokenOutputLease& lease) const {
  const auto valid = Identity(lease); if (!valid.ok()) return valid;
  return credits_.validate_transfer(lease.credit);
}
Status TokenOutputQueue::AbortPrepared(const TokenOutputLease& lease) {
  const auto valid = Identity(lease); if (!valid.ok()) return valid;
  const auto aborted = credits_.abort(lease.credit); if (!aborted.ok()) return aborted;
  slots_[lease.credit.credit_index] = {}; return Status::Ok();
}
Status TokenOutputQueue::DiscardRetired(const TokenOutputLease& lease) {
  const auto valid = ValidateCommitted(lease); if (!valid.ok()) return valid;
  const auto transfer = credits_.transfer(lease.credit); if (!transfer.ok()) return transfer;
  const auto released = credits_.release(lease.credit); if (!released.ok()) return released;
  slots_[lease.credit.credit_index] = {}; return Status::Ok();
}
Result<TokenOutputLease> TokenOutputQueue::Publish(const TokenOutputLease& lease, const TokenPublication& publication) {
  const auto valid = ValidateCommitted(lease); if (!valid.ok()) return valid;
  if (publication.visible_size > publication.visible.size() || publication.record.observation.identity.plan_seq != lease.credit.plan_sequence)
    return Status::InvalidArgument("Publication differs from reserved output envelope or plan");
  const auto transfer = credits_.transfer(lease.credit); if (!transfer.ok()) return transfer;
  static_assert(std::is_nothrow_copy_assignable_v<TokenPublication>);
  auto& slot = slots_[lease.credit.credit_index]; slot.publication = publication; slot.ready = true; return lease;
}
Result<TokenPublication> TokenOutputQueue::Read(const TokenOutputLease& lease) const {
  const auto valid = Identity(lease); if (!valid.ok()) return valid;
  const auto& slot = slots_[lease.credit.credit_index]; const auto& c = slot.lease.credit;
  if (!slot.ready || c.credit_generation != lease.credit.credit_generation || c.plan_sequence != lease.credit.plan_sequence)
    return Status::FailedPrecondition("Output is pending, released or belongs to an older slot generation");
  return slot.publication;
}
Status TokenOutputQueue::Release(const TokenOutputLease& lease) {
  const auto ready = Read(lease); if (!ready.ok()) return ready.status();
  const auto released = credits_.release(lease.credit); if (!released.ok()) return released;
  slots_[lease.credit.credit_index] = {}; return Status::Ok();
}
}  // namespace pih::deepseek_v41
