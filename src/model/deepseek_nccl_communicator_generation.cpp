#include "pih/model/deepseek_nccl_communicator_generation.h"

#include "pih/core/checked_math.h"

namespace pih {

Result<DeepSeekNcclCommunicatorGeneration>
DeepSeekNcclCommunicatorGeneration::Create(
    DeepSeekNcclGenerationIdentity identity,
    std::vector<DeepSeekNcclEdgePair> edges) {
  if (identity.engine_epoch == 0 || identity.communicator_generation == 0 ||
      identity.world_size < 2 || identity.world_size > 4 ||
      identity.edge_timeout_ns == 0 || identity.set_timeout_ns == 0 ||
      edges.size() != identity.world_size - 1) {
    return Status::InvalidArgument(
        "DeepSeek NCCL generation identity is invalid");
  }
  auto minimum_set_timeout = checked_mul_u64(
      identity.edge_timeout_ns, identity.world_size - 1);
  if (!minimum_set_timeout.ok()) return minimum_set_timeout.status();
  if (identity.set_timeout_ns < *minimum_set_timeout) {
    return Status::InvalidArgument(
        "DeepSeek NCCL set timeout is shorter than serialized edges");
  }
  for (std::uint32_t edge = 0; edge + 1 < identity.world_size; ++edge) {
    if (edges[edge].state() != DeepSeekNcclEdgePairState::kPrepared ||
        edges[edge].edge_id() != edge ||
        edges[edge].engine_epoch() != identity.engine_epoch ||
        edges[edge].communicator_generation() !=
            identity.communicator_generation) {
      return Status::FailedPrecondition(
          "DeepSeek NCCL generation edge set is inconsistent");
    }
    if (edge > 0 &&
        edges[edge].config_identity() != edges[0].config_identity()) {
      return Status::FailedPrecondition(
          "DeepSeek NCCL generation mixes communicator configuration");
    }
    for (std::uint32_t prior = 0; prior < edge; ++prior) {
      if (edges[edge].bootstrap_lease_id() ==
              edges[prior].bootstrap_lease_id() ||
          edges[edge].bootstrap_commitment_id() ==
              edges[prior].bootstrap_commitment_id()) {
        return Status::FailedPrecondition(
            "DeepSeek NCCL generation reuses an edge bootstrap capability");
      }
    }
  }
  return DeepSeekNcclCommunicatorGeneration(identity, std::move(edges));
}

DeepSeekNcclCommunicatorGeneration::~DeepSeekNcclCommunicatorGeneration() {
  if (edges_.empty()) return;
  if (state_ != DeepSeekNcclGenerationState::kDestroyed &&
      state_ != DeepSeekNcclGenerationState::kAborted) {
    (void)abort();
  }
}

Status DeepSeekNcclCommunicatorGeneration::fail(Status cause) noexcept {
  (void)abort();
  return cause.ok() ? Status::Internal("DeepSeek NCCL generation failed")
                    : cause;
}

Status DeepSeekNcclCommunicatorGeneration::check_time(
    std::uint64_t now_ns) noexcept {
  if (now_ns == 0 || now_ns < last_observed_ns_) {
    return fail(Status::InvalidArgument(
        "DeepSeek NCCL generation clock is invalid"));
  }
  last_observed_ns_ = now_ns;
  if (now_ns >= set_deadline_ns_ || now_ns >= edge_deadline_ns_) {
    return fail(Status::DeadlineExceeded(
        "DeepSeek NCCL generation absolute deadline expired"));
  }
  return Status::Ok();
}

Status DeepSeekNcclCommunicatorGeneration::begin_init(std::uint64_t now_ns) {
  if (state_ != DeepSeekNcclGenerationState::kPrepared || now_ns == 0) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL generation is not prepared");
  }
  auto set_deadline = checked_add_u64(now_ns, identity_.set_timeout_ns);
  auto edge_deadline = checked_add_u64(now_ns, identity_.edge_timeout_ns);
  if (!set_deadline.ok()) return fail(set_deadline.status());
  if (!edge_deadline.ok()) return fail(edge_deadline.status());
  set_deadline_ns_ = *set_deadline;
  edge_deadline_ns_ = *edge_deadline;
  last_observed_ns_ = now_ns;
  active_edge_ = 0;
  state_ = DeepSeekNcclGenerationState::kInitializing;
  return advance_init(now_ns);
}

Status DeepSeekNcclCommunicatorGeneration::advance_init(
    std::uint64_t now_ns) {
  if (state_ != DeepSeekNcclGenerationState::kInitializing) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL generation init is not active");
  }
  auto time = check_time(now_ns);
  if (!time.ok()) return time;
  while (active_edge_ < edges_.size()) {
    auto& edge = edges_[active_edge_];
    Status status = Status::Ok();
    if (edge.state() == DeepSeekNcclEdgePairState::kPrepared) {
      auto deadline = checked_add_u64(now_ns, identity_.edge_timeout_ns);
      if (!deadline.ok()) return fail(deadline.status());
      edge_deadline_ns_ = *deadline;
      status = edge.begin_init();
    } else if (edge.state() ==
               DeepSeekNcclEdgePairState::kInitInProgress) {
      status = edge.poll_init();
    }
    if (!status.ok()) {
      if (status.code() == StatusCode::kUnavailable) return status;
      return fail(status);
    }
    if (edge.state() != DeepSeekNcclEdgePairState::kReconciled) {
      return fail(Status::Internal(
          "DeepSeek NCCL edge did not reconcile after successful advance"));
    }
    ++active_edge_;
  }
  state_ = DeepSeekNcclGenerationState::kReconciled;
  return Status::Ok();
}

Status DeepSeekNcclCommunicatorGeneration::seal(
    std::span<const DeepSeekNcclWarmupReceipt> receipts) {
  if (state_ != DeepSeekNcclGenerationState::kReconciled ||
      receipts.size() != edges_.size()) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL generation cannot seal warm-up receipts");
  }
  for (std::size_t edge = 0; edge < edges_.size(); ++edge) {
    auto status = edges_[edge].seal(receipts[edge]);
    if (!status.ok()) return fail(status);
  }
  state_ = DeepSeekNcclGenerationState::kSealed;
  return Status::Ok();
}

Status DeepSeekNcclCommunicatorGeneration::begin_teardown(
    std::uint64_t now_ns) {
  if (state_ != DeepSeekNcclGenerationState::kSealed || now_ns == 0) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL generation is not sealed for teardown");
  }
  auto deadline = checked_add_u64(now_ns, identity_.set_timeout_ns);
  auto edge_deadline = checked_add_u64(now_ns, identity_.edge_timeout_ns);
  if (!deadline.ok()) return fail(deadline.status());
  if (!edge_deadline.ok()) return fail(edge_deadline.status());
  set_deadline_ns_ = *deadline;
  edge_deadline_ns_ = *edge_deadline;
  last_observed_ns_ = now_ns;
  active_edge_ = edges_.size();
  state_ = DeepSeekNcclGenerationState::kFinalizing;
  return advance_teardown(now_ns);
}

Status DeepSeekNcclCommunicatorGeneration::advance_teardown(
    std::uint64_t now_ns) {
  if (state_ != DeepSeekNcclGenerationState::kFinalizing) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL generation teardown is not active");
  }
  auto time = check_time(now_ns);
  if (!time.ok()) return time;
  while (active_edge_ > 0) {
    auto& edge = edges_[active_edge_ - 1];
    Status status = Status::Ok();
    if (edge.state() == DeepSeekNcclEdgePairState::kSealed) {
      auto deadline = checked_add_u64(now_ns, identity_.edge_timeout_ns);
      if (!deadline.ok()) return fail(deadline.status());
      edge_deadline_ns_ = *deadline;
      status = edge.begin_finalize();
    } else if (edge.state() ==
               DeepSeekNcclEdgePairState::kFinalizeInProgress) {
      status = edge.poll_finalize();
    }
    if (!status.ok() && status.code() != StatusCode::kUnavailable) {
      return fail(status);
    }
    if (status.code() == StatusCode::kUnavailable) return status;
    status = edge.destroy();
    if (!status.ok()) return fail(status);
    --active_edge_;
  }
  state_ = DeepSeekNcclGenerationState::kDestroyed;
  return Status::Ok();
}

Status DeepSeekNcclCommunicatorGeneration::abort() noexcept {
  if (state_ == DeepSeekNcclGenerationState::kAborted) return Status::Ok();
  if (state_ == DeepSeekNcclGenerationState::kDestroyed) {
    return Status::FailedPrecondition(
        "destroyed DeepSeek NCCL generation cannot abort");
  }
  Status first = Status::Ok();
  for (auto edge = edges_.rbegin(); edge != edges_.rend(); ++edge) {
    const auto status = edge->abort();
    if (first.ok() && !status.ok()) first = status;
  }
  state_ = DeepSeekNcclGenerationState::kAborted;
  return first;
}

DeepSeekBoundaryTransportDriver*
DeepSeekNcclCommunicatorGeneration::incoming_transport(
    std::uint32_t rank) noexcept {
  if (state_ != DeepSeekNcclGenerationState::kSealed || rank == 0 ||
      rank >= identity_.world_size) return nullptr;
  return edges_[rank - 1].upper_transport();
}

DeepSeekBoundaryTransportDriver*
DeepSeekNcclCommunicatorGeneration::outgoing_transport(
    std::uint32_t rank) noexcept {
  if (state_ != DeepSeekNcclGenerationState::kSealed ||
      rank + 1 >= identity_.world_size) return nullptr;
  return edges_[rank].lower_transport();
}

DeepSeekBoundaryTransportDriver*
DeepSeekNcclCommunicatorGeneration::incoming_warmup_transport(
    std::uint32_t rank) noexcept {
  if (state_ != DeepSeekNcclGenerationState::kReconciled || rank == 0 ||
      rank >= identity_.world_size) return nullptr;
  return edges_[rank - 1].upper_warmup_transport();
}

DeepSeekBoundaryTransportDriver*
DeepSeekNcclCommunicatorGeneration::outgoing_warmup_transport(
    std::uint32_t rank) noexcept {
  if (state_ != DeepSeekNcclGenerationState::kReconciled ||
      rank + 1 >= identity_.world_size) return nullptr;
  return edges_[rank].lower_warmup_transport();
}

}  // namespace pih
