#include "pih/model/deepseek_attention_state.h"

#include <limits>
#include <utility>

namespace pih {
namespace {

Result<std::uint64_t> layer_bytes(std::uint32_t layer, std::uint32_t context) {
  if (context == 0 || context > 1048576) {
    return Status::InvalidArgument("DeepSeek reserved context is out of range");
  }
  if (layer < 2) return std::uint64_t{131072};
  if ((layer & 1U) == 0) {
    return std::uint64_t{212992} + (context / 4U) * std::uint64_t{1280};
  }
  return std::uint64_t{655360} + (context / 128U) * std::uint64_t{1024};
}

}  // namespace

Result<std::uint64_t> DeepSeekAttentionStateGeometry::MainLogicalBytes(
    std::uint32_t context) {
  DeepSeekStagePlan all{0, {0, 42}, true, true, false};
  return StageLogicalBytes(all, context);
}

Result<std::uint64_t> DeepSeekAttentionStateGeometry::StageLogicalBytes(
    const DeepSeekStagePlan& stage, std::uint32_t context) {
  if (stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43) {
    return Status::InvalidArgument("DeepSeek stage layer range is invalid");
  }
  std::uint64_t total = 0;
  for (std::uint32_t layer = stage.layers.first_layer;
       layer <= stage.layers.last_layer; ++layer) {
    auto bytes = layer_bytes(layer, context);
    if (!bytes.ok()) return bytes.status();
    if (total > std::numeric_limits<std::uint64_t>::max() - *bytes) {
      return Status::ResourceExhausted("DeepSeek attention state byte overflow");
    }
    total += *bytes;
  }
  if (stage.owns_dspark) total += 393216U;
  return total;
}

DeepSeekAttentionStateReservation::DeepSeekAttentionStateReservation(
    DeepSeekAttentionStatePool& pool, std::uint32_t sequence,
    std::uint64_t generation, std::uint64_t bytes) noexcept
    : pool_(&pool), sequence_(sequence), generation_(generation), bytes_(bytes) {}

DeepSeekAttentionStateReservation::DeepSeekAttentionStateReservation(
    DeepSeekAttentionStateReservation&& other) noexcept {
  *this = std::move(other);
}

DeepSeekAttentionStateReservation& DeepSeekAttentionStateReservation::operator=(
    DeepSeekAttentionStateReservation&& other) noexcept {
  if (this != &other) {
    abandon();
    pool_ = std::exchange(other.pool_, nullptr);
    sequence_ = other.sequence_;
    generation_ = other.generation_;
    bytes_ = other.bytes_;
    state_ = other.state_;
  }
  return *this;
}

DeepSeekAttentionStateReservation::~DeepSeekAttentionStateReservation() {
  abandon();
}

void DeepSeekAttentionStateReservation::abandon() noexcept {
  if (pool_ != nullptr && state_ == DeepSeekAttentionReservationState::kReserved) {
    pool_->reserved_bytes_ -= bytes_;
    pool_->live_sequences_.erase(sequence_);
    state_ = DeepSeekAttentionReservationState::kRolledBack;
  }
}

Status DeepSeekAttentionStateReservation::publish() {
  if (pool_ == nullptr || state_ != DeepSeekAttentionReservationState::kReserved) {
    return Status::FailedPrecondition("DeepSeek attention state is not reserved");
  }
  pool_->reserved_bytes_ -= bytes_;
  pool_->owned_bytes_ += bytes_;
  state_ = DeepSeekAttentionReservationState::kPublished;
  return Status::Ok();
}

Status DeepSeekAttentionStateReservation::rollback() {
  if (state_ == DeepSeekAttentionReservationState::kRolledBack) return Status::Ok();
  if (pool_ == nullptr || state_ != DeepSeekAttentionReservationState::kReserved) {
    return Status::FailedPrecondition("DeepSeek published state cannot roll back");
  }
  pool_->reserved_bytes_ -= bytes_;
  pool_->live_sequences_.erase(sequence_);
  state_ = DeepSeekAttentionReservationState::kRolledBack;
  return Status::Ok();
}

Status DeepSeekAttentionStateReservation::release() {
  auto validation = validate_release();
  if (!validation.ok()) return validation;
  pool_->owned_bytes_ -= bytes_;
  pool_->live_sequences_.erase(sequence_);
  state_ = DeepSeekAttentionReservationState::kReleased;
  return Status::Ok();
}

Status DeepSeekAttentionStateReservation::validate_release() const {
  if (pool_ == nullptr ||
      state_ != DeepSeekAttentionReservationState::kPublished) {
    return Status::FailedPrecondition(
        "DeepSeek attention state is not published");
  }
  return Status::Ok();
}

Result<DeepSeekAttentionStateReservation> DeepSeekAttentionStatePool::reserve(
    std::uint32_t sequence, const DeepSeekStagePlan& stage,
    std::uint32_t context) {
  auto bytes = DeepSeekAttentionStateGeometry::StageLogicalBytes(stage, context);
  if (!bytes.ok()) return bytes.status();
  if (live_sequences_.contains(sequence)) {
    return Status::FailedPrecondition(
        "DeepSeek sequence already owns attention state");
  }
  if (*bytes > available_bytes()) {
    return Status::ResourceExhausted("DeepSeek attention state capacity exhausted");
  }
  if (next_generation_ == 0) {
    return Status::ResourceExhausted("DeepSeek sequence generation exhausted");
  }
  const auto generation = next_generation_++;
  reserved_bytes_ += *bytes;
  live_sequences_.emplace(sequence, generation);
  return DeepSeekAttentionStateReservation(*this, sequence, generation, *bytes);
}

}  // namespace pih
