#include "pih/model/deepseek_attention_sequence_transaction.h"

namespace pih {
namespace {

bool same(const DeepSeekRatio4PagePair& left,
          const DeepSeekRatio4PagePair& right) {
  return left.main == right.main && left.index == right.index &&
         left.sequence == right.sequence &&
         left.layer_id == right.layer_id &&
         left.logical_page == right.logical_page &&
         left.generation == right.generation;
}

bool same(const DeepSeekRatio128Page& left,
          const DeepSeekRatio128Page& right) {
  return left.handle == right.handle && left.sequence == right.sequence &&
         left.layer_id == right.layer_id &&
         left.logical_page == right.logical_page;
}

}  // namespace

Result<DeepSeekAttentionSequenceTransaction>
DeepSeekAttentionSequenceTransaction::Create(
    std::uint32_t sequence, std::uint32_t maximum_ratio4_mutations,
    std::uint32_t maximum_ratio128_mutations,
    DeepSeekFixedStateBanks& fixed_banks,
    DeepSeekRatio4PagePool& ratio4_pool,
    DeepSeekRatio128PagePool& ratio128_pool) {
  if (maximum_ratio4_mutations > 43U ||
      maximum_ratio128_mutations > 43U) {
    return Status::InvalidArgument(
        "DeepSeek attention sequence mutation bound is invalid");
  }
  DeepSeekAttentionSequenceTransaction value;
  value.sequence_ = sequence;
  value.ratio4_.resize(maximum_ratio4_mutations);
  value.ratio128_.resize(maximum_ratio128_mutations);
  value.fixed_banks_ = &fixed_banks;
  value.ratio4_pool_ = &ratio4_pool;
  value.ratio128_pool_ = &ratio128_pool;
  return value;
}

Status DeepSeekAttentionSequenceTransaction::begin(std::uintptr_t stream) {
  auto status = validate_begin(stream);
  if (!status.ok()) return status;
  status = fixed_banks_->prepare(stream);
  if (!status.ok()) {
    state_ = DeepSeekAttentionSequenceTransactionState::kPoisoned;
    return status;
  }
  ratio4_count_ = 0;
  ratio128_count_ = 0;
  external_error_channels_.clear();
  ++prepare_epoch_;
  stream_ = stream;
  state_ = DeepSeekAttentionSequenceTransactionState::kPreparing;
  return Status::Ok();
}

Status DeepSeekAttentionSequenceTransaction::validate_begin(
    std::uintptr_t stream) const {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kIdle ||
      stream == 0) {
    return Status::FailedPrecondition(
        "DeepSeek attention sequence transaction is not idle");
  }
  return fixed_banks_->validate_prepare(stream);
}

Result<bool> DeepSeekAttentionSequenceTransaction::claim_external_error_channel(
    std::uint32_t* host_error_flag,
    std::uintptr_t device_error_flag_u32) {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing ||
      host_error_flag == nullptr || device_error_flag_u32 == 0) {
    return Status::FailedPrecondition(
        "DeepSeek attention external error channel is not claimable");
  }
  for (const auto& channel : external_error_channels_) {
    if (channel.host == host_error_flag &&
        channel.device == device_error_flag_u32) {
      return false;
    }
    if (channel.host == host_error_flag) {
      return Status::FailedPrecondition(
          "DeepSeek attention external error channel aliases another pair");
    }
  }
  external_error_channels_.push_back(
      {host_error_flag, device_error_flag_u32});
  return true;
}

Result<DeepSeekExpertArenaSpan>
DeepSeekAttentionSequenceTransaction::tentative_fixed_state() const {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing) {
    return Status::FailedPrecondition(
        "DeepSeek tentative fixed state is not writable");
  }
  return DeepSeekExpertArenaSpan{fixed_banks_->tentative_address(),
                                 fixed_banks_->bank_bytes()};
}

Result<DeepSeekRatio4PagePair>
DeepSeekAttentionSequenceTransaction::reserve_ratio4_append(
    std::uint32_t layer_id, std::uint32_t logical_page) {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing ||
      ratio4_count_ >= ratio4_.size()) {
    return Status::ResourceExhausted(
        "DeepSeek ratio-4 transaction mutation bound is exhausted");
  }
  auto page = ratio4_pool_->reserve(sequence_, layer_id, logical_page);
  if (!page.ok()) return page.status();
  ratio4_[ratio4_count_++] = {false, {}, *page};
  return *page;
}

Result<DeepSeekRatio4PagePair>
DeepSeekAttentionSequenceTransaction::reserve_ratio4_tail_cow(
    const DeepSeekRatio4PagePair& committed) {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing ||
      committed.sequence != sequence_ || ratio4_count_ >= ratio4_.size()) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-4 tail COW is not reservable");
  }
  auto page = ratio4_pool_->reserve_tail_cow(committed);
  if (!page.ok()) return page.status();
  ratio4_[ratio4_count_++] = {true, committed, *page};
  return *page;
}

Result<DeepSeekRatio128Page>
DeepSeekAttentionSequenceTransaction::reserve_ratio128_append(
    std::uint32_t layer_id, std::uint32_t logical_page) {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing ||
      ratio128_count_ >= ratio128_.size()) {
    return Status::ResourceExhausted(
        "DeepSeek ratio-128 transaction mutation bound is exhausted");
  }
  auto page = ratio128_pool_->reserve(sequence_, layer_id, logical_page);
  if (!page.ok()) return page.status();
  ratio128_[ratio128_count_++] = {false, {}, *page};
  return *page;
}

Result<DeepSeekRatio128Page>
DeepSeekAttentionSequenceTransaction::reserve_ratio128_tail_cow(
    const DeepSeekRatio128Page& committed) {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing ||
      committed.sequence != sequence_ ||
      ratio128_count_ >= ratio128_.size()) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-128 tail COW is not reservable");
  }
  auto page = ratio128_pool_->reserve_tail_cow(committed);
  if (!page.ok()) return page.status();
  ratio128_[ratio128_count_++] = {true, committed, *page};
  return *page;
}

Result<std::optional<DeepSeekRatio4PagePair>>
DeepSeekAttentionSequenceTransaction::find_published_ratio4(
    std::uint32_t layer_id, std::uint32_t logical_page) const {
  if (ratio4_pool_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-4 page pool is unavailable");
  }
  return ratio4_pool_->find_published(sequence_, layer_id, logical_page);
}

Result<std::vector<std::uint32_t>>
DeepSeekAttentionSequenceTransaction::ratio4_index_page_slots(
    std::uint32_t layer_id, std::uint32_t logical_page_count) const {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing ||
      ratio4_pool_ == nullptr || layer_id >= 43 ||
      logical_page_count == 0 || logical_page_count > 4096) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-4 index page table is not materializable");
  }
  std::vector<std::uint32_t> result;
  result.reserve(logical_page_count);
  for (std::uint32_t logical_page = 0;
       logical_page < logical_page_count; ++logical_page) {
    const DeepSeekRatio4PagePair* tentative = nullptr;
    for (std::uint32_t mutation = 0; mutation < ratio4_count_; ++mutation) {
      const auto& replacement = ratio4_[mutation].replacement;
      if (replacement.layer_id == layer_id &&
          replacement.logical_page == logical_page) {
        if (tentative != nullptr) {
          return Status::Internal(
              "DeepSeek ratio-4 index page table has duplicate mutation");
        }
        tentative = &replacement;
      }
    }
    if (tentative != nullptr) {
      result.push_back(tentative->index.slot());
      continue;
    }
    auto published = ratio4_pool_->find_published(
        sequence_, layer_id, logical_page);
    if (!published.ok()) return published.status();
    if (!published->has_value()) {
      return Status::FailedPrecondition(
          "DeepSeek ratio-4 index page table has a logical hole");
    }
    result.push_back((**published).index.slot());
  }
  return result;
}

Result<std::vector<std::uint32_t>>
DeepSeekAttentionSequenceTransaction::ratio4_main_page_slots(
    std::uint32_t layer_id, std::uint32_t logical_page_count) const {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing ||
      ratio4_pool_ == nullptr || layer_id >= 43 ||
      logical_page_count == 0 || logical_page_count > 4096) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-4 main page table is not materializable");
  }
  std::vector<std::uint32_t> result;
  result.reserve(logical_page_count);
  for (std::uint32_t logical_page = 0;
       logical_page < logical_page_count; ++logical_page) {
    const DeepSeekRatio4PagePair* tentative = nullptr;
    for (std::uint32_t mutation = 0; mutation < ratio4_count_; ++mutation) {
      const auto& replacement = ratio4_[mutation].replacement;
      if (replacement.layer_id == layer_id &&
          replacement.logical_page == logical_page) {
        if (tentative != nullptr) {
          return Status::Internal(
              "DeepSeek ratio-4 main page table has duplicate mutation");
        }
        tentative = &replacement;
      }
    }
    if (tentative != nullptr) {
      result.push_back(tentative->main.slot());
      continue;
    }
    auto published = ratio4_pool_->find_published(
        sequence_, layer_id, logical_page);
    if (!published.ok()) return published.status();
    if (!published->has_value()) {
      return Status::FailedPrecondition(
          "DeepSeek ratio-4 main page table has a logical hole");
    }
    result.push_back((**published).main.slot());
  }
  return result;
}

std::uint32_t DeepSeekAttentionSequenceTransaction::
    ratio4_physical_page_count() const noexcept {
  return ratio4_pool_ == nullptr ? 0 : ratio4_pool_->page_pair_capacity();
}

Result<std::optional<DeepSeekRatio128Page>>
DeepSeekAttentionSequenceTransaction::find_published_ratio128(
    std::uint32_t layer_id, std::uint32_t logical_page) const {
  if (ratio128_pool_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-128 page pool is unavailable");
  }
  return ratio128_pool_->find_published(sequence_, layer_id, logical_page);
}

Result<std::vector<std::uint32_t>>
DeepSeekAttentionSequenceTransaction::ratio128_main_page_slots(
    std::uint32_t layer_id, std::uint32_t logical_page_count) const {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing ||
      ratio128_pool_ == nullptr || layer_id >= 43 ||
      logical_page_count == 0 || logical_page_count > 4096) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-128 main page table is not materializable");
  }
  std::vector<std::uint32_t> result;
  result.reserve(logical_page_count);
  for (std::uint32_t logical_page = 0;
       logical_page < logical_page_count; ++logical_page) {
    const DeepSeekRatio128Page* tentative = nullptr;
    for (std::uint32_t mutation = 0; mutation < ratio128_count_; ++mutation) {
      const auto& replacement = ratio128_[mutation].replacement;
      if (replacement.layer_id == layer_id &&
          replacement.logical_page == logical_page) {
        if (tentative != nullptr) {
          return Status::Internal(
              "DeepSeek ratio-128 main page table has duplicate mutation");
        }
        tentative = &replacement;
      }
    }
    if (tentative != nullptr) {
      result.push_back(tentative->handle.slot());
      continue;
    }
    auto published = ratio128_pool_->find_published(
        sequence_, layer_id, logical_page);
    if (!published.ok()) return published.status();
    if (!published->has_value()) {
      return Status::FailedPrecondition(
          "DeepSeek ratio-128 main page table has a logical hole");
    }
    result.push_back((**published).handle.slot());
  }
  return result;
}

std::uint32_t DeepSeekAttentionSequenceTransaction::
    ratio128_physical_page_count() const noexcept {
  return ratio128_pool_ == nullptr ? 0 : ratio128_pool_->page_capacity();
}

Status DeepSeekAttentionSequenceTransaction::validate_ratio4_append_target(
    const DeepSeekRatio4PagePair& target) const {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-4 append target is not writable");
  }
  for (std::uint32_t index = 0; index < ratio4_count_; ++index) {
    if (!ratio4_[index].cow && same(ratio4_[index].replacement, target)) {
      return Status::Ok();
    }
  }
  return Status::FailedPrecondition(
      "DeepSeek ratio-4 append target is not transaction-owned");
}

Status DeepSeekAttentionSequenceTransaction::validate_ratio4_cow_target(
    const DeepSeekRatio4PagePair& committed,
    const DeepSeekRatio4PagePair& target) const {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-4 COW target is not writable");
  }
  for (std::uint32_t index = 0; index < ratio4_count_; ++index) {
    if (ratio4_[index].cow && same(ratio4_[index].committed, committed) &&
        same(ratio4_[index].replacement, target)) {
      return Status::Ok();
    }
  }
  return Status::FailedPrecondition(
      "DeepSeek ratio-4 COW target is not transaction-owned");
}

Status DeepSeekAttentionSequenceTransaction::validate_ratio128_append_target(
    const DeepSeekRatio128Page& target) const {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-128 append target is not writable");
  }
  for (std::uint32_t index = 0; index < ratio128_count_; ++index) {
    if (!ratio128_[index].cow && same(ratio128_[index].replacement, target)) {
      return Status::Ok();
    }
  }
  return Status::FailedPrecondition(
      "DeepSeek ratio-128 append target is not transaction-owned");
}

Status DeepSeekAttentionSequenceTransaction::validate_ratio128_cow_target(
    const DeepSeekRatio128Page& committed,
    const DeepSeekRatio128Page& target) const {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-128 COW target is not writable");
  }
  for (std::uint32_t index = 0; index < ratio128_count_; ++index) {
    if (ratio128_[index].cow && same(ratio128_[index].committed, committed) &&
        same(ratio128_[index].replacement, target)) {
      return Status::Ok();
    }
  }
  return Status::FailedPrecondition(
      "DeepSeek ratio-128 COW target is not transaction-owned");
}

Status DeepSeekAttentionSequenceTransaction::seal(std::uintptr_t stream) {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing ||
      stream != stream_) {
    return Status::FailedPrecondition(
        "DeepSeek attention sequence transaction is not sealable");
  }
  auto status = fixed_banks_->seal(stream);
  if (!status.ok()) {
    state_ = DeepSeekAttentionSequenceTransactionState::kPoisoned;
    return status;
  }
  state_ = DeepSeekAttentionSequenceTransactionState::kAwaitingCompletion;
  return Status::Ok();
}

Result<DeepSeekExpertAsyncStatus>
DeepSeekAttentionSequenceTransaction::poll() {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kAwaitingCompletion) {
    return Status::FailedPrecondition(
        "DeepSeek attention sequence transaction has no completion");
  }
  auto result = fixed_banks_->poll();
  if (!result.ok()) {
    state_ = DeepSeekAttentionSequenceTransactionState::kPoisoned;
    return result.status();
  }
  if (*result == DeepSeekExpertAsyncStatus::kSuccess) {
    for (const auto& channel : external_error_channels_) {
      if (*channel.host != 0) {
        state_ = DeepSeekAttentionSequenceTransactionState::kPoisoned;
        return DeepSeekExpertAsyncStatus::kError;
      }
    }
    state_ = DeepSeekAttentionSequenceTransactionState::kReadyToResolve;
  } else if (*result == DeepSeekExpertAsyncStatus::kError) {
    state_ = DeepSeekAttentionSequenceTransactionState::kPoisoned;
  } else if (*result != DeepSeekExpertAsyncStatus::kInProgress) {
    state_ = DeepSeekAttentionSequenceTransactionState::kPoisoned;
    return Status::Internal(
        "DeepSeek attention transaction received invalid async state");
  }
  return *result;
}

Status DeepSeekAttentionSequenceTransaction::commit() {
  auto status = validate_commit();
  if (!status.ok()) return status;
  for (std::uint32_t index = 0; index < ratio4_count_; ++index) {
    const auto& item = ratio4_[index];
    status = item.cow ? ratio4_pool_->publish_tail_cow(
                            item.committed, item.replacement)
                      : ratio4_pool_->publish(item.replacement);
    if (!status.ok()) {
      state_ = DeepSeekAttentionSequenceTransactionState::kPoisoned;
      return status;
    }
  }
  for (std::uint32_t index = 0; index < ratio128_count_; ++index) {
    const auto& item = ratio128_[index];
    status = item.cow ? ratio128_pool_->publish_tail_cow(
                            item.committed, item.replacement)
                      : ratio128_pool_->publish(item.replacement);
    if (!status.ok()) {
      state_ = DeepSeekAttentionSequenceTransactionState::kPoisoned;
      return status;
    }
  }
  status = fixed_banks_->commit();
  if (!status.ok()) {
    state_ = DeepSeekAttentionSequenceTransactionState::kPoisoned;
    return status;
  }
  state_ = DeepSeekAttentionSequenceTransactionState::kIdle;
  return Status::Ok();
}

Status DeepSeekAttentionSequenceTransaction::validate_commit() const {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kReadyToResolve) {
    return Status::FailedPrecondition(
        "DeepSeek attention sequence transaction is not committable");
  }
  auto status = fixed_banks_->validate_commit();
  if (!status.ok()) return status;
  for (std::uint32_t index = 0; index < ratio4_count_; ++index) {
    const auto& item = ratio4_[index];
    status = item.cow
                 ? ratio4_pool_->validate_publish_tail_cow(
                       item.committed, item.replacement)
                 : ratio4_pool_->validate_publish(item.replacement);
    if (!status.ok()) return status;
  }
  for (std::uint32_t index = 0; index < ratio128_count_; ++index) {
    const auto& item = ratio128_[index];
    status = item.cow
                 ? ratio128_pool_->validate_publish_tail_cow(
                       item.committed, item.replacement)
                 : ratio128_pool_->validate_publish(item.replacement);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status DeepSeekAttentionSequenceTransaction::cancel_preparing() {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kPreparing) {
    return Status::FailedPrecondition(
        "DeepSeek attention sequence transaction is not cancellable");
  }
  Status first = Status::Ok();
  for (std::uint32_t index = 0; index < ratio4_count_; ++index) {
    auto status = ratio4_pool_->rollback(ratio4_[index].replacement);
    if (first.ok() && !status.ok()) first = status;
  }
  for (std::uint32_t index = 0; index < ratio128_count_; ++index) {
    auto status = ratio128_pool_->rollback(ratio128_[index].replacement);
    if (first.ok() && !status.ok()) first = status;
  }
  auto fixed = fixed_banks_->cancel_prepared();
  if (first.ok() && !fixed.ok()) first = fixed;
  ratio4_count_ = 0;
  ratio128_count_ = 0;
  state_ = first.ok() ? DeepSeekAttentionSequenceTransactionState::kIdle
                      : DeepSeekAttentionSequenceTransactionState::kPoisoned;
  return first;
}

Status DeepSeekAttentionSequenceTransaction::abort() {
  if (state_ != DeepSeekAttentionSequenceTransactionState::kReadyToResolve) {
    return Status::FailedPrecondition(
        "DeepSeek attention sequence transaction is not abortable");
  }
  Status first = Status::Ok();
  for (std::uint32_t index = 0; index < ratio4_count_; ++index) {
    auto status = ratio4_pool_->rollback(ratio4_[index].replacement);
    if (first.ok() && !status.ok()) first = status;
  }
  for (std::uint32_t index = 0; index < ratio128_count_; ++index) {
    auto status = ratio128_pool_->rollback(ratio128_[index].replacement);
    if (first.ok() && !status.ok()) first = status;
  }
  auto fixed = fixed_banks_->abort();
  if (first.ok() && !fixed.ok()) first = fixed;
  state_ = first.ok() ? DeepSeekAttentionSequenceTransactionState::kIdle
                      : DeepSeekAttentionSequenceTransactionState::kPoisoned;
  return first;
}

}  // namespace pih
