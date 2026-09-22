#include "pih/model/deepseek_online_topk.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace pih {

Status DeepSeekOnlineTopK::consume(std::uint32_t first_ordinal,
                                   std::span<const float> scores) {
  if (first_ordinal != consumed_ ||
      scores.size() > std::numeric_limits<std::uint32_t>::max() - consumed_) {
    return Status::InvalidArgument("DeepSeek online top-k tile is not contiguous");
  }
  for (const auto score : scores) {
    if (!std::isfinite(score)) {
      return Status::InvalidArgument(
          "DeepSeek online top-k score is nonfinite");
    }
  }
  for (std::uint32_t item = 0; item < scores.size(); ++item) {
    const auto score = scores[item];
    const auto ordinal = consumed_ + item;
    if (size_ < kCapacity) {
      scores_[size_] = score;
      ordinals_[size_++] = ordinal;
      continue;
    }
    std::uint32_t worst = 0;
    for (std::uint32_t candidate = 1; candidate < size_; ++candidate) {
      if (scores_[candidate] < scores_[worst] ||
          (scores_[candidate] == scores_[worst] &&
           ordinals_[candidate] > ordinals_[worst])) {
        worst = candidate;
      }
    }
    if (score > scores_[worst]) {
      if (!has_rejected_ || scores_[worst] > best_rejected_score_) {
        best_rejected_score_ = scores_[worst];
        has_rejected_ = true;
      }
      scores_[worst] = score;
      ordinals_[worst] = ordinal;
    } else {
      if (!has_rejected_ || score > best_rejected_score_) {
        best_rejected_score_ = score;
        has_rejected_ = true;
      }
    }
  }
  consumed_ += static_cast<std::uint32_t>(scores.size());
  return Status::Ok();
}

Result<std::vector<std::uint32_t>> DeepSeekOnlineTopK::finish() const {
  if (size_ == 0) {
    return Status::InvalidArgument("DeepSeek online top-k has no candidates");
  }
  std::vector<std::uint32_t> order(size_);
  std::iota(order.begin(), order.end(), 0U);
  std::stable_sort(order.begin(), order.end(), [&](const auto left,
                                                   const auto right) {
    return scores_[left] > scores_[right];
  });
  if (has_rejected_ && best_rejected_score_ == scores_[order.back()]) {
    return Status::InvalidArgument(
        "DeepSeek online top-k boundary has no portable tie order");
  }
  std::vector<std::uint32_t> result(size_);
  for (std::uint32_t index = 0; index < size_; ++index) {
    result[index] = ordinals_[order[index]];
  }
  return result;
}

}  // namespace pih
