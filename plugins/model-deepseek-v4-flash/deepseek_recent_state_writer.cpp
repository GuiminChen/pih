#include "pih/model/deepseek_recent_state_writer.h"

#include <utility>

namespace pih {
namespace {

constexpr std::uint32_t kRecentRows = 128;
constexpr std::size_t kLatentWidth = 512;
constexpr std::size_t kBf16Bytes = 2;

}  // namespace

Result<DeepSeekRecentStateWriter> DeepSeekRecentStateWriter::Create(
    DeepSeekFixedStateLayout layout,
    DeepSeekRecentStateOperations& operations) {
  if (layout.total_bytes() == 0) {
    return Status::InvalidArgument(
        "DeepSeek recent state writer layout is empty");
  }
  DeepSeekRecentStateWriter writer;
  writer.layout_ = std::move(layout);
  writer.operations_ = &operations;
  return writer;
}

Status DeepSeekRecentStateWriter::launch(
    const DeepSeekRecentStateSubmission& submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  auto status = validate(submission, transaction);
  if (!status.ok()) return status;
  auto bank = transaction.tentative_fixed_state();
  if (!bank.ok()) return bank.status();
  auto layer = layout_.Resolve(submission.layer_id, *bank);
  if (!layer.ok()) return layer.status();
  const auto row = submission.absolute_position % kRecentRows;
  const auto destination = layer->recent_bf16.address +
                           row * kLatentWidth * kBf16Bytes;
  status = operations_->copy_d2d_async(
      destination, submission.latent_bf16, kLatentWidth * kBf16Bytes,
      submission.stream);
  if (!status.ok()) poisoned_ = true;
  return status;
}

Status DeepSeekRecentStateWriter::validate(
    const DeepSeekRecentStateSubmission& submission,
    const DeepSeekAttentionSequenceTransaction& transaction) const {
  if (poisoned_ ||
      transaction.state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      submission.latent_bf16 == 0 ||
      submission.stream != transaction.stream() ||
      submission.batch_count != 1 || submission.absolute_position >= 1048576) {
    return Status::FailedPrecondition(
        "DeepSeek recent state writer is not launchable");
  }
  auto bank = transaction.tentative_fixed_state();
  if (!bank.ok()) return bank.status();
  auto layer = layout_.Resolve(submission.layer_id, *bank);
  if (!layer.ok()) return layer.status();
  if (layer->recent_bf16.address == 0 ||
      layer->recent_bf16.bytes != kRecentRows * kLatentWidth * kBf16Bytes) {
    return Status::Internal("DeepSeek recent state view is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
