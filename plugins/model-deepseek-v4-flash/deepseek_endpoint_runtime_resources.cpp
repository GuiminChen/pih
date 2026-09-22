#include "pih/model/deepseek_endpoint_runtime_resources.h"

#include <cstring>

namespace pih {

Result<DeepSeekEndpointRuntimeResources>
DeepSeekEndpointRuntimeResources::Allocate(
    DeepSeekStagePlan stage, DeepSeekEndpointSequenceOperations* operations,
    RegisteredPinnedAllocator& allocator) {
  if (stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43) {
    return Status::InvalidArgument(
        "DeepSeek endpoint runtime stage is invalid");
  }
  const bool endpoint = stage.owns_embedding || stage.owns_lm_head;
  if (!endpoint) {
    return DeepSeekEndpointRuntimeResources(
        stage, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
  }
  if (operations == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek endpoint runtime operations are unavailable");
  }
  auto error = Buffer::Allocate(allocator, sizeof(std::uint32_t), 256);
  if (!error.ok()) return error.status();
  if (error->data() == nullptr || error->generation() == 0 ||
      error->device().type() != DeviceType::kCpu) {
    return Status::FailedPrecondition(
        "DeepSeek endpoint pinned error allocation is invalid");
  }
  std::memset(error->data(), 0, sizeof(std::uint32_t));
  auto owned_error = std::make_unique<Buffer>(std::move(*error));
  auto sampled = Buffer::Allocate(allocator, sizeof(std::uint32_t), 256);
  if (!sampled.ok()) return sampled.status();
  auto owned_sampled = std::make_unique<Buffer>(std::move(*sampled));
  std::memset(owned_sampled->data(), 0, sizeof(std::uint32_t));
  auto logprob = Buffer::Allocate(allocator, sizeof(float), 256);
  if (!logprob.ok()) return logprob.status();
  auto owned_logprob = std::make_unique<Buffer>(std::move(*logprob));
  std::memset(owned_logprob->data(), 0, sizeof(float));
  auto rng = Buffer::Allocate(allocator, sizeof(std::uint32_t), 256);
  if (!rng.ok()) return rng.status();
  auto owned_rng = std::make_unique<Buffer>(std::move(*rng));
  std::memset(owned_rng->data(), 0, sizeof(std::uint32_t));
  auto top_ids = Buffer::Allocate(
      allocator, 20U * sizeof(std::uint32_t), 256);
  if (!top_ids.ok()) return top_ids.status();
  auto owned_top_ids = std::make_unique<Buffer>(std::move(*top_ids));
  std::memset(owned_top_ids->data(), 0, 20U * sizeof(std::uint32_t));
  auto top_logprobs = Buffer::Allocate(allocator, 20U * sizeof(float), 256);
  if (!top_logprobs.ok()) return top_logprobs.status();
  auto owned_top_logprobs =
      std::make_unique<Buffer>(std::move(*top_logprobs));
  std::memset(owned_top_logprobs->data(), 0, 20U * sizeof(float));
  auto executor = DeepSeekEndpointSequenceExecutor::Create(
      *operations, static_cast<std::uint32_t*>(owned_error->data()),
      static_cast<std::uint32_t*>(owned_sampled->data()),
      static_cast<float*>(owned_logprob->data()),
      static_cast<std::uint32_t*>(owned_rng->data()),
      static_cast<std::uint32_t*>(owned_top_ids->data()),
      static_cast<float*>(owned_top_logprobs->data()));
  if (!executor.ok()) return executor.status();
  return DeepSeekEndpointRuntimeResources(
      stage, std::move(owned_error),
      std::move(owned_sampled),
      std::move(owned_logprob), std::move(owned_rng),
      std::move(owned_top_ids), std::move(owned_top_logprobs),
      std::make_unique<DeepSeekEndpointSequenceExecutor>(
          std::move(*executor)));
}

Result<DeepSeekSamplingResult>
DeepSeekEndpointRuntimeResources::sampling_result(
    std::uint32_t top_logprobs_count) const {
  if (top_logprobs_count > 20U) {
    return Status::InvalidArgument(
        "DeepSeek top-logprob result count exceeds 20");
  }
  auto token = sampled_token();
  if (!token.ok()) return token.status();
  if (host_selected_logprob_ == nullptr || host_rng_word_ == nullptr ||
      host_selected_logprob_->data() == nullptr ||
      host_rng_word_->data() == nullptr || host_top_ids_ == nullptr ||
      host_top_logprobs_ == nullptr || host_top_ids_->data() == nullptr ||
      host_top_logprobs_->data() == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek sampling receipt is unavailable on this rank");
  }
  DeepSeekSamplingResult result{
      *token, *static_cast<const float*>(host_selected_logprob_->data()),
      *static_cast<const std::uint32_t*>(host_rng_word_->data())};
  const auto* ids = static_cast<const std::uint32_t*>(host_top_ids_->data());
  const auto* values = static_cast<const float*>(host_top_logprobs_->data());
  result.top_token_ids.assign(ids, ids + top_logprobs_count);
  result.top_logprobs.assign(values, values + top_logprobs_count);
  return result;
}

Result<std::uint32_t> DeepSeekEndpointRuntimeResources::sampled_token() const {
  if (!stage_.owns_lm_head || host_sampled_token_ == nullptr ||
      host_sampled_token_->data() == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek sampled token is unavailable on this rank");
  }
  return *static_cast<const std::uint32_t*>(host_sampled_token_->data());
}

}  // namespace pih
