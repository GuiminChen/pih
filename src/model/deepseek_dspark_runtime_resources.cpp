#include "pih/model/deepseek_dspark_runtime_resources.h"

#include <cstring>

namespace pih {
namespace {

constexpr std::uint64_t kRouterScoreOffset = 256;
constexpr std::uint64_t kRouterScoreCount = 5U * 256U;
constexpr std::uint64_t kRouterBiasOffset =
    kRouterScoreOffset + kRouterScoreCount * sizeof(float);
constexpr std::uint64_t kHostBytes =
    kRouterBiasOffset + 256U * sizeof(float);

}  // namespace

Result<DeepSeekDsparkRuntimeResources> DeepSeekDsparkRuntimeResources::Allocate(
    DeepSeekStagePlan stage, DeepSeekDsparkEmbedOperations* embed_operations,
    DeepSeekDsparkHeadOperations* head_operations,
    DeepSeekDsparkPrefillStageOperations* prefill_operations,
    DeepSeekDsparkDecodeStageOperations* decode_operations,
    DeepSeekDsparkMoeStageOperations* moe_operations,
    RegisteredPinnedAllocator& allocator) {
  if (stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43) {
    return Status::InvalidArgument("DeepSeek DSpark runtime stage is invalid");
  }
  if (!stage.owns_dspark) {
    return DeepSeekDsparkRuntimeResources(
        nullptr, nullptr, nullptr, {}, {}, {});
  }
  if (!stage.owns_lm_head || stage.layers.last_layer != 42 ||
      embed_operations == nullptr || head_operations == nullptr ||
      prefill_operations == nullptr || decode_operations == nullptr ||
      moe_operations == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek DSpark runtime operations are unavailable");
  }
  auto error = Buffer::Allocate(allocator, kHostBytes, 256);
  if (!error.ok()) return error.status();
  if (error->data() == nullptr || error->generation() == 0 ||
      error->device().type() != DeviceType::kCpu) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark pinned error allocation is invalid");
  }
  std::memset(error->data(), 0, static_cast<std::size_t>(kHostBytes));
  auto owned_error = std::make_unique<Buffer>(std::move(*error));
  auto* flag = static_cast<std::uint32_t*>(owned_error->data());
  auto embed = DeepSeekDsparkEmbedCoordinator::Create(*embed_operations, flag);
  if (!embed.ok()) return embed.status();
  auto head = DeepSeekDsparkHeadExecutor::Create(*head_operations, flag);
  if (!head.ok()) return head.status();
  std::array<std::optional<DeepSeekDsparkPrefillStageOperation>, 3> prefill;
  std::array<std::optional<DeepSeekDsparkDecodeStageOperation>, 3>
      decode_attention;
  std::array<std::optional<DeepSeekDsparkMoeStageOperation>, 3> moe;
  for (std::size_t index = 0; index < prefill.size(); ++index) {
    auto& operation = prefill[index];
    auto value = DeepSeekDsparkPrefillStageOperation::Create(
        *prefill_operations, flag);
    if (!value.ok()) return value.status();
    operation.emplace(std::move(*value));
    const auto stage_id = static_cast<DeepSeekDsparkStageId>(index);
    auto decode = DeepSeekDsparkDecodeStageOperation::Create(
        stage_id, *decode_operations, flag);
    if (!decode.ok()) return decode.status();
    decode_attention[index].emplace(std::move(*decode));
    auto moe_value = DeepSeekDsparkMoeStageOperation::Create(
        stage_id, *moe_operations, flag);
    if (!moe_value.ok()) return moe_value.status();
    moe[index].emplace(std::move(*moe_value));
  }
  return DeepSeekDsparkRuntimeResources(
      std::move(owned_error),
      std::make_unique<DeepSeekDsparkEmbedCoordinator>(std::move(*embed)),
      std::make_unique<DeepSeekDsparkHeadExecutor>(std::move(*head)),
      std::move(prefill), std::move(decode_attention), std::move(moe));
}

DeepSeekDsparkMtpStageOperation*
DeepSeekDsparkRuntimeResources::moe_operation(
    DeepSeekDsparkStageId stage) noexcept {
  if (!is_valid_deepseek_dspark_stage(stage)) return nullptr;
  auto& operation = moe_[static_cast<std::size_t>(stage)];
  return operation.has_value() ? &*operation : nullptr;
}

std::span<float> DeepSeekDsparkRuntimeResources::router_host_scores()
    noexcept {
  if (host_error_ == nullptr) return {};
  auto* base = static_cast<std::byte*>(host_error_->data());
  return {reinterpret_cast<float*>(base + kRouterScoreOffset),
          static_cast<std::size_t>(kRouterScoreCount)};
}

std::span<float> DeepSeekDsparkRuntimeResources::router_host_bias() noexcept {
  if (host_error_ == nullptr) return {};
  auto* base = static_cast<std::byte*>(host_error_->data());
  return {reinterpret_cast<float*>(base + kRouterBiasOffset), 256U};
}

DeepSeekDsparkMtpStageOperation*
DeepSeekDsparkRuntimeResources::prefill_operation(
    DeepSeekDsparkStageId stage) noexcept {
  if (!is_valid_deepseek_dspark_stage(stage)) return nullptr;
  auto& operation = prefill_[static_cast<std::size_t>(stage)];
  return operation.has_value() ? &*operation : nullptr;
}

DeepSeekDsparkMtpStageOperation*
DeepSeekDsparkRuntimeResources::decode_attention_operation(
    DeepSeekDsparkStageId stage) noexcept {
  if (!is_valid_deepseek_dspark_stage(stage)) return nullptr;
  auto& operation = decode_attention_[static_cast<std::size_t>(stage)];
  return operation.has_value() ? &*operation : nullptr;
}

}  // namespace pih
