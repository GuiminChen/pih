#include "pih/model/deepseek_dspark_expert_compute_driver.h"

#include <cmath>

namespace pih {
namespace {

bool fits(const DeepSeekExpertArenaSpan& span, std::uint64_t bytes) {
  return span.address != 0 && span.bytes >= bytes;
}

bool valid_arena(const DeepSeekExpertComputeArena& arena) {
  constexpr auto tokens = static_cast<std::uint64_t>(
      DeepSeekDsparkExpertComputeDriver::kTokenCount);
  return fits(arena.route_input_bf16, tokens * 4096U * 2U) &&
         arena.expert_output_bf16.address ==
             arena.route_input_bf16.address &&
         fits(arena.expert_output_bf16, tokens * 4096U * 2U) &&
         fits(arena.activation_e4m3, tokens * 4096U) &&
         fits(arena.activation_scale_bits, tokens * 32U) &&
         fits(arena.gate_or_middle_bf16, tokens * 2048U * 2U) &&
         fits(arena.up_bf16, tokens * 2048U * 2U) &&
         fits(arena.route_weights_f32, tokens * sizeof(float)) &&
         fits(arena.token_indices_u32, tokens * sizeof(std::uint32_t)) &&
         fits(arena.error_flag_u32, sizeof(std::uint32_t));
}

bool valid_matrix(const DeepSeekExpertMatrixDeviceView& matrix) {
  return matrix.packed.address != 0 &&
         matrix.packed.bytes ==
             DeepSeekExpertBundleLayout::kPackedBytesPerMatrix &&
         matrix.scales.address != 0 &&
         matrix.scales.bytes ==
             DeepSeekExpertBundleLayout::kScaleBytesPerMatrix;
}

}  // namespace

Result<DeepSeekDsparkExpertComputeDriver>
DeepSeekDsparkExpertComputeDriver::Create(
    DeepSeekExpertComputeArena arena,
    std::uintptr_t source_hidden_bf16,
    std::uintptr_t accumulator_f32,
    std::uintptr_t stream,
    std::uint64_t context_identity,
    DeepSeekExpertComputeBackend& backend) {
  if (!valid_arena(arena) || source_hidden_bf16 == 0 ||
      accumulator_f32 == 0 || stream == 0 || context_identity == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark expert compute resources are invalid");
  }
  DeepSeekDsparkExpertComputeDriver result;
  result.arena_ = arena;
  result.source_hidden_bf16_ = source_hidden_bf16;
  result.accumulator_f32_ = accumulator_f32;
  result.stream_ = stream;
  result.context_identity_ = context_identity;
  result.backend_ = &backend;
  return result;
}

Status DeepSeekDsparkExpertComputeDriver::launch_resident(
    DeepSeekDsparkStageId stage, std::uint16_t expert,
    std::uint64_t generation,
    const DeepSeekExpertBundleDeviceView& bundle,
    const DeepSeekExpertRoute* routes,
    std::uint32_t route_count) {
  if (state_ != State::kIdle || !is_valid_deepseek_dspark_stage(stage) ||
      expert >= DeepSeekExpertSubwavePlan::kExpertCount ||
      generation == 0 || !valid_matrix(bundle.w1) ||
      !valid_matrix(bundle.w2) || !valid_matrix(bundle.w3) ||
      routes == nullptr || route_count == 0 ||
      route_count > kTokenCount) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark expert compute lane is not launchable");
  }
  std::uint32_t previous_token = 0;
  for (std::uint32_t index = 0; index < route_count; ++index) {
    const auto& route = routes[index];
    if (route.expert_id != expert || route.token_index >= kTokenCount ||
        route.route_ordinal >= DeepSeekExpertSubwavePlan::kRoutesPerToken ||
        !std::isfinite(route.weight) || route.weight < 0.0F ||
        (index != 0 && route.token_index <= previous_token)) {
      return Status::InvalidArgument(
          "DeepSeek DSpark expert route slice is not canonical");
    }
    previous_token = route.token_index;
  }
  const DeepSeekExpertComputeSubmission submission{
      .context_identity = context_identity_,
      .bundle = bundle,
      .arena = arena_,
      .routes = routes,
      .route_count = route_count,
      .packed_token_count = kTokenCount,
      .source_hidden_bf16 = source_hidden_bf16_,
      .accumulator_f32 = accumulator_f32_,
      .stream = stream_,
  };
  const auto status = backend_->submit(submission);
  if (!status.ok()) {
    state_ = State::kPoisoned;
    return status;
  }
  state_ = State::kInflight;
  return Status::Ok();
}

Result<DeepSeekExpertAsyncStatus>
DeepSeekDsparkExpertComputeDriver::poll() {
  if (state_ == State::kPoisoned) {
    return DeepSeekExpertAsyncStatus::kError;
  }
  if (state_ != State::kInflight) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark expert compute lane has no inflight work");
  }
  auto status = backend_->poll();
  if (!status.ok()) {
    state_ = State::kPoisoned;
    return status.status();
  }
  if (*status == DeepSeekExpertAsyncStatus::kSuccess) {
    state_ = State::kIdle;
  } else if (*status == DeepSeekExpertAsyncStatus::kError) {
    state_ = State::kPoisoned;
  } else if (*status != DeepSeekExpertAsyncStatus::kInProgress) {
    state_ = State::kPoisoned;
    return Status::Internal(
        "DeepSeek DSpark expert backend returned invalid async state");
  }
  return *status;
}

}  // namespace pih
