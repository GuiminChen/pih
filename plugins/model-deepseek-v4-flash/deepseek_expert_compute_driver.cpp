#include "pih/model/deepseek_expert_compute_driver.h"

#include <cmath>

namespace pih {
namespace {

bool arena_fits(const DeepSeekExpertComputeArena& arena,
                std::uint32_t tokens) {
  const auto count = static_cast<std::uint64_t>(tokens);
  return arena.route_input_bf16.address != 0 &&
         arena.route_input_bf16.bytes >= count * 4096U * 2U &&
         arena.expert_output_bf16.address == arena.route_input_bf16.address &&
         arena.expert_output_bf16.bytes >= count * 4096U * 2U &&
         arena.activation_e4m3.address != 0 &&
         arena.activation_e4m3.bytes >= count * 4096U &&
         arena.activation_scale_bits.address != 0 &&
         arena.activation_scale_bits.bytes >= count * 32U &&
         arena.gate_or_middle_bf16.address != 0 &&
         arena.gate_or_middle_bf16.bytes >= count * 2048U * 2U &&
         arena.up_bf16.address != 0 &&
         arena.up_bf16.bytes >= count * 2048U * 2U &&
         arena.route_weights_f32.address != 0 &&
         arena.route_weights_f32.bytes >= count * 4U &&
         arena.token_indices_u32.address != 0 &&
         arena.token_indices_u32.bytes >= count * 4U &&
         arena.error_flag_u32.address != 0 &&
         arena.error_flag_u32.bytes >= sizeof(std::uint32_t);
}

}  // namespace

Result<DeepSeekExpertComputeDriver> DeepSeekExpertComputeDriver::Create(
    DeepSeekExpertSlotTable slots, DeepSeekExpertComputeArena arena,
    std::uint32_t packed_token_count, std::uintptr_t source_hidden_bf16,
    std::uintptr_t accumulator_f32, std::uintptr_t stream,
    DeepSeekExpertComputeBackend& backend) {
  if (packed_token_count == 0 || source_hidden_bf16 == 0 ||
      accumulator_f32 == 0 || stream == 0 || !arena_fits(arena, packed_token_count)) {
    return Status::InvalidArgument(
        "DeepSeek expert compute driver resources are invalid");
  }
  DeepSeekExpertComputeDriver driver;
  driver.slots_ = std::move(slots);
  driver.arena_ = arena;
  driver.packed_token_count_ = packed_token_count;
  driver.source_hidden_bf16_ = source_hidden_bf16;
  driver.accumulator_f32_ = accumulator_f32;
  driver.stream_ = stream;
  driver.backend_ = &backend;
  return driver;
}

Status DeepSeekExpertComputeDriver::launch(
    const DeepSeekExpertLease& lease, const DeepSeekExpertRoute* routes,
    std::uint32_t route_count) {
  auto expert = slots_.bind(lease);
  if (!expert.ok()) return expert.status();
  return launch_bound(*expert, routes, route_count);
}

Status DeepSeekExpertComputeDriver::launch_resident(
    DeepSeekExpertIdentity identity, std::uint64_t generation,
    const DeepSeekExpertBundleDeviceView& bundle,
    const DeepSeekExpertRoute* routes, std::uint32_t route_count) {
  auto expert = slots_.bind_resident(identity, generation, bundle);
  if (!expert.ok()) return expert.status();
  return launch_bound(*expert, routes, route_count);
}

Status DeepSeekExpertComputeDriver::launch_bound(
    DeepSeekExpertLeaseDeviceView expert, const DeepSeekExpertRoute* routes,
    std::uint32_t route_count) {
  if (state_ != State::kIdle || routes == nullptr || route_count == 0 ||
      route_count > packed_token_count_) {
    return Status::FailedPrecondition(
        "DeepSeek expert compute lane is not launchable");
  }
  std::uint32_t previous_token = 0;
  for (std::uint32_t index = 0; index < route_count; ++index) {
    const auto& route = routes[index];
    if (route.expert_id != expert.identity.expert ||
        route.token_index >= packed_token_count_ || route.route_ordinal >= 6 ||
        !std::isfinite(route.weight) || route.weight < 0.0F ||
        (index != 0 && route.token_index <= previous_token)) {
      return Status::InvalidArgument(
          "DeepSeek expert compute route slice is not canonical");
    }
    previous_token = route.token_index;
  }
  const DeepSeekExpertComputeSubmission submission{
      .context_identity = expert.context_identity,
      .bundle = expert.bundle,
      .expert = expert,
      .arena = arena_,
      .routes = routes,
      .route_count = route_count,
      .packed_token_count = packed_token_count_,
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

Result<DeepSeekExpertAsyncStatus> DeepSeekExpertComputeDriver::poll() {
  if (state_ == State::kPoisoned) {
    return DeepSeekExpertAsyncStatus::kError;
  }
  if (state_ != State::kInflight) {
    return Status::FailedPrecondition(
        "DeepSeek expert compute lane has no inflight submission");
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
        "DeepSeek expert compute backend returned invalid async state");
  }
  return *status;
}

}  // namespace pih
