#include "pih/model/deepseek_rank_control_codec.h"

#include <type_traits>

namespace pih {
namespace {

constexpr std::uint32_t kMagic = 0x50524958U;
constexpr std::uint16_t kChallenge = 1;
constexpr std::uint16_t kReady = 2;
constexpr std::uint16_t kPostExecResourceObservation = 3;
constexpr std::uint16_t kPostExecResourceAuthority = 4;
constexpr std::uint16_t kServingCommand = 5;
constexpr std::uint16_t kServingCompletion = 6;
static_assert(kDeepSeekRankPostExecResourceObservationBytes ==
              8 + (2 * 8) + 4 + (2 * 8) + (2 * 32) + (8 * 8) + 4);
static_assert(kDeepSeekRankPostExecResourceAuthorityBytes ==
              8 + (2 * 8) + (2 * 4) + (5 * 8) + (4 * 32) + 8 +
                  (6 * 8));
static_assert(kDeepSeekRankServingCommandBytes ==
              8 + 80 + 8 + 4 + (2 * 8) + 28 + (3 * 8));
static_assert(kDeepSeekRankServingCompletionBytes ==
              8 + 80 + (3 * 8) + 28 + 4 + 8);

template <class T, std::size_t N>
void put(std::array<std::byte, N>& output, std::size_t& offset, T value) {
  using U = std::make_unsigned_t<T>;
  auto bits = static_cast<U>(value);
  for (std::size_t i = 0; i < sizeof(T); ++i)
    output[offset++] = static_cast<std::byte>((bits >> (i * 8U)) & 0xffU);
}

template <class T>
T get(std::span<const std::byte> input, std::size_t& offset) {
  using U = std::make_unsigned_t<T>;
  U value = 0;
  for (std::size_t i = 0; i < sizeof(T); ++i)
    value |= static_cast<U>(std::to_integer<unsigned>(input[offset++])) << (i * 8U);
  return static_cast<T>(value);
}

template <std::size_t N>
void header(std::array<std::byte, N>& output, std::size_t& offset,
            std::uint16_t type) {
  put(output, offset, kMagic); put(output, offset, type);
  put(output, offset, std::uint16_t{1});
}

template <std::size_t N>
void put_digest(std::array<std::byte, N>& output, std::size_t& offset,
                const Sha256Digest& value) {
  for (const auto byte : value.bytes) output[offset++] = byte;
}

Sha256Digest get_digest(std::span<const std::byte> input,
                        std::size_t& offset) {
  Sha256Digest value{};
  for (auto& byte : value.bytes) byte = input[offset++];
  return value;
}

template <std::size_t N>
void put_session(std::array<std::byte, N>& output, std::size_t& offset,
                 const DeepSeekRankServingSessionBinding& session) {
  put(output, offset, session.engine_epoch);
  put(output, offset, session.worker_generation);
  put(output, offset, session.world_size);
  put(output, offset, session.rank);
  put(output, offset, session.process_identity);
  put(output, offset, session.pidfd_identity);
  put(output, offset, session.control_identity);
  put_digest(output, offset, session.materialization_warm_seal_root);
}

DeepSeekRankServingSessionBinding get_session(
    std::span<const std::byte> input, std::size_t& offset) {
  DeepSeekRankServingSessionBinding session;
  session.engine_epoch = get<std::uint64_t>(input, offset);
  session.worker_generation = get<std::uint64_t>(input, offset);
  session.world_size = get<std::uint32_t>(input, offset);
  session.rank = get<std::uint32_t>(input, offset);
  session.process_identity = get<std::uint64_t>(input, offset);
  session.pidfd_identity = get<std::uint64_t>(input, offset);
  session.control_identity = get<std::uint64_t>(input, offset);
  session.materialization_warm_seal_root = get_digest(input, offset);
  return session;
}

template <std::size_t N>
void put_plan(std::array<std::byte, N>& output, std::size_t& offset,
              const DeepSeekPipelinePlanDescriptor& plan) {
  put(output, offset, plan.engine_epoch);
  put(output, offset, plan.plan_sequence);
  put(output, offset, static_cast<std::uint32_t>(plan.phase));
  put(output, offset, plan.token_count);
  put(output, offset, plan.sequence_count);
}

Result<DeepSeekPipelinePlanDescriptor> get_plan(
    std::span<const std::byte> input, std::size_t& offset) {
  DeepSeekPipelinePlanDescriptor plan;
  plan.engine_epoch = get<std::uint64_t>(input, offset);
  plan.plan_sequence = get<std::uint64_t>(input, offset);
  const auto phase = get<std::uint32_t>(input, offset);
  if (phase > static_cast<std::uint32_t>(DeepSeekPlanPhase::kDrain)) {
    return Status::InvalidArgument("DeepSeek rank serving plan phase is invalid");
  }
  plan.phase = static_cast<DeepSeekPlanPhase>(phase);
  plan.token_count = get<std::uint32_t>(input, offset);
  plan.sequence_count = get<std::uint32_t>(input, offset);
  return plan;
}

Status validate_header(std::span<const std::byte> input, std::size_t expected,
                       std::uint16_t type, std::size_t& offset) {
  if (input.size() != expected) return Status::InvalidArgument("DeepSeek rank control frame size is invalid");
  if (get<std::uint32_t>(input, offset) != kMagic ||
      get<std::uint16_t>(input, offset) != type ||
      get<std::uint16_t>(input, offset) != 1)
    return Status::InvalidArgument("DeepSeek rank control frame header is invalid");
  return Status::Ok();
}

}  // namespace

std::array<std::byte, kDeepSeekRankChallengeBytes>
encode_deepseek_rank_challenge(const DeepSeekRankExecChallenge& v) {
  std::array<std::byte, kDeepSeekRankChallengeBytes> out{}; std::size_t o = 0;
  header(out, o, kChallenge); put(out, o, v.manifest.engine_epoch);
  put(out, o, v.manifest.worker_generation); put(out, o, v.manifest.world_size);
  put(out, o, v.manifest.rank); put(out, o, v.manifest.physical_device_identity);
  put(out, o, v.manifest.process_manifest_identity); put(out, o, v.handle.process_identity);
  put(out, o, v.handle.pidfd_identity); put(out, o, v.handle.control_identity);
  put(out, o, v.controller_process_identity); put(out, o, v.challenge_identity);
  put(out, o, v.manifest.startup_device_ordinal);
  put(out, o, v.manifest.startup_deadline_ns);
  put_digest(out, o, v.manifest.physical_device_uuid_commitment);
  return out;
}

Result<DeepSeekRankExecChallenge> decode_deepseek_rank_challenge(
    std::span<const std::byte> in) {
  std::size_t o = 0; auto status = validate_header(in, kDeepSeekRankChallengeBytes, kChallenge, o);
  if (!status.ok()) return status;
  DeepSeekRankExecChallenge v; v.protocol_version = 1;
  v.manifest.engine_epoch = get<std::uint64_t>(in, o);
  v.manifest.worker_generation = get<std::uint64_t>(in, o);
  v.manifest.world_size = get<std::uint32_t>(in, o); v.manifest.rank = get<std::uint32_t>(in, o);
  v.manifest.physical_device_identity = get<std::uint64_t>(in, o);
  v.manifest.process_manifest_identity = get<std::uint64_t>(in, o);
  v.handle.process_identity = get<std::uint64_t>(in, o); v.handle.pidfd_identity = get<std::uint64_t>(in, o);
  v.handle.control_identity = get<std::uint64_t>(in, o); v.controller_process_identity = get<std::uint64_t>(in, o);
  v.challenge_identity = get<std::uint64_t>(in, o);
  v.manifest.startup_device_ordinal = get<std::int32_t>(in, o);
  v.manifest.startup_deadline_ns = get<std::uint64_t>(in, o);
  v.manifest.physical_device_uuid_commitment = get_digest(in, o); return v;
}

std::array<std::byte, kDeepSeekRankReadyBytes>
encode_deepseek_rank_ready(const DeepSeekRankExecReady& v) {
  std::array<std::byte, kDeepSeekRankReadyBytes> out{}; std::size_t o = 0;
  header(out, o, kReady); const auto& r = v.receipt;
  put(out, o, r.engine_epoch); put(out, o, r.worker_generation); put(out, o, r.rank);
  put(out, o, r.physical_device_identity); put(out, o, r.process_manifest_identity);
  put(out, o, r.process_identity); put(out, o, r.pidfd_identity); put(out, o, r.control_identity);
  put(out, o, v.challenge_identity);
  put(out, o, r.startup_device_ordinal);
  put(out, o, r.startup_deadline_ns);
  put_digest(out, o, r.physical_device_uuid_commitment); return out;
}

Result<DeepSeekRankExecReady> decode_deepseek_rank_ready(std::span<const std::byte> in) {
  std::size_t o = 0; auto status = validate_header(in, kDeepSeekRankReadyBytes, kReady, o);
  if (!status.ok()) return status;
  DeepSeekRankExecReady v; auto& r = v.receipt;
  r.engine_epoch = get<std::uint64_t>(in, o); r.worker_generation = get<std::uint64_t>(in, o);
  r.rank = get<std::uint32_t>(in, o); r.physical_device_identity = get<std::uint64_t>(in, o);
  r.process_manifest_identity = get<std::uint64_t>(in, o); r.process_identity = get<std::uint64_t>(in, o);
  r.pidfd_identity = get<std::uint64_t>(in, o); r.control_identity = get<std::uint64_t>(in, o);
  v.challenge_identity = get<std::uint64_t>(in, o);
  r.startup_device_ordinal = get<std::int32_t>(in, o);
  r.startup_deadline_ns = get<std::uint64_t>(in, o);
  r.physical_device_uuid_commitment = get_digest(in, o); return v;
}

std::array<std::byte, kDeepSeekRankPostExecResourceObservationBytes>
encode_deepseek_rank_post_exec_resource_observation(
    const DeepSeekRankPostExecResourceObservation& v) {
  std::array<std::byte, kDeepSeekRankPostExecResourceObservationBytes> out{};
  std::size_t o = 0;
  header(out, o, kPostExecResourceObservation);
  put(out, o, v.engine_epoch);
  put(out, o, v.worker_generation);
  put(out, o, v.rank);
  put(out, o, v.process_identity);
  put(out, o, v.challenge_identity);
  put_digest(out, o, v.acknowledged_capacity_plan_instance_root);
  put_digest(out, o, v.acknowledged_os_resource_envelope_root);
  put(out, o, v.task_count);
  put(out, o, v.open_fd_count);
  put(out, o, v.scm_rights_inflight_fd_count);
  put(out, o, v.vma_count);
  put(out, o, v.rlimit_nofile_soft);
  put(out, o, v.rlimit_nofile_hard);
  put(out, o, v.fs_nr_open);
  put(out, o, v.vm_max_map_count);
  put(out, o, static_cast<std::uint32_t>(v.non_dumpable ? 1U : 0U));
  return out;
}

Result<DeepSeekRankPostExecResourceObservation>
decode_deepseek_rank_post_exec_resource_observation(
    std::span<const std::byte> in) {
  std::size_t o = 0;
  auto status = validate_header(
      in, kDeepSeekRankPostExecResourceObservationBytes,
      kPostExecResourceObservation, o);
  if (!status.ok()) return status;
  DeepSeekRankPostExecResourceObservation v{};
  v.engine_epoch = get<std::uint64_t>(in, o);
  v.worker_generation = get<std::uint64_t>(in, o);
  v.rank = get<std::uint32_t>(in, o);
  v.process_identity = get<std::uint64_t>(in, o);
  v.challenge_identity = get<std::uint64_t>(in, o);
  v.acknowledged_capacity_plan_instance_root = get_digest(in, o);
  v.acknowledged_os_resource_envelope_root = get_digest(in, o);
  v.task_count = get<std::uint64_t>(in, o);
  v.open_fd_count = get<std::uint64_t>(in, o);
  v.scm_rights_inflight_fd_count = get<std::uint64_t>(in, o);
  v.vma_count = get<std::uint64_t>(in, o);
  v.rlimit_nofile_soft = get<std::uint64_t>(in, o);
  v.rlimit_nofile_hard = get<std::uint64_t>(in, o);
  v.fs_nr_open = get<std::uint64_t>(in, o);
  v.vm_max_map_count = get<std::uint64_t>(in, o);
  const auto non_dumpable = get<std::uint32_t>(in, o);
  if (non_dumpable > 1) {
    return Status::InvalidArgument(
        "DeepSeek rank post-exec non-dumpable encoding is invalid");
  }
  v.non_dumpable = non_dumpable == 1;
  return v;
}

std::array<std::byte, kDeepSeekRankPostExecResourceAuthorityBytes>
encode_deepseek_rank_post_exec_resource_authority(
    const DeepSeekRankPostExecResourceAuthority& v) {
  std::array<std::byte, kDeepSeekRankPostExecResourceAuthorityBytes> out{};
  std::size_t o = 0;
  header(out, o, kPostExecResourceAuthority);
  put(out, o, v.engine_epoch);
  put(out, o, v.worker_generation);
  put(out, o, v.world_size);
  put(out, o, v.rank);
  put(out, o, v.process_manifest_identity);
  put(out, o, v.process_identity);
  put(out, o, v.pidfd_identity);
  put(out, o, v.control_identity);
  put(out, o, v.challenge_identity);
  put_digest(out, o, v.manifest_root);
  put_digest(out, o, v.spawn_resource_plan_root);
  put_digest(out, o, v.capacity_plan_instance_root);
  put_digest(out, o, v.os_resource_envelope_root);
  put(out, o, v.deadline_ns);
  put(out, o, v.resource_plan.worker_task_peak);
  put(out, o, v.resource_plan.worker_fd_peak);
  put(out, o, v.resource_plan.worker_vma_peak);
  put(out, o, v.resource_plan.task_emergency_reserve);
  put(out, o, v.resource_plan.fd_emergency_reserve);
  put(out, o, v.resource_plan.vma_emergency_reserve);
  return out;
}

Result<DeepSeekRankPostExecResourceAuthority>
decode_deepseek_rank_post_exec_resource_authority(
    std::span<const std::byte> in) {
  std::size_t o = 0;
  auto status = validate_header(
      in, kDeepSeekRankPostExecResourceAuthorityBytes,
      kPostExecResourceAuthority, o);
  if (!status.ok()) return status;
  DeepSeekRankPostExecResourceAuthority v{};
  v.protocol_version = 1;
  v.engine_epoch = get<std::uint64_t>(in, o);
  v.worker_generation = get<std::uint64_t>(in, o);
  v.world_size = get<std::uint32_t>(in, o);
  v.rank = get<std::uint32_t>(in, o);
  v.process_manifest_identity = get<std::uint64_t>(in, o);
  v.process_identity = get<std::uint64_t>(in, o);
  v.pidfd_identity = get<std::uint64_t>(in, o);
  v.control_identity = get<std::uint64_t>(in, o);
  v.challenge_identity = get<std::uint64_t>(in, o);
  v.manifest_root = get_digest(in, o);
  v.spawn_resource_plan_root = get_digest(in, o);
  v.capacity_plan_instance_root = get_digest(in, o);
  v.os_resource_envelope_root = get_digest(in, o);
  v.deadline_ns = get<std::uint64_t>(in, o);
  v.resource_plan.worker_task_peak = get<std::uint64_t>(in, o);
  v.resource_plan.worker_fd_peak = get<std::uint64_t>(in, o);
  v.resource_plan.worker_vma_peak = get<std::uint64_t>(in, o);
  v.resource_plan.task_emergency_reserve = get<std::uint64_t>(in, o);
  v.resource_plan.fd_emergency_reserve = get<std::uint64_t>(in, o);
  v.resource_plan.vma_emergency_reserve = get<std::uint64_t>(in, o);
  v.resource_plan.os_resource_envelope_root =
      v.os_resource_envelope_root;
  return v;
}

std::array<std::byte, kDeepSeekRankServingCommandBytes>
encode_deepseek_rank_serving_command(const DeepSeekRankServingCommand& value) {
  std::array<std::byte, kDeepSeekRankServingCommandBytes> output{};
  std::size_t offset = 0;
  header(output, offset, kServingCommand);
  put_session(output, offset, value.session);
  put(output, offset, value.command_sequence);
  put(output, offset, static_cast<std::uint32_t>(value.kind));
  put(output, offset, value.request_id);
  put(output, offset, value.request_generation);
  put_plan(output, offset, value.plan);
  put(output, offset, value.target_execution_sequence);
  put(output, offset, value.input_lease_identity);
  put(output, offset, value.output_lease_identity);
  return output;
}

Result<DeepSeekRankServingCommand> decode_deepseek_rank_serving_command(
    std::span<const std::byte> input) {
  std::size_t offset = 0;
  auto status = validate_header(input, kDeepSeekRankServingCommandBytes,
                                kServingCommand, offset);
  if (!status.ok()) return status;
  DeepSeekRankServingCommand command;
  command.session = get_session(input, offset);
  command.command_sequence = get<std::uint64_t>(input, offset);
  const auto kind = get<std::uint32_t>(input, offset);
  if (kind > static_cast<std::uint32_t>(DeepSeekRankServingCommandKind::kCancel)) {
    return Status::InvalidArgument("DeepSeek rank serving command kind is invalid");
  }
  command.kind = static_cast<DeepSeekRankServingCommandKind>(kind);
  command.request_id = get<std::uint64_t>(input, offset);
  command.request_generation = get<std::uint64_t>(input, offset);
  auto plan = get_plan(input, offset);
  if (!plan.ok()) return plan.status();
  command.plan = *plan;
  command.target_execution_sequence = get<std::uint64_t>(input, offset);
  command.input_lease_identity = get<std::uint64_t>(input, offset);
  command.output_lease_identity = get<std::uint64_t>(input, offset);
  return command;
}

std::array<std::byte, kDeepSeekRankServingCompletionBytes>
encode_deepseek_rank_serving_completion(
    const DeepSeekRankServingCompletion& value) {
  std::array<std::byte, kDeepSeekRankServingCompletionBytes> output{};
  std::size_t offset = 0;
  header(output, offset, kServingCompletion);
  put_session(output, offset, value.session);
  put(output, offset, value.execution_command_sequence);
  put(output, offset, value.request_id);
  put(output, offset, value.request_generation);
  put_plan(output, offset, value.plan);
  put(output, offset, static_cast<std::uint32_t>(value.outcome));
  put(output, offset, value.output_lease_identity);
  return output;
}

Result<DeepSeekRankServingCompletion> decode_deepseek_rank_serving_completion(
    std::span<const std::byte> input) {
  std::size_t offset = 0;
  auto status = validate_header(input, kDeepSeekRankServingCompletionBytes,
                                kServingCompletion, offset);
  if (!status.ok()) return status;
  DeepSeekRankServingCompletion completion;
  completion.session = get_session(input, offset);
  completion.execution_command_sequence = get<std::uint64_t>(input, offset);
  completion.request_id = get<std::uint64_t>(input, offset);
  completion.request_generation = get<std::uint64_t>(input, offset);
  auto plan = get_plan(input, offset);
  if (!plan.ok()) return plan.status();
  completion.plan = *plan;
  const auto outcome = get<std::uint32_t>(input, offset);
  if (outcome > static_cast<std::uint32_t>(
                    DeepSeekRankServingCompletionOutcome::kFailed)) {
    return Status::InvalidArgument(
        "DeepSeek rank serving completion outcome is invalid");
  }
  completion.outcome = static_cast<DeepSeekRankServingCompletionOutcome>(outcome);
  completion.output_lease_identity = get<std::uint64_t>(input, offset);
  return completion;
}

}  // namespace pih
