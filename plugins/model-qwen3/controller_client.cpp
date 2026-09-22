#include "controller_client.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace pih::qwen_plugin {
namespace {
static_assert(static_cast<uint32_t>(ControllerRuntimeStep::kIdle) ==
              PIH_EXECUTION_STEP_IDLE_V1);
static_assert(static_cast<uint32_t>(ControllerRuntimeStep::kAdmissionRejected) ==
              PIH_EXECUTION_STEP_ADMISSION_REJECTED_V1);
static_assert(static_cast<uint32_t>(ControllerOutputEventKind::kAdmitted) ==
              PIH_EXECUTION_EVENT_ADMITTED_V1);
static_assert(static_cast<uint32_t>(ControllerOutputEventKind::kFailed) ==
              PIH_EXECUTION_EVENT_FAILED_V1);
static_assert(static_cast<uint32_t>(ControllerFinishReason::kLength) ==
              PIH_EXECUTION_FINISH_LENGTH_V1);
std::atomic<const pih_execution_controller_api_v1*> bound_api{nullptr};

Sha256Digest Digest(const uint8_t* input) noexcept {
  Sha256Digest result{};
  std::memcpy(result.bytes.data(), input, result.bytes.size());
  return result;
}
void CopyDigest(uint8_t* output, const Sha256Digest& digest) noexcept {
  std::memcpy(output, digest.bytes.data(), digest.bytes.size());
}
Status FromWire(const pih_status_v1& wire) {
  if (!pih_status_is_valid_v1(&wire))
    return Status::Internal("execution controller returned an invalid status");
  const std::string message(wire.message,
                            ::strnlen(wire.message, sizeof(wire.message)));
  switch (wire.code) {
    case PIH_STATUS_OK_V1: return Status::Ok();
    case PIH_STATUS_INVALID_ARGUMENT_V1: return Status::InvalidArgument(message);
    case PIH_STATUS_FAILED_PRECONDITION_V1: return Status::FailedPrecondition(message);
    case PIH_STATUS_INTERNAL_V1: return Status::Internal(message);
    case PIH_STATUS_UNAVAILABLE_V1: return Status::Unavailable(message);
    case PIH_STATUS_RESOURCE_EXHAUSTED_V1: return Status::ResourceExhausted(message);
    case PIH_STATUS_DEADLINE_EXCEEDED_V1: return Status::DeadlineExceeded(message);
  }
  return Status::Internal("execution controller status code invalid");
}
pih_status_v1 ToWire(const Status& status) noexcept {
  pih_status_v1 wire{};
  wire.struct_size = sizeof(wire);
  wire.abi_version = PIH_STATUS_ABI_VERSION_V1;
  switch (status.code()) {
    case StatusCode::kOk: wire.code = PIH_STATUS_OK_V1; break;
    case StatusCode::kInvalidArgument: wire.code = PIH_STATUS_INVALID_ARGUMENT_V1; break;
    case StatusCode::kFailedPrecondition: wire.code = PIH_STATUS_FAILED_PRECONDITION_V1; break;
    case StatusCode::kInternal: wire.code = PIH_STATUS_INTERNAL_V1; break;
    case StatusCode::kUnavailable: wire.code = PIH_STATUS_UNAVAILABLE_V1; break;
    case StatusCode::kResourceExhausted: wire.code = PIH_STATUS_RESOURCE_EXHAUSTED_V1; break;
    case StatusCode::kDeadlineExceeded: wire.code = PIH_STATUS_DEADLINE_EXCEEDED_V1; break;
  }
  std::memcpy(wire.message, status.message().data(),
              std::min(status.message().size(), sizeof(wire.message) - 1));
  return wire;
}
pih_execution_sampling_v1 ToWire(const ControllerRequestSampling& input) noexcept {
  pih_execution_sampling_v1 wire{};
  wire.struct_size = sizeof(wire);
  wire.abi_version = PIH_EXECUTION_CONTROLLER_ABI_V1;
  wire.mode = static_cast<uint32_t>(input.mode);
  wire.temperature = input.temperature;
  wire.top_p = input.top_p;
  wire.has_top_k = input.top_k.has_value();
  wire.top_k = input.top_k.value_or(0);
  wire.effective_seed = input.effective_seed;
  wire.sample_ordinal = input.sample_ordinal;
  wire.minimum_new_tokens = input.minimum_new_tokens;
  wire.logprobs_enabled = input.logprobs_enabled;
  wire.top_logprobs_count = input.top_logprobs_count;
  wire.stop_token_count = input.stop_token_count;
  if (input.stop_token_count <= input.stop_token_ids.size())
    std::copy_n(input.stop_token_ids.begin(), input.stop_token_count,
                wire.stop_token_ids);
  return wire;
}
Result<ControllerRequestSampling> FromWire(const pih_execution_sampling_v1& wire) {
  if (wire.struct_size != sizeof(wire) ||
      wire.abi_version != PIH_EXECUTION_CONTROLLER_ABI_V1 ||
      wire.mode > 1 || wire.has_top_k > 1 || wire.logprobs_enabled > 1 ||
      wire.stop_token_count > ControllerRequestSampling::kMaximumStopTokenIds)
    return Status::Internal("execution sampling projection invalid");
  ControllerRequestSampling result{};
  result.mode = static_cast<ControllerSamplingMode>(wire.mode);
  result.temperature = wire.temperature;
  result.top_p = wire.top_p;
  if (wire.has_top_k) result.top_k = wire.top_k;
  result.effective_seed = wire.effective_seed;
  result.sample_ordinal = wire.sample_ordinal;
  result.minimum_new_tokens = wire.minimum_new_tokens;
  result.logprobs_enabled = wire.logprobs_enabled != 0;
  result.top_logprobs_count = wire.top_logprobs_count;
  result.stop_token_count = wire.stop_token_count;
  std::copy_n(wire.stop_token_ids, wire.stop_token_count,
              result.stop_token_ids.begin());
  return result;
}
bool ValidApi(const pih_execution_controller_api_v1& api) noexcept {
  return api.struct_size == sizeof(api) &&
      api.contract_version == PIH_EXECUTION_CONTROLLER_ABI_V1 && api.context &&
      api.create && api.destroy && api.submit_admit && api.submit_cancel &&
      api.step && api.prepare && api.commit && api.mark_in_flight && api.abort &&
      api.complete_sampled_tokens && api.acknowledge_output &&
      api.finalize_draining && api.take_event && api.counts;
}

}  // namespace

ControllerClient::ControllerClient(const pih_execution_controller_api_v1& api,
    pih_execution_controller_v1* handle, ControllerRuntimeLimits limits) noexcept
    : api_(api), handle_(handle), limits_(std::move(limits)) {}

ControllerClient::ControllerClient(ControllerClient&& other) noexcept
    : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)),
      limits_(std::move(other.limits_)),
      plan_(std::move(other.plan_)), plan_inputs_(std::move(other.plan_inputs_)),
      input_tokens_(std::move(other.input_tokens_)),
      request_index_(std::move(other.request_index_)),
      query_offsets_(std::move(other.query_offsets_)),
      sample_rows_(std::move(other.sample_rows_)),
      selected_slots_(std::move(other.selected_slots_)),
      positions_(std::move(other.positions_)),
      selected_sampling_(std::move(other.selected_sampling_)) {}

ControllerClient::~ControllerClient() {
  if (handle_) {
    try { (void)close(); } catch (...) { /* Provider retains an unretired handle. */ }
  }
}

Status ControllerClient::Bind(const pih_execution_controller_api_v1* api) {
  if (!api || !ValidApi(*api))
    return Status::FailedPrecondition("execution controller capability invalid");
  const pih_execution_controller_api_v1* expected = nullptr;
  if (!bound_api.compare_exchange_strong(expected, api))
    return Status::FailedPrecondition("execution controller already bound");
  return Status::Ok();
}
void ControllerClient::Unbind() noexcept { bound_api.store(nullptr); }
Result<ControllerClient> ControllerClient::CreateBound(
    const ControllerRuntimeLimits& limits) {
  const auto* api = bound_api.load();
  if (!api) return Status::FailedPrecondition("execution controller not bound");
  return Create(*api, limits);
}

Result<ControllerClient> ControllerClient::Create(
    const pih_execution_controller_api_v1& api,
    const ControllerRuntimeLimits& limits) {
  if (!ValidApi(api) || limits.profile_revision.empty() ||
      limits.profile_revision.size() > 128 || limits.maximum_sequences == 0)
    return Status::InvalidArgument("execution controller capability or limits invalid");
  pih_execution_controller_limits_v1 wire{};
  wire.struct_size = sizeof(wire);
  wire.abi_version = PIH_EXECUTION_CONTROLLER_ABI_V1;
  wire.epoch = limits.epoch;
  wire.maximum_sequences = limits.maximum_sequences;
  wire.command_capacity = limits.ingress.mailbox.command_capacity;
  wire.event_capacity = limits.ingress.mailbox.event_capacity;
  wire.maximum_requests = limits.ingress.requests.maximum_requests;
  wire.maximum_prompt_tokens_per_request =
      limits.ingress.requests.maximum_prompt_tokens_per_request;
  wire.maximum_context_tokens = limits.ingress.requests.maximum_context_tokens;
  wire.maximum_candidates = limits.scheduler.maximum_candidates;
  wire.maximum_sequences_per_plan = limits.scheduler.maximum_sequences_per_plan;
  wire.maximum_real_tokens_per_plan = limits.scheduler.maximum_real_tokens_per_plan;
  wire.scheduler_scan_limit = limits.scheduler.scheduler_scan_limit;
  wire.max_consecutive_decode_rounds = limits.scheduler.max_consecutive_decode_rounds;
  wire.prefill_starvation_threshold_ns = limits.scheduler.prefill_starvation_threshold_ns;
  wire.metadata_maximum_sequences = limits.metadata.maximum_sequences;
  wire.metadata_maximum_execution_bucket_tokens =
      limits.metadata.maximum_execution_bucket_tokens;
  wire.maximum_prefill_chunk_tokens = limits.maximum_prefill_chunk_tokens;
  wire.pad_to_maximum_execution_bucket = limits.pad_to_maximum_execution_bucket;
  wire.profile_revision = limits.profile_revision.data();
  wire.profile_revision_bytes = static_cast<uint32_t>(limits.profile_revision.size());
  CopyDigest(wire.resource_vector_hash, limits.resource_vector_hash);
  wire.eos_token_id = limits.eos_token_id;
  wire.output_burst_credit_count = limits.output_burst_credit_count;
  wire.output_maximum_slots_per_credit = limits.output_maximum_slots_per_credit;
  wire.output_maximum_bytes_per_credit = limits.output_maximum_bytes_per_credit;
  wire.output_plan_kind = static_cast<uint32_t>(limits.output_plan_kind);
  pih_execution_controller_v1* handle = nullptr;
  const auto status = FromWire(api.create(api.context, &wire, &handle));
  if (!status.ok()) return status;
  if (!handle) return Status::Internal("execution provider returned no controller handle");
  return ControllerClient(api, handle, limits);
}

Status ControllerClient::close() {
  if (!handle_) return Status::Ok();
  return FromWire(api_.destroy(api_.context, &handle_));
}
Status ControllerClient::submit_admit(uint64_t generation,
    std::span<const uint32_t> prompt, uint32_t maximum_new_tokens,
    const ControllerRequestSampling& sampling) {
  if (!handle_ || prompt.empty() || prompt.size() > UINT32_MAX ||
      sampling.stop_token_count > ControllerRequestSampling::kMaximumStopTokenIds)
    return Status::InvalidArgument("execution admission arguments invalid");
  const auto wire = ToWire(sampling);
  return FromWire(api_.submit_admit(api_.context, handle_, generation,
      prompt.data(), static_cast<uint32_t>(prompt.size()), maximum_new_tokens,
      &wire));
}
Status ControllerClient::submit_cancel(uint64_t generation) {
  if (!handle_) return Status::FailedPrecondition("execution controller closed");
  return FromWire(api_.submit_cancel(api_.context, handle_, generation));
}
pih_status_v1 ControllerClient::Reserve(void* context,
    const pih_execution_admission_v1* wire) noexcept {
  try {
    auto& self = *static_cast<ControllerClient*>(context);
    if (!self.admission_ || !wire || wire->struct_size != sizeof(*wire) ||
        wire->abi_version != PIH_EXECUTION_CONTROLLER_ABI_V1 ||
        !wire->command_sequence || !wire->epoch || !wire->request_generation ||
        !wire->slot_generation || !wire->prompt_token_ids || !wire->prompt_token_count ||
        wire->slot_index >= self.limits_.ingress.requests.maximum_requests ||
        wire->prompt_token_count >
            self.limits_.ingress.requests.maximum_prompt_tokens_per_request)
      return ToWire(Status::Internal("execution admission callback view invalid"));
    auto sampling = FromWire(wire->sampling);
    if (!sampling.ok()) return ToWire(sampling.status());
    const auto digest = Digest(wire->payload_digest);
    const ControllerAdmissionView admission{
        {ControllerCommandKind::kAdmit, wire->command_sequence, wire->epoch,
         wire->request_generation, digest},
        {wire->slot_index, wire->slot_generation, wire->request_generation,
         std::span(wire->prompt_token_ids, wire->prompt_token_count),
         wire->maximum_new_tokens, *sampling, digest}};
    return ToWire(self.admission_->reserve(admission));
  } catch (...) {
    return ToWire(Status::Internal("execution admission reserve exception"));
  }
}
pih_status_v1 ControllerClient::Publish(void* context, uint64_t generation) noexcept {
  try {
    auto& self = *static_cast<ControllerClient*>(context);
    return self.admission_ ? ToWire(self.admission_->publish(generation)) :
        ToWire(Status::Internal("execution admission callback absent"));
  } catch (...) {
    return ToWire(Status::Internal("execution admission publish exception"));
  }
}
pih_status_v1 ControllerClient::Rollback(void* context, uint64_t generation) noexcept {
  try {
    auto& self = *static_cast<ControllerClient*>(context);
    return self.admission_ ? ToWire(self.admission_->rollback(generation)) :
        ToWire(Status::Internal("execution admission callback absent"));
  } catch (...) {
    return ToWire(Status::Internal("execution admission rollback exception"));
  }
}
Result<ControllerRuntimeStep> ControllerClient::step(
    int64_t now_ns, ControllerAdmissionParticipant& admission) {
  if (!handle_ || admission_) return Status::FailedPrecondition("execution step unavailable");
  admission_ = &admission;
  const pih_execution_admission_callbacks_v1 callbacks{
      sizeof(callbacks), PIH_EXECUTION_CONTROLLER_ABI_V1,
      this, Reserve, Publish, Rollback};
  uint32_t value = 0;
  const auto wire = api_.step(api_.context, handle_, now_ns, &callbacks, &value);
  admission_ = nullptr;
  const auto status = FromWire(wire);
  if (!status.ok()) return status;
  if (value < PIH_EXECUTION_STEP_IDLE_V1 ||
      value > PIH_EXECUTION_STEP_ADMISSION_REJECTED_V1)
    return Status::Internal("execution step value invalid");
  return static_cast<ControllerRuntimeStep>(value);
}

Result<std::optional<ControllerPreparedPlanView>>
ControllerClient::prepare_next_plan(int64_t now_ns) {
  if (!handle_) return Status::FailedPrecondition("execution controller closed");
  pih_execution_plan_view_v1 wire{};
  wire.struct_size = sizeof(wire);
  wire.abi_version = PIH_EXECUTION_CONTROLLER_ABI_V1;
  uint32_t has_plan = 0;
  const auto status = FromWire(api_.prepare(api_.context, handle_, now_ns,
                                            &has_plan, &wire));
  if (!status.ok()) return status;
  if (has_plan == 0) return std::optional<ControllerPreparedPlanView>{};
  if (has_plan != 1 || wire.struct_size != sizeof(wire) ||
      wire.abi_version != PIH_EXECUTION_CONTROLLER_ABI_V1 ||
      wire.epoch != limits_.epoch || !wire.plan_sequence ||
      wire.sequence_count == 0 ||
      wire.sequence_count > limits_.scheduler.maximum_sequences_per_plan ||
      wire.execution_bucket_tokens == 0 ||
      wire.execution_bucket_tokens > limits_.metadata.maximum_execution_bucket_tokens ||
      wire.sample_row_count > wire.sequence_count ||
      !wire.profile_revision || !wire.profile_revision_bytes ||
      wire.profile_revision_bytes > 128 ||
      !wire.sequence_generations || !wire.committed_start_positions ||
      !wire.sequence_real_token_counts || !wire.packed_offsets ||
      !wire.state_generations || !wire.input_digests ||
      !wire.selected_slot_indices || !wire.selected_sampling ||
      !wire.input_token_ids || !wire.positions || !wire.request_index ||
      !wire.query_start_offsets ||
      (wire.sample_row_count && !wire.sample_row_index))
    return Status::Internal("execution plan projection invalid");
  const std::string_view revision(wire.profile_revision, wire.profile_revision_bytes);
  if (revision != limits_.profile_revision ||
      Digest(wire.resource_vector_hash) != limits_.resource_vector_hash)
    return Status::Internal("execution plan identity drifted");
  plan_inputs_.clear();
  plan_inputs_.reserve(wire.sequence_count);
  for (uint32_t i = 0; i < wire.sequence_count; ++i) {
    plan_inputs_.push_back({wire.sequence_generations[i],
        wire.committed_start_positions[i], wire.sequence_real_token_counts[i],
        wire.state_generations[i], Digest(wire.input_digests + i * 32U)});
  }
  auto plan = PackedTokenPlan::Create(wire.epoch, wire.plan_sequence,
      static_cast<PackedTokenPhase>(wire.phase), std::string(revision),
      limits_.resource_vector_hash, plan_inputs_, wire.execution_bucket_tokens,
      {limits_.scheduler.maximum_sequences_per_plan,
       limits_.scheduler.maximum_real_tokens_per_plan,
       limits_.metadata.maximum_execution_bucket_tokens});
  if (!plan.ok() || plan->total_real_tokens() != wire.real_token_count)
    return Status::Internal("execution packed plan invalid");
  auto digest = plan->semantic_digest();
  if (!digest.ok() || *digest != Digest(wire.plan_digest))
    return Status::Internal("execution packed plan digest invalid");
  for (uint32_t i = 0; i <= wire.sequence_count; ++i)
    if (wire.packed_offsets[i] != plan->packed_offsets()[i])
      return Status::Internal("execution packed offsets drifted");
  input_tokens_.assign(wire.input_token_ids,
                       wire.input_token_ids + wire.execution_bucket_tokens);
  positions_.assign(wire.positions, wire.positions + wire.execution_bucket_tokens);
  request_index_.assign(wire.request_index,
                        wire.request_index + wire.execution_bucket_tokens);
  query_offsets_.assign(wire.query_start_offsets,
                        wire.query_start_offsets + wire.sequence_count + 1);
  for (uint32_t i = 0; i <= wire.sequence_count; ++i)
    if (query_offsets_[i] != plan->packed_offsets()[i])
      return Status::Internal("execution metadata query offsets drifted");
  for (uint32_t i = 0; i < wire.sequence_count; ++i) {
    const auto begin = plan->packed_offsets()[i];
    const auto count = plan->real_token_counts()[i];
    auto token_digest = packed_token_input_digest(
        std::span(input_tokens_).subspan(begin, count));
    if (!token_digest.ok() || *token_digest != plan->input_digests()[i])
      return Status::Internal("execution metadata token digest drifted");
    for (uint32_t local = 0; local < count; ++local) {
      const auto row = begin + local;
      if (plan->committed_start_positions()[i] > UINT64_MAX - local ||
          request_index_[row] != i ||
          positions_[row] != plan->committed_start_positions()[i] + local)
        return Status::Internal("execution metadata position drifted");
    }
  }
  sample_rows_.clear();
  if (wire.sample_row_count)
    sample_rows_.assign(wire.sample_row_index,
                        wire.sample_row_index + wire.sample_row_count);
  for (size_t i = 0; i < sample_rows_.size(); ++i)
    if (sample_rows_[i] >= wire.real_token_count ||
        (i && sample_rows_[i] <= sample_rows_[i - 1]))
      return Status::Internal("execution sampling rows invalid");
  selected_slots_.assign(wire.selected_slot_indices,
                         wire.selected_slot_indices + wire.sequence_count);
  selected_sampling_.clear();
  selected_sampling_.reserve(wire.sequence_count);
  for (uint32_t i = 0; i < wire.sequence_count; ++i) {
    auto sampling = FromWire(wire.selected_sampling[i]);
    if (!sampling.ok()) return sampling.status();
    selected_sampling_.push_back(*sampling);
  }
  plan_.emplace(std::move(*plan));
  PackedTokenMetadataView metadata{wire.metadata_generation, input_tokens_,
      positions_, request_index_, query_offsets_, sample_rows_,
      wire.real_token_count};
  return std::optional<ControllerPreparedPlanView>{ControllerPreparedPlanView{
      &*plan_, metadata, selected_slots_, selected_sampling_, *digest}};
}

Status ControllerClient::commit_current_plan(uint64_t sequence, Sha256Digest digest) {
  if (!handle_) return Status::FailedPrecondition("execution controller closed");
  return FromWire(api_.commit(api_.context, handle_, sequence,
      reinterpret_cast<const uint8_t*>(digest.bytes.data())));
}
Status ControllerClient::mark_current_plan_in_flight(uint64_t sequence, Sha256Digest digest) {
  if (!handle_) return Status::FailedPrecondition("execution controller closed");
  return FromWire(api_.mark_in_flight(api_.context, handle_, sequence,
      reinterpret_cast<const uint8_t*>(digest.bytes.data())));
}
Status ControllerClient::abort_current_plan(uint64_t sequence, Sha256Digest digest) {
  if (!handle_) return Status::FailedPrecondition("execution controller closed");
  return FromWire(api_.abort(api_.context, handle_, sequence,
      reinterpret_cast<const uint8_t*>(digest.bytes.data())));
}
Status ControllerClient::complete_current_sampled_tokens(
    std::span<const uint32_t> tokens, int64_t now_ns) {
  if (!handle_ || tokens.empty() || tokens.size() > UINT32_MAX)
    return Status::InvalidArgument("execution sampled token span invalid");
  return FromWire(api_.complete_sampled_tokens(api_.context, handle_, tokens.data(),
      static_cast<uint32_t>(tokens.size()), now_ns));
}
Status ControllerClient::acknowledge_output_plan(uint64_t sequence) {
  if (!handle_) return Status::FailedPrecondition("execution controller closed");
  return FromWire(api_.acknowledge_output(api_.context, handle_, sequence));
}
Status ControllerClient::finalize_draining(uint64_t generation) {
  if (!handle_) return Status::FailedPrecondition("execution controller closed");
  return FromWire(api_.finalize_draining(api_.context, handle_, generation));
}
Result<std::optional<ControllerOutputEvent>> ControllerClient::try_take_event() {
  if (!handle_) return Status::FailedPrecondition("execution controller closed");
  pih_execution_event_v1 wire{};
  wire.struct_size = sizeof(wire);
  wire.abi_version = PIH_EXECUTION_CONTROLLER_ABI_V1;
  uint32_t has_event = 0;
  const auto status = FromWire(api_.take_event(api_.context, handle_, &has_event, &wire));
  if (!status.ok()) return status;
  if (!has_event) return std::optional<ControllerOutputEvent>{};
  if (has_event != 1 || wire.struct_size != sizeof(wire) ||
      wire.abi_version != PIH_EXECUTION_CONTROLLER_ABI_V1 ||
      wire.epoch != limits_.epoch ||
      wire.kind < PIH_EXECUTION_EVENT_ADMITTED_V1 ||
      wire.kind > PIH_EXECUTION_EVENT_FAILED_V1 ||
      wire.finish_reason > PIH_EXECUTION_FINISH_LENGTH_V1 ||
      !wire.event_sequence || !wire.request_generation)
    return Status::Internal("execution event projection invalid");
  return std::optional<ControllerOutputEvent>{ControllerOutputEvent{
      static_cast<ControllerOutputEventKind>(wire.kind), wire.event_sequence,
      wire.epoch, wire.request_generation, wire.token_ordinal, wire.token_id,
      Digest(wire.state_digest), wire.plan_sequence, wire.plan_event_index,
      wire.plan_event_count, static_cast<ControllerFinishReason>(wire.finish_reason)}};
}
uint32_t ControllerClient::active_sequence_count() const noexcept {
  if (!handle_) return UINT32_MAX;
  uint32_t active = UINT32_MAX, queued = UINT32_MAX;
  const auto status = api_.counts(api_.context, handle_, &active, &queued);
  return pih_status_is_ok_v1(&status) ? active : UINT32_MAX;
}
uint32_t ControllerClient::queued_command_count() const noexcept {
  if (!handle_) return UINT32_MAX;
  uint32_t active = UINT32_MAX, queued = UINT32_MAX;
  const auto status = api_.counts(api_.context, handle_, &active, &queued);
  return pih_status_is_ok_v1(&status) ? queued : UINT32_MAX;
}

}  // namespace pih::qwen_plugin
