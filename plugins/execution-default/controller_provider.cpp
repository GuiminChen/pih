#include "controller_provider.h"

#include "pih/scheduler/controller_runtime.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <string>
#include <vector>

static_assert(sizeof(pih::Sha256Digest) == PIH_EXECUTION_CONTROLLER_DIGEST_BYTES_V1);

struct pih_execution_controller_v1 final {
  explicit pih_execution_controller_v1(pih::ControllerRuntime&& value,
                                       uint32_t maximum_sequences)
      : runtime(std::move(value)), sampling(maximum_sequences) {}
  pih::ControllerRuntime runtime;
  std::vector<pih_execution_sampling_v1> sampling;
};

namespace pih::execution_plugin {
namespace {

struct Provider final {
  std::mutex mutex;
  bool ready{};
  std::vector<std::unique_ptr<pih_execution_controller_v1>> handles;
};
Provider provider;

pih_status_v1 WireStatus(uint32_t code, std::string_view message = {}) noexcept {
  pih_status_v1 result{};
  result.struct_size = sizeof(result);
  result.abi_version = PIH_STATUS_ABI_VERSION_V1;
  result.code = code;
  std::memcpy(result.message, message.data(),
              std::min(message.size(), sizeof(result.message) - 1));
  return result;
}
pih_status_v1 WireStatus(const Status& status) noexcept {
  uint32_t code = PIH_STATUS_INTERNAL_V1;
  switch (status.code()) {
    case StatusCode::kOk: code = PIH_STATUS_OK_V1; break;
    case StatusCode::kInvalidArgument: code = PIH_STATUS_INVALID_ARGUMENT_V1; break;
    case StatusCode::kFailedPrecondition: code = PIH_STATUS_FAILED_PRECONDITION_V1; break;
    case StatusCode::kInternal: code = PIH_STATUS_INTERNAL_V1; break;
    case StatusCode::kUnavailable: code = PIH_STATUS_UNAVAILABLE_V1; break;
    case StatusCode::kResourceExhausted: code = PIH_STATUS_RESOURCE_EXHAUSTED_V1; break;
    case StatusCode::kDeadlineExceeded: code = PIH_STATUS_DEADLINE_EXCEEDED_V1; break;
  }
  return WireStatus(code, status.message());
}
Status CoreStatus(const pih_status_v1& wire) {
  if (!pih_status_is_valid_v1(&wire)) return Status::Internal("admission callback status invalid");
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
  return Status::Internal("admission callback code invalid");
}
template<class Operation>
pih_status_v1 Guard(Operation operation) noexcept {
  try { return operation(); }
  catch (const std::bad_alloc&) {
    return WireStatus(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "controller allocation failed");
  } catch (...) {
    return WireStatus(PIH_STATUS_INTERNAL_V1, "controller provider operation failed");
  }
}
pih_execution_controller_v1* Find(pih_execution_controller_v1* handle) noexcept {
  for (auto& owned : provider.handles)
    if (owned.get() == handle) return owned.get();
  return nullptr;
}
Sha256Digest Digest(const uint8_t* bytes) noexcept {
  Sha256Digest result{};
  std::memcpy(result.bytes.data(), bytes, result.bytes.size());
  return result;
}
void CopyDigest(uint8_t* output, const Sha256Digest& digest) noexcept {
  std::memcpy(output, digest.bytes.data(), digest.bytes.size());
}
bool ValidLimits(const pih_execution_controller_limits_v1* limits) noexcept {
  return limits && limits->struct_size == sizeof(*limits) &&
      limits->abi_version == PIH_EXECUTION_CONTROLLER_ABI_V1 && limits->epoch &&
      limits->maximum_sequences && limits->maximum_sequences <= 256 &&
      limits->command_capacity && limits->command_capacity <= 4096 &&
      limits->event_capacity && limits->event_capacity <= 4096 &&
      limits->maximum_requests && limits->maximum_requests <= 256 &&
      limits->maximum_prompt_tokens_per_request &&
      limits->maximum_prompt_tokens_per_request <= 65536 &&
      limits->maximum_context_tokens && limits->maximum_context_tokens <= 65536 &&
      limits->maximum_candidates && limits->maximum_candidates <= 256 &&
      limits->maximum_sequences_per_plan &&
      limits->maximum_sequences_per_plan <= limits->maximum_sequences &&
      limits->maximum_real_tokens_per_plan &&
      limits->maximum_real_tokens_per_plan <= 65536 &&
      limits->scheduler_scan_limit && limits->scheduler_scan_limit <= 4096 &&
      limits->max_consecutive_decode_rounds &&
      limits->prefill_starvation_threshold_ns >= 0 &&
      limits->metadata_maximum_sequences &&
      limits->metadata_maximum_sequences <= limits->maximum_sequences &&
      limits->metadata_maximum_execution_bucket_tokens &&
      limits->metadata_maximum_execution_bucket_tokens <= 65536 &&
      limits->maximum_prefill_chunk_tokens &&
      limits->maximum_prefill_chunk_tokens <= 65536 &&
      limits->pad_to_maximum_execution_bucket <= 1 &&
      limits->profile_revision && limits->profile_revision_bytes &&
      limits->profile_revision_bytes <= 128 &&
      limits->output_burst_credit_count && limits->output_burst_credit_count <= 4096 &&
      limits->output_maximum_slots_per_credit &&
      limits->output_maximum_slots_per_credit <= 256 &&
      limits->output_maximum_bytes_per_credit &&
      (limits->output_plan_kind == 1 || limits->output_plan_kind == 2);
}
bool ValidSampling(const pih_execution_sampling_v1* sampling) noexcept {
  return sampling && sampling->struct_size == sizeof(*sampling) &&
      sampling->abi_version == PIH_EXECUTION_CONTROLLER_ABI_V1 &&
      sampling->mode <= 1 && sampling->has_top_k <= 1 &&
      sampling->logprobs_enabled <= 1 &&
      sampling->stop_token_count <= PIH_EXECUTION_CONTROLLER_MAX_STOP_TOKENS_V1;
}
ControllerRequestSampling CoreSampling(const pih_execution_sampling_v1& input) {
  ControllerRequestSampling result{};
  result.mode = static_cast<ControllerSamplingMode>(input.mode);
  result.temperature = input.temperature;
  result.top_p = input.top_p;
  if (input.has_top_k) result.top_k = input.top_k;
  result.effective_seed = input.effective_seed;
  result.sample_ordinal = input.sample_ordinal;
  result.minimum_new_tokens = input.minimum_new_tokens;
  result.logprobs_enabled = input.logprobs_enabled != 0;
  result.top_logprobs_count = input.top_logprobs_count;
  result.stop_token_count = input.stop_token_count;
  std::copy_n(input.stop_token_ids, result.stop_token_count,
              result.stop_token_ids.begin());
  return result;
}
pih_execution_sampling_v1 WireSampling(const ControllerRequestSampling& input) noexcept {
  pih_execution_sampling_v1 result{};
  result.struct_size = sizeof(result);
  result.abi_version = PIH_EXECUTION_CONTROLLER_ABI_V1;
  result.mode = static_cast<uint32_t>(input.mode);
  result.temperature = input.temperature;
  result.top_p = input.top_p;
  result.has_top_k = input.top_k.has_value();
  result.top_k = input.top_k.value_or(0);
  result.effective_seed = input.effective_seed;
  result.sample_ordinal = input.sample_ordinal;
  result.minimum_new_tokens = input.minimum_new_tokens;
  result.logprobs_enabled = input.logprobs_enabled;
  result.top_logprobs_count = input.top_logprobs_count;
  result.stop_token_count = input.stop_token_count;
  std::copy_n(input.stop_token_ids.begin(), input.stop_token_count,
              result.stop_token_ids);
  return result;
}
class Admission final : public ControllerAdmissionParticipant {
 public:
  explicit Admission(const pih_execution_admission_callbacks_v1& callbacks)
      : callbacks_(callbacks) {}
  Status reserve(const ControllerAdmissionView& view) override {
    pih_execution_admission_v1 request{};
    request.struct_size = sizeof(request);
    request.abi_version = PIH_EXECUTION_CONTROLLER_ABI_V1;
    request.command_sequence = view.command.command_sequence;
    request.epoch = view.command.epoch;
    request.request_generation = view.request.request_generation;
    request.slot_index = view.request.slot_index;
    request.slot_generation = view.request.slot_generation;
    CopyDigest(request.payload_digest, view.request.payload_digest);
    request.prompt_token_ids = view.request.prompt_token_ids.data();
    request.prompt_token_count = static_cast<uint32_t>(view.request.prompt_token_ids.size());
    request.maximum_new_tokens = view.request.maximum_new_tokens;
    request.sampling = WireSampling(view.request.sampling);
    return CoreStatus(callbacks_.reserve(callbacks_.context, &request));
  }
  Status publish(uint64_t generation) override {
    return CoreStatus(callbacks_.publish(callbacks_.context, generation));
  }
  Status rollback(uint64_t generation) override {
    return CoreStatus(callbacks_.rollback(callbacks_.context, generation));
  }
 private:
  const pih_execution_admission_callbacks_v1& callbacks_;
};
bool ValidCallbacks(const pih_execution_admission_callbacks_v1* callbacks) noexcept {
  return callbacks && callbacks->struct_size == sizeof(*callbacks) &&
      callbacks->abi_version == PIH_EXECUTION_CONTROLLER_ABI_V1 &&
      callbacks->context && callbacks->reserve && callbacks->publish &&
      callbacks->rollback;
}

pih_status_v1 Create(void* context, const pih_execution_controller_limits_v1* limits,
                     pih_execution_controller_v1** output) noexcept {
  if (output) *output = nullptr;
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !ValidLimits(limits) || !output)
      return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller limits invalid");
    std::lock_guard lock(provider.mutex);
    if (!provider.ready) return WireStatus(PIH_STATUS_FAILED_PRECONDITION_V1, "provider not ready");
    if (provider.handles.size() >= 4)
      return WireStatus(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "controller handle limit");
    ControllerRuntimeLimits core{};
    core.epoch = limits->epoch;
    core.maximum_sequences = limits->maximum_sequences;
    core.ingress.mailbox = {limits->command_capacity, limits->event_capacity};
    core.ingress.requests = {limits->maximum_requests,
        limits->maximum_prompt_tokens_per_request, limits->maximum_context_tokens};
    core.scheduler = {limits->maximum_candidates,
        limits->maximum_sequences_per_plan, limits->maximum_real_tokens_per_plan,
        limits->scheduler_scan_limit, limits->max_consecutive_decode_rounds,
        limits->prefill_starvation_threshold_ns};
    core.metadata = {limits->metadata_maximum_sequences,
        limits->metadata_maximum_execution_bucket_tokens};
    core.maximum_prefill_chunk_tokens = limits->maximum_prefill_chunk_tokens;
    core.pad_to_maximum_execution_bucket =
        limits->pad_to_maximum_execution_bucket != 0;
    core.profile_revision.assign(limits->profile_revision,
        limits->profile_revision_bytes);
    core.resource_vector_hash = Digest(limits->resource_vector_hash);
    core.eos_token_id = limits->eos_token_id;
    core.output_burst_credit_count = limits->output_burst_credit_count;
    core.output_maximum_slots_per_credit = limits->output_maximum_slots_per_credit;
    core.output_maximum_bytes_per_credit = limits->output_maximum_bytes_per_credit;
    core.output_plan_kind = static_cast<OutputPlanKind>(limits->output_plan_kind);
    auto runtime = ControllerRuntime::Create(std::move(core));
    if (!runtime.ok()) return WireStatus(runtime.status());
    auto handle = std::make_unique<pih_execution_controller_v1>(
        std::move(*runtime), limits->maximum_sequences);
    *output = handle.get();
    provider.handles.emplace_back(std::move(handle));
    return WireStatus(PIH_STATUS_OK_V1);
  });
}
pih_status_v1 Destroy(void* context, pih_execution_controller_v1** handle) noexcept {
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !handle || !*handle)
      return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller handle invalid");
    std::lock_guard lock(provider.mutex);
    auto item = std::find_if(provider.handles.begin(), provider.handles.end(),
        [&](const auto& owned) { return owned.get() == *handle; });
    if (item == provider.handles.end())
      return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller handle foreign");
    if (!(*item)->runtime.retirable())
      return WireStatus(PIH_STATUS_FAILED_PRECONDITION_V1, "controller has unretired work");
    provider.handles.erase(item);
    *handle = nullptr;
    return WireStatus(PIH_STATUS_OK_V1);
  });
}
pih_status_v1 SubmitAdmit(void* context, pih_execution_controller_v1* handle,
    uint64_t generation, const uint32_t* tokens, uint32_t count,
    uint32_t maximum_new_tokens, const pih_execution_sampling_v1* sampling) noexcept {
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !generation || !tokens || !count || count > 65536 ||
        !ValidSampling(sampling))
      return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller admission invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner) return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller handle foreign");
    auto submitted = owner->runtime.submit_admit(generation,
        std::span(tokens, count), maximum_new_tokens, CoreSampling(*sampling));
    return submitted.ok() ? WireStatus(PIH_STATUS_OK_V1) : WireStatus(submitted.status());
  });
}
pih_status_v1 SubmitCancel(void* context, pih_execution_controller_v1* handle,
                          uint64_t generation) noexcept {
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !generation)
      return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "cancel generation invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner) return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller handle foreign");
    auto submitted = owner->runtime.submit_cancel(generation);
    return submitted.ok() ? WireStatus(PIH_STATUS_OK_V1) : WireStatus(submitted.status());
  });
}
pih_status_v1 Step(void* context, pih_execution_controller_v1* handle,
    int64_t now_ns, const pih_execution_admission_callbacks_v1* callbacks,
    uint32_t* step) noexcept {
  if (step) *step = 0;
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !step || (callbacks && !ValidCallbacks(callbacks)))
      return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller step invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner) return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller handle foreign");
    if (callbacks) {
      Admission admission(*callbacks);
      auto result = owner->runtime.step(now_ns, admission);
      if (!result.ok()) return WireStatus(result.status());
      *step = static_cast<uint32_t>(*result);
      return WireStatus(PIH_STATUS_OK_V1);
    }
    auto result = owner->runtime.step(now_ns);
    if (!result.ok()) return WireStatus(result.status());
    *step = static_cast<uint32_t>(*result);
    return WireStatus(PIH_STATUS_OK_V1);
  });
}

pih_status_v1 Prepare(void* context, pih_execution_controller_v1* handle,
    int64_t now_ns, uint32_t* has_plan, pih_execution_plan_view_v1* output) noexcept {
  if (has_plan) *has_plan = 0;
  if (output && output->struct_size == sizeof(*output) &&
      output->abi_version == PIH_EXECUTION_CONTROLLER_ABI_V1) {
    *output = {};
    output->struct_size = sizeof(*output);
    output->abi_version = PIH_EXECUTION_CONTROLLER_ABI_V1;
  }
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !has_plan || !output ||
        output->struct_size != sizeof(*output) ||
        output->abi_version != PIH_EXECUTION_CONTROLLER_ABI_V1)
      return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller plan output invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner) return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller handle foreign");
    auto prepared = owner->runtime.prepare_next_plan(now_ns);
    if (!prepared.ok()) return WireStatus(prepared.status());
    if (!prepared->has_value()) return WireStatus(PIH_STATUS_OK_V1);
    const auto& view = **prepared;
    const auto& plan = *view.plan;
    if (plan.sequence_count() > owner->sampling.size() ||
        view.selected_sampling.size() != plan.sequence_count() ||
        view.selected_slot_indices.size() != plan.sequence_count() ||
        view.metadata.query_start_offsets.size() != plan.sequence_count() + 1 ||
        view.metadata.input_token_ids.size() != plan.execution_bucket_tokens() ||
        view.metadata.positions.size() != plan.execution_bucket_tokens() ||
        view.metadata.request_index.size() != plan.execution_bucket_tokens() ||
        view.metadata.sample_row_index.size() > plan.sequence_count())
      return WireStatus(PIH_STATUS_INTERNAL_V1, "controller plan projection invalid");
    for (size_t i = 0; i < plan.sequence_count(); ++i)
      owner->sampling[i] = WireSampling(view.selected_sampling[i]);
    output->epoch = plan.epoch();
    output->plan_sequence = plan.plan_sequence();
    output->phase = static_cast<uint32_t>(plan.phase());
    output->sequence_count = static_cast<uint32_t>(plan.sequence_count());
    output->real_token_count = plan.total_real_tokens();
    output->execution_bucket_tokens = plan.execution_bucket_tokens();
    output->profile_revision = plan.profile_revision().data();
    output->profile_revision_bytes = static_cast<uint32_t>(plan.profile_revision().size());
    CopyDigest(output->resource_vector_hash, plan.resource_vector_hash());
    CopyDigest(output->plan_digest, view.plan_digest);
    output->sequence_generations = plan.ordered_sequence_generations().data();
    output->committed_start_positions = plan.committed_start_positions().data();
    output->sequence_real_token_counts = plan.real_token_counts().data();
    output->packed_offsets = plan.packed_offsets().data();
    output->state_generations = plan.state_generations().data();
    output->input_digests = reinterpret_cast<const uint8_t*>(plan.input_digests().data());
    output->selected_slot_indices = view.selected_slot_indices.data();
    output->selected_sampling = owner->sampling.data();
    output->metadata_generation = view.metadata.generation;
    output->input_token_ids = view.metadata.input_token_ids.data();
    output->positions = view.metadata.positions.data();
    output->request_index = view.metadata.request_index.data();
    output->query_start_offsets = view.metadata.query_start_offsets.data();
    output->sample_row_count = static_cast<uint32_t>(view.metadata.sample_row_index.size());
    output->sample_row_index = view.metadata.sample_row_index.data();
    *has_plan = 1;
    return WireStatus(PIH_STATUS_OK_V1);
  });
}
template<class Operation>
pih_status_v1 PlanIdentity(void* context, pih_execution_controller_v1* handle,
    uint64_t sequence, const uint8_t* digest, Operation operation) noexcept {
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !sequence || !digest)
      return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "plan identity invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner) return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller handle foreign");
    return WireStatus(operation(owner->runtime, sequence, Digest(digest)));
  });
}
pih_status_v1 Commit(void* context, pih_execution_controller_v1* handle,
    uint64_t sequence, const uint8_t* digest) noexcept {
  return PlanIdentity(context, handle, sequence, digest,
      [](auto& runtime, auto number, auto hash) {
        return runtime.commit_current_plan(number, hash);
      });
}
pih_status_v1 MarkInFlight(void* context, pih_execution_controller_v1* handle,
    uint64_t sequence, const uint8_t* digest) noexcept {
  return PlanIdentity(context, handle, sequence, digest,
      [](auto& runtime, auto number, auto hash) {
        return runtime.mark_current_plan_in_flight(number, hash);
      });
}
pih_status_v1 Abort(void* context, pih_execution_controller_v1* handle,
    uint64_t sequence, const uint8_t* digest) noexcept {
  return PlanIdentity(context, handle, sequence, digest,
      [](auto& runtime, auto number, auto hash) {
        return runtime.abort_current_plan(number, hash);
      });
}
pih_status_v1 CompleteSampledTokens(void* context,
    pih_execution_controller_v1* handle, const uint32_t* tokens, uint32_t count,
    int64_t now_ns) noexcept {
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !tokens || !count || count > 256)
      return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "sampled tokens invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner) return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller handle foreign");
    return WireStatus(owner->runtime.complete_current_sampled_tokens(
        std::span(tokens, count), now_ns));
  });
}
pih_status_v1 AcknowledgeOutput(void* context,
    pih_execution_controller_v1* handle, uint64_t sequence) noexcept {
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !sequence)
      return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "output plan invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner) return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller handle foreign");
    return WireStatus(owner->runtime.acknowledge_output_plan(sequence));
  });
}
pih_status_v1 FinalizeDraining(void* context,
    pih_execution_controller_v1* handle, uint64_t generation) noexcept {
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !generation)
      return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "drain generation invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner) return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller handle foreign");
    return WireStatus(owner->runtime.finalize_draining(generation));
  });
}
pih_status_v1 TakeEvent(void* context, pih_execution_controller_v1* handle,
    uint32_t* has_event, pih_execution_event_v1* output) noexcept {
  if (has_event) *has_event = 0;
  if (output && output->struct_size == sizeof(*output) &&
      output->abi_version == PIH_EXECUTION_CONTROLLER_ABI_V1) {
    *output = {};
    output->struct_size = sizeof(*output);
    output->abi_version = PIH_EXECUTION_CONTROLLER_ABI_V1;
  }
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !has_event || !output ||
        output->struct_size != sizeof(*output) ||
        output->abi_version != PIH_EXECUTION_CONTROLLER_ABI_V1)
      return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "event output invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner) return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller handle foreign");
    auto event = owner->runtime.try_take_event();
    if (!event.ok()) return WireStatus(event.status());
    if (!event->has_value()) return WireStatus(PIH_STATUS_OK_V1);
    const auto& value = **event;
    output->kind = static_cast<uint32_t>(value.kind);
    output->finish_reason = static_cast<uint32_t>(value.finish_reason);
    output->event_sequence = value.event_sequence;
    output->epoch = value.epoch;
    output->request_generation = value.request_generation;
    output->token_ordinal = value.token_ordinal;
    output->token_id = value.token_id;
    CopyDigest(output->state_digest, value.state_digest);
    output->plan_sequence = value.plan_sequence;
    output->plan_event_index = value.plan_event_index;
    output->plan_event_count = value.plan_event_count;
    *has_event = 1;
    return WireStatus(PIH_STATUS_OK_V1);
  });
}
pih_status_v1 Counts(void* context, pih_execution_controller_v1* handle,
    uint32_t* active, uint32_t* queued) noexcept {
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !active || !queued)
      return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller counts output invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner) return WireStatus(PIH_STATUS_INVALID_ARGUMENT_V1, "controller handle foreign");
    *active = owner->runtime.active_sequence_count();
    *queued = owner->runtime.queued_command_count();
    return WireStatus(PIH_STATUS_OK_V1);
  });
}

}  // namespace

const pih_execution_controller_api_v1* ControllerApi() noexcept {
  static const pih_execution_controller_api_v1 api{
      sizeof(api), PIH_EXECUTION_CONTROLLER_ABI_V1, &provider,
      Create, Destroy, SubmitAdmit, SubmitCancel, Step, Prepare,
      Commit, MarkInFlight, Abort, CompleteSampledTokens,
      AcknowledgeOutput, FinalizeDraining, TakeEvent, Counts};
  return &api;
}
void SetControllerReady(bool ready) noexcept {
  std::lock_guard lock(provider.mutex);
  provider.ready = ready;
}
bool ControllerHandlesEmpty() noexcept {
  std::lock_guard lock(provider.mutex);
  return provider.handles.empty();
}

}  // namespace pih::execution_plugin
