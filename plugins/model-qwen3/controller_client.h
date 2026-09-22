#pragma once

#include "pih/contracts/execution_controller_v1.h"
#include "pih/scheduler/controller_runtime.h"

#include <optional>
#include <vector>

namespace pih::qwen_plugin {

// Model-owned value adapter. The provider owns ControllerRuntime; this object
// owns only a handle and bounded copies of the current plan projection.
class ControllerClient final {
 public:
  static Result<ControllerClient> Create(
      const pih_execution_controller_api_v1& api,
      const ControllerRuntimeLimits& limits);
  static Result<ControllerClient> CreateBound(const ControllerRuntimeLimits& limits);
  static Status Bind(const pih_execution_controller_api_v1* api);
  static void Unbind() noexcept;
  ControllerClient(const ControllerClient&) = delete;
  ControllerClient& operator=(const ControllerClient&) = delete;
  ControllerClient(ControllerClient&& other) noexcept;
  ControllerClient& operator=(ControllerClient&&) = delete;
  ~ControllerClient();

  Status close();
  Status submit_admit(uint64_t generation, std::span<const uint32_t> prompt,
                      uint32_t maximum_new_tokens,
                      const ControllerRequestSampling& sampling);
  Status submit_cancel(uint64_t generation);
  Result<ControllerRuntimeStep> step(int64_t now_ns,
                                     ControllerAdmissionParticipant& admission);
  Result<std::optional<ControllerPreparedPlanView>> prepare_next_plan(int64_t now_ns);
  Status commit_current_plan(uint64_t sequence, Sha256Digest digest);
  Status mark_current_plan_in_flight(uint64_t sequence, Sha256Digest digest);
  Status abort_current_plan(uint64_t sequence, Sha256Digest digest);
  Status complete_current_sampled_tokens(std::span<const uint32_t> tokens,
                                         int64_t now_ns);
  Status acknowledge_output_plan(uint64_t sequence);
  Status finalize_draining(uint64_t generation);
  Result<std::optional<ControllerOutputEvent>> try_take_event();
  uint32_t active_sequence_count() const noexcept;
  uint32_t queued_command_count() const noexcept;
  uint32_t eos_token_id() const noexcept { return limits_.eos_token_id; }

 private:
  ControllerClient(const pih_execution_controller_api_v1& api,
                   pih_execution_controller_v1* handle,
                   ControllerRuntimeLimits limits) noexcept;
  static pih_status_v1 Reserve(void* context,
      const pih_execution_admission_v1* admission) noexcept;
  static pih_status_v1 Publish(void* context, uint64_t generation) noexcept;
  static pih_status_v1 Rollback(void* context, uint64_t generation) noexcept;

  pih_execution_controller_api_v1 api_{};
  pih_execution_controller_v1* handle_{};
  ControllerRuntimeLimits limits_{};
  ControllerAdmissionParticipant* admission_{};
  std::optional<PackedTokenPlan> plan_;
  std::vector<PackedSequenceInput> plan_inputs_;
  std::vector<uint32_t> input_tokens_, request_index_, query_offsets_;
  std::vector<uint32_t> sample_rows_, selected_slots_;
  std::vector<uint64_t> positions_;
  std::vector<ControllerRequestSampling> selected_sampling_;
};

}  // namespace pih::qwen_plugin
