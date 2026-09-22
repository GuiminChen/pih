#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/status.h"

#ifdef __cplusplus
extern "C" {
#endif

// This is a wire-only controller boundary. No C++ scheduler object, STL type,
// model KV owner, or provider allocation may be interpreted by a consumer.
#define PIH_EXECUTION_CONTROLLER_ABI_V1 1U
#define PIH_EXECUTION_CONTROLLER_MAX_STOP_TOKENS_V1 16U
#define PIH_EXECUTION_CONTROLLER_DIGEST_BYTES_V1 32U

enum pih_execution_sampling_mode_v1 {
  PIH_EXECUTION_SAMPLING_GREEDY_V1 = 0,
  PIH_EXECUTION_SAMPLING_STOCHASTIC_V1 = 1,
};
enum pih_execution_step_v1 {
  PIH_EXECUTION_STEP_IDLE_V1 = 1,
  PIH_EXECUTION_STEP_OUTPUT_BACKPRESSURED_V1 = 2,
  PIH_EXECUTION_STEP_ADMITTED_V1 = 3,
  PIH_EXECUTION_STEP_CANCEL_REQUESTED_V1 = 4,
  PIH_EXECUTION_STEP_ADMISSION_REJECTED_V1 = 5,
};
enum pih_execution_plan_phase_v1 {
  PIH_EXECUTION_PLAN_PREFILL_V1 = 1,
  PIH_EXECUTION_PLAN_DECODE_V1 = 2,
  PIH_EXECUTION_PLAN_VERIFY_V1 = 3,
};
enum pih_execution_event_kind_v1 {
  PIH_EXECUTION_EVENT_ADMITTED_V1 = 1,
  PIH_EXECUTION_EVENT_TOKEN_COMMITTED_V1 = 2,
  PIH_EXECUTION_EVENT_DRAINING_V1 = 3,
  PIH_EXECUTION_EVENT_COMPLETED_V1 = 4,
  PIH_EXECUTION_EVENT_CANCELLED_V1 = 5,
  PIH_EXECUTION_EVENT_FAILED_V1 = 6,
};
enum pih_execution_finish_reason_v1 {
  PIH_EXECUTION_FINISH_NONE_V1 = 0,
  PIH_EXECUTION_FINISH_STOP_V1 = 1,
  PIH_EXECUTION_FINISH_LENGTH_V1 = 2,
};
enum pih_execution_output_plan_kind_v1 {
  PIH_EXECUTION_OUTPUT_QWEN_V1 = 1,
  PIH_EXECUTION_OUTPUT_DEEPSEEK_V1 = 2,
};

typedef struct pih_execution_controller_v1 pih_execution_controller_v1;

typedef struct pih_execution_controller_limits_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t epoch;
  uint32_t maximum_sequences;
  uint32_t command_capacity;
  uint32_t event_capacity;
  uint32_t maximum_requests;
  uint32_t maximum_prompt_tokens_per_request;
  uint32_t maximum_context_tokens;
  uint32_t maximum_candidates;
  uint32_t maximum_sequences_per_plan;
  uint32_t maximum_real_tokens_per_plan;
  uint32_t scheduler_scan_limit;
  uint32_t max_consecutive_decode_rounds;
  int64_t prefill_starvation_threshold_ns;
  uint32_t metadata_maximum_sequences;
  uint32_t metadata_maximum_execution_bucket_tokens;
  uint32_t maximum_prefill_chunk_tokens;
  uint32_t pad_to_maximum_execution_bucket;
  const char* profile_revision;
  uint32_t profile_revision_bytes;
  uint8_t resource_vector_hash[PIH_EXECUTION_CONTROLLER_DIGEST_BYTES_V1];
  uint32_t eos_token_id;
  uint32_t output_burst_credit_count;
  uint32_t output_maximum_slots_per_credit;
  uint64_t output_maximum_bytes_per_credit;
  uint32_t output_plan_kind;
} pih_execution_controller_limits_v1;

typedef struct pih_execution_sampling_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t mode;
  float temperature;
  float top_p;
  uint32_t has_top_k;
  uint32_t top_k;
  uint64_t effective_seed;
  uint64_t sample_ordinal;
  uint32_t minimum_new_tokens;
  uint32_t logprobs_enabled;
  uint32_t top_logprobs_count;
  uint32_t stop_token_count;
  uint32_t stop_token_ids[PIH_EXECUTION_CONTROLLER_MAX_STOP_TOKENS_V1];
} pih_execution_sampling_v1;

typedef struct pih_execution_admission_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t command_sequence;
  uint64_t epoch;
  uint64_t request_generation;
  uint32_t slot_index;
  uint64_t slot_generation;
  uint8_t payload_digest[PIH_EXECUTION_CONTROLLER_DIGEST_BYTES_V1];
  const uint32_t* prompt_token_ids;
  uint32_t prompt_token_count;
  uint32_t maximum_new_tokens;
  pih_execution_sampling_v1 sampling;
} pih_execution_admission_v1;

// Callback pointers are borrowed only for the duration of step(). Callbacks
// must not throw, retain pointers, or reenter this controller handle.
typedef struct pih_execution_admission_callbacks_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  void* context;
  pih_status_v1 (*reserve)(void*, const pih_execution_admission_v1*);
  pih_status_v1 (*publish)(void*, uint64_t sequence_generation);
  pih_status_v1 (*rollback)(void*, uint64_t request_generation);
} pih_execution_admission_callbacks_v1;

typedef struct pih_execution_event_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t kind;
  uint32_t finish_reason;
  uint64_t event_sequence;
  uint64_t epoch;
  uint64_t request_generation;
  uint64_t token_ordinal;
  uint32_t token_id;
  uint8_t state_digest[PIH_EXECUTION_CONTROLLER_DIGEST_BYTES_V1];
  uint64_t plan_sequence;
  uint32_t plan_event_index;
  uint32_t plan_event_count;
} pih_execution_event_v1;

// Every pointer belongs to the provider and remains valid until the next
// mutating operation on this handle. The consumer must copy any retained data.
// Arrays are bounded by the limits supplied at creation, not by caller trust.
typedef struct pih_execution_plan_view_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t epoch;
  uint64_t plan_sequence;
  uint32_t phase;
  uint32_t sequence_count;
  uint32_t real_token_count;
  uint32_t execution_bucket_tokens;
  const char* profile_revision;
  uint32_t profile_revision_bytes;
  uint8_t resource_vector_hash[PIH_EXECUTION_CONTROLLER_DIGEST_BYTES_V1];
  uint8_t plan_digest[PIH_EXECUTION_CONTROLLER_DIGEST_BYTES_V1];
  const uint64_t* sequence_generations;
  const uint64_t* committed_start_positions;
  const uint32_t* sequence_real_token_counts;
  const uint32_t* packed_offsets;
  const uint64_t* state_generations;
  const uint8_t* input_digests;
  const uint32_t* selected_slot_indices;
  const pih_execution_sampling_v1* selected_sampling;
  uint64_t metadata_generation;
  const uint32_t* input_token_ids;
  const uint64_t* positions;
  const uint32_t* request_index;
  const uint32_t* query_start_offsets;
  uint32_t sample_row_count;
  const uint32_t* sample_row_index;
} pih_execution_plan_view_v1;

typedef struct pih_execution_controller_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  pih_status_v1 (*create)(void*, const pih_execution_controller_limits_v1*,
                          pih_execution_controller_v1**);
  pih_status_v1 (*destroy)(void*, pih_execution_controller_v1**);
  pih_status_v1 (*submit_admit)(void*, pih_execution_controller_v1*, uint64_t,
      const uint32_t*, uint32_t, uint32_t, const pih_execution_sampling_v1*);
  pih_status_v1 (*submit_cancel)(void*, pih_execution_controller_v1*, uint64_t);
  pih_status_v1 (*step)(void*, pih_execution_controller_v1*, int64_t,
      const pih_execution_admission_callbacks_v1*, uint32_t*);
  pih_status_v1 (*prepare)(void*, pih_execution_controller_v1*, int64_t,
      uint32_t*, pih_execution_plan_view_v1*);
  pih_status_v1 (*commit)(void*, pih_execution_controller_v1*, uint64_t,
      const uint8_t[PIH_EXECUTION_CONTROLLER_DIGEST_BYTES_V1]);
  pih_status_v1 (*mark_in_flight)(void*, pih_execution_controller_v1*, uint64_t,
      const uint8_t[PIH_EXECUTION_CONTROLLER_DIGEST_BYTES_V1]);
  pih_status_v1 (*abort)(void*, pih_execution_controller_v1*, uint64_t,
      const uint8_t[PIH_EXECUTION_CONTROLLER_DIGEST_BYTES_V1]);
  pih_status_v1 (*complete_sampled_tokens)(void*, pih_execution_controller_v1*,
      const uint32_t*, uint32_t, int64_t);
  pih_status_v1 (*acknowledge_output)(void*, pih_execution_controller_v1*, uint64_t);
  pih_status_v1 (*finalize_draining)(void*, pih_execution_controller_v1*, uint64_t);
  pih_status_v1 (*take_event)(void*, pih_execution_controller_v1*, uint32_t*,
      pih_execution_event_v1*);
  pih_status_v1 (*counts)(void*, pih_execution_controller_v1*, uint32_t*, uint32_t*);
} pih_execution_controller_api_v1;

#ifdef __cplusplus
}
#endif
