#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PIH_TRANSPORT_COLLECTIVE_ABI_V1 1U
#define PIH_TRANSPORT_NCCL_ID_BYTES_V1 128U

enum pih_transport_transition_v1 {
  PIH_TRANSPORT_PENDING_V1 = 0,
  PIH_TRANSPORT_READY_V1 = 1,
};
enum pih_transport_datatype_v1 {
  PIH_TRANSPORT_BF16_V1 = 1,
  PIH_TRANSPORT_FP32_V1 = 2,
};

typedef struct pih_transport_communicator_v1 pih_transport_communicator_v1;

// The provider owns every communicator and verifies the current CUDA device,
// rank identity and device-buffer extent before a collective is enqueued.
// A handle is valid until release/abort consumes it; it is never a raw NCCL
// pointer and may not be dereferenced by the model.
// start() can return a failure with a non-null handle after NCCL has allocated
// state; the caller must abort that handle before attempting plugin retirement.
// A collective READY transition proves NCCL enqueue progress only, not CUDA
// stream completion. The caller must prove stream/event completion before
// reusing buffers, destroying streams, or releasing its CUDA context. Poll
// collective health while work is outstanding even after enqueue is READY.
// Finalize/release or abort must complete before the CUDA provider is retired.
// Once provider drain begins, start rejects new communicators; existing
// handles remain available for health polling and finalize/release or abort.
typedef struct pih_transport_collective_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  pih_status_v1 (*start)(void*, const uint8_t id[PIH_TRANSPORT_NCCL_ID_BYTES_V1],
      uint32_t world, uint32_t rank, int32_t device,
      pih_transport_communicator_v1** output);
  pih_status_v1 (*poll)(void*, pih_transport_communicator_v1*, uint32_t* transition);
  pih_status_v1 (*validate_buffer)(void*, pih_transport_communicator_v1*,
      uintptr_t address, uint64_t bytes, uint32_t datatype,
      uint32_t world, uint32_t rank);
  pih_status_v1 (*all_reduce_sum)(void*, pih_transport_communicator_v1*,
      uintptr_t address, uint64_t bytes, uint32_t datatype,
      uintptr_t stream, uint32_t* transition);
  pih_status_v1 (*all_gather)(void*, pih_transport_communicator_v1*,
      uintptr_t input, uint64_t input_bytes, uintptr_t output,
      uint64_t output_bytes, uint32_t datatype, uintptr_t stream,
      uint32_t* transition);
  pih_status_v1 (*poll_collective)(void*, pih_transport_communicator_v1*,
      uint32_t* transition);
  pih_status_v1 (*begin_finalize)(void*, pih_transport_communicator_v1*);
  pih_status_v1 (*release)(void*, pih_transport_communicator_v1**);
  pih_status_v1 (*abort)(void*, pih_transport_communicator_v1**);
} pih_transport_collective_api_v1;

#ifdef __cplusplus
}
#endif
