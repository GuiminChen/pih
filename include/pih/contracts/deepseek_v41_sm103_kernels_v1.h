#pragma once
#include <stdint.h>
#include "pih/plugin_sdk/kernel_pack.h"
#include "pih/plugin_sdk/status.h"

#ifdef __cplusplus
extern "C" {
#endif
#define PIH_V41_KERNEL_ABI_V1 1U
#define PIH_V41_KERNEL_WORDS_V1 256U
// Explicit field-order encoding specified by kernel_wire.h. Each scalar is one
// uint64 word; floats use their uint32 IEEE bits; regions are address,bytes.
// No native struct padding, C++ object, host pointer or ownership crosses ABI.
// Unused words must be zero. Launches enqueue on caller-owned streams; callers
// retain allocations, provider and pack until completion and inspect error flags.
enum pih_v41_kernel_operation_v1 {
  PIH_V41_FP8LINEAR_V1 = 1,
  PIH_V41_FP4LINEAR_V1 = 2,
  PIH_V41_ENGRAMLOOKUP_V1 = 3,
  PIH_V41_ENGRAMGATE_V1 = 4,
  PIH_V41_ENGRAMPROJECTION_V1 = 5,
  PIH_V41_ENGRAMSINGLERANK_V1 = 6,
  PIH_V41_RMSNORM_V1 = 7,
  PIH_V41_MHCMIX_V1 = 8,
  PIH_V41_MHCPRE_V1 = 9,
  PIH_V41_MHCPOST_V1 = 10,
  PIH_V41_ROPETABLE_V1 = 11,
  PIH_V41_ROPEAPPLY_V1 = 12,
  PIH_V41_ROPESEQUENCE_V1 = 13,
  PIH_V41_EXPERTDISPATCH_V1 = 14,
  PIH_V41_EXPERTGATHER_V1 = 15,
  PIH_V41_EXPERTSCATTER_V1 = 16,
  PIH_V41_EXPERTACTIVATION_V1 = 17,
  PIH_V41_EXPERTMERGE_V1 = 18,
  PIH_V41_INDEXERQUANTIZE_V1 = 19,
  PIH_V41_INDEXERKEYPROJECTION_V1 = 20,
  PIH_V41_INDEXERKEYCACHE_V1 = 21,
  PIH_V41_INDEXERWEIGHTS_V1 = 22,
  PIH_V41_INDEXERSCORE_V1 = 23,
  PIH_V41_INDEXERSELECT_V1 = 24,
  PIH_V41_INDEXERCANDIDATES_V1 = 25,
  PIH_V41_GROUPEDOUTPUT_V1 = 26,
  PIH_V41_SPARSEATTENTION_V1 = 27,
  PIH_V41_ATTENTIONASSEMBLY_V1 = 28,
  PIH_V41_COMPRESSORPOOL_V1 = 29,
  PIH_V41_COMPRESSORPROJECTION_V1 = 30,
  PIH_V41_COMPRESSEDKV_V1 = 31,
  PIH_V41_WINDOWKV_V1 = 32,
  PIH_V41_HEADPROJECTION_V1 = 33,
  PIH_V41_TOKENEMBEDDINGLOOKUP_V1 = 34,
  PIH_V41_TOKENEMBEDDINGEXPAND_V1 = 35,
  PIH_V41_ROUTER_V1 = 36,
  PIH_V41_SAMPLING_V1 = 37,
  PIH_V41_PROMOTEATTENTIONOUTPUT_V1 = 38,
  PIH_V41_ROUNDATTENTIONOUTPUT_V1 = 39,
  PIH_V41_MHCINITIALPRE_V1 = 40,
};
typedef struct pih_v41_kernel_command_v1 {
  uint32_t struct_size, abi_version, operation, word_count;
  uint64_t words[PIH_V41_KERNEL_WORDS_V1];
} pih_v41_kernel_command_v1;
typedef struct pih_v41_sm103_kernels_api_v1 {
  uint32_t struct_size, contract_version;
  const pih_kernel_pack_identity_v1* identity;
  uint32_t target_sm, operation_count;
  // Called after the backend prepares the rank-local device, before allocation
  // or launch. The initial profile admits exact NVIDIA B300 SM103 devices.
  pih_status_v1 (*admit_device)(int32_t device_ordinal);
  pih_status_v1 (*launch)(const pih_v41_kernel_command_v1*);
} pih_v41_sm103_kernels_api_v1;
#ifdef __cplusplus
}
#endif
