/* Compile-only: every installed public SDK header must be usable from C11.
 * No function is executed and no inference implementation is linked. */
#include "pih/plugin_sdk/abi.h"
#include "pih/plugin_sdk/capability.h"
#include "pih/plugin_sdk/handles.h"
#include "pih/plugin_sdk/kernel_pack.h"
#include "pih/plugin_sdk/lifecycle.h"
#include "pih/plugin_sdk/status.h"
#include "pih/contracts/engine_v1.h"
#include "pih/contracts/token_generation_v1.h"
#include "pih/contracts/text_inference_v2.h"
#include "pih/contracts/qwen_kernels_v1.h"
#include "pih/contracts/deepseek_kernels_v1.h"
#include "pih/contracts/deepseek_v41_sm103_kernels_v1.h"
#include "pih/contracts/execution_default_v1.h"
#include "pih/contracts/execution_controller_v1.h"
#include "pih/contracts/transport_collective_v1.h"
#include "pih/contracts/health_v1.h"
#include "pih/contracts/memory_host_spill_v1.h"
#include "pih/contracts/nvidia_cuda_v1.h"
#include "pih/contracts/nvidia_cuda_memory_v1.h"
#include "pih/contracts/nvidia_cuda_resources_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"
#include "pih/contracts/openai_http_v1.h"
#include "pih/contracts/platform_linux_v1.h"
#include "pih/contracts/verified_artifact_v1.h"
#include "pih/contracts/artifact_snapshot_v1.h"

_Static_assert(PIH_TEXT_INFERENCE_ABI_V2 == 2U, "unexpected text ABI");
_Static_assert(sizeof(((pih_text_inference_api_v2*)0)->struct_size) == 4,
               "ABI struct_size must remain uint32_t");

int pih_sdk_c_header_probe(void) {
    pih_text_inference_api_v2 model = {0};
    pih_text_output_sink_v2 sink = {0};
    pih_text_service_api_v2 service = {0};
    return (int)(model.struct_size + sink.struct_size + service.struct_size);
}
