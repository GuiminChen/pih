#pragma once
#include "pih/contracts/deepseek_v41_sm103_kernels_v1.h"
#include "pih/core/status.h"
namespace pih::deepseek_v41 {
// One exclusive rank thread. Binding must outlive all submissions and be
// removed only after WorkerRuntime reaches kReleased, before plugin unload.
Status BindSm103Kernels(const pih_v41_sm103_kernels_api_v1& api);
void UnbindSm103Kernels() noexcept;
}

