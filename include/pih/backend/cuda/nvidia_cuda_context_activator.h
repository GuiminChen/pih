#pragma once

#include "pih/model/deepseek_context_bound_nccl_api.h"

namespace pih {

class NvidiaCudaContextActivator final
    : public DeepSeekNcclContextActivator {
 public:
  Status activate(std::uintptr_t context_identity) override;
};

}  // namespace pih
