#pragma once

#include "pih/core/allocator.h"

namespace pih {

class RegisteredPinnedAllocator : public Allocator {
 public:
  ~RegisteredPinnedAllocator() override = default;
};

}  // namespace pih
