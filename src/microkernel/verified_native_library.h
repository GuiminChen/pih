#pragma once

#include <string>
#include <string_view>

namespace pih::microkernel {

void* OpenVerifiedNativeLibrary(const std::string& absolute_path,
                                std::string_view expected_sha256_hex,
                                const char* load_failure);

}  // namespace pih::microkernel
