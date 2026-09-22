#pragma once

#include "pih/model/profile_authority.h"

namespace pih {

class OpenSslProfileEd25519Verifier final : public ProfileEd25519Verifier {
 public:
  Status verify(std::string_view key_id,
                std::span<const std::byte> public_key,
                std::span<const std::byte> message,
                std::span<const std::byte> signature) override;
};

}  // namespace pih
