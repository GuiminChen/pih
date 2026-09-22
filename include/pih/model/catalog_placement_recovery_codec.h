#pragma once

#include <span>
#include <vector>

#include "pih/model/catalog_placement_transaction.h"

namespace pih {

Result<std::vector<std::byte>> encode_catalog_placement_recovery_snapshot(
    const CatalogPlacementRecoverySnapshot& value);
Result<CatalogPlacementRecoverySnapshot> parse_catalog_placement_recovery_snapshot(
    std::span<const std::byte> bytes);

}  // namespace pih
