#pragma once

#include <filesystem>

#include "pih/io/exclusive_file_publisher.h"
#include "pih/model/catalog_placement_recovery_codec.h"

namespace pih {

enum class CatalogPlacementRecoveryAction : std::uint8_t {
  kReleased = 1,
  kRequireReconciliation = 2,
  kQuarantined = 3,
};

Result<ExclusiveFilePublicationReceipt> publish_catalog_placement_recovery_snapshot(
    const std::filesystem::path& staging_path,
    const std::filesystem::path& published_path,
    const CatalogPlacementRecoverySnapshot& snapshot);
Result<CatalogPlacementRecoverySnapshot> load_catalog_placement_recovery_snapshot(
    const std::filesystem::path& published_path);
CatalogPlacementRecoveryAction classify_catalog_placement_recovery(
    const CatalogPlacementRecoverySnapshot& snapshot) noexcept;

}  // namespace pih
