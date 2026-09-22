#include "pih/model/catalog_placement_recovery_journal.h"

#include "pih/io/durable_file_sink.h"
#include "pih/io/mapped_file.h"

namespace pih {

Result<ExclusiveFilePublicationReceipt> publish_catalog_placement_recovery_snapshot(
    const std::filesystem::path& staging_path,
    const std::filesystem::path& published_path,
    const CatalogPlacementRecoverySnapshot& snapshot) {
  auto encoded = encode_catalog_placement_recovery_snapshot(snapshot);
  if (!encoded.ok()) return encoded.status();
  auto digest = sha256(*encoded);
  if (!digest.ok()) return digest.status();
  {
    auto sink = DurableFileSink::CreateExclusive(staging_path, encoded->size());
    if (!sink.ok()) return sink.status();
    std::size_t offset = 0;
    while (offset < encoded->size()) {
      auto written = sink->write(std::span<const std::byte>(encoded->data() + offset,
                                                             encoded->size() - offset));
      if (!written.ok()) return written.status();
      offset += *written;
    }
    auto status = sink->sync();
    if (!status.ok()) return status;
  }
  return publish_verified_file_exclusive(staging_path, published_path,
                                        encoded->size(), *digest);
}

Result<CatalogPlacementRecoverySnapshot> load_catalog_placement_recovery_snapshot(
    const std::filesystem::path& published_path) {
  if (published_path.empty()) {
    return Status::InvalidArgument("placement recovery path is empty");
  }
  std::error_code error;
  const auto bytes = std::filesystem::file_size(published_path, error);
  if (error || bytes == 0 || bytes > 4096) {
    return Status::FailedPrecondition("placement recovery file is invalid");
  }
  auto mapped = MappedFile::OpenReadOnly(published_path, bytes);
  if (!mapped.ok()) return mapped.status();
  return parse_catalog_placement_recovery_snapshot(
      std::span<const std::byte>(mapped->data(),
                                 static_cast<std::size_t>(mapped->size_bytes())));
}

CatalogPlacementRecoveryAction classify_catalog_placement_recovery(
    const CatalogPlacementRecoverySnapshot& snapshot) noexcept {
  switch (snapshot.state) {
    case CatalogPlacementState::kWithdrawn:
      return CatalogPlacementRecoveryAction::kReleased;
    case CatalogPlacementState::kQuarantined:
      return CatalogPlacementRecoveryAction::kQuarantined;
    default:
      return CatalogPlacementRecoveryAction::kRequireReconciliation;
  }
}

}  // namespace pih
