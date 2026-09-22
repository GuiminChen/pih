// Offline plugin-owned conversion implementation.
#include "qwen3_int4_converter.h"
#include "qwen3_int4_artifact_writer.h"

#include <span>

#include "pih/io/durable_file_sink.h"
#include "pih/io/mapped_file.h"
#include "qwen3_int4_conversion_stream.h"
#include "pih/model/qwen3_int4_disposition_plan.h"
#include "pih/model/qwen3_int4_linear_shape_ledger.h"
#include "qwen3_int4_safetensors_source.h"

namespace pih {

Result<QwenInt4ConversionReceipt> convert_qwen3_official_to_int4(
    const SafetensorsFile& source,
    const Qwen3SourceArtifactReceipt& source_receipt,
    const QwenInt4SourceBinding& source_binding,
    const std::filesystem::path& staging_path,
    const std::filesystem::path& published_path,
    std::size_t writer_workspace_bytes) {
  if (writer_workspace_bytes == 0 ||
      writer_workspace_bytes > kMaximumCanonicalWriterWorkspaceBytes ||
      staging_path.empty() || published_path.empty() ||
      staging_path == published_path ||
      staging_path.parent_path() != published_path.parent_path()) {
    return Status::InvalidArgument("Qwen INT4 conversion publication bounds invalid");
  }
  Status source_status =
      revalidate_qwen3_source_artifact_receipt(source, source_receipt);
  if (!source_status.ok()) return source_status;
  auto layout = QwenInt4ArtifactLayout::CreateOfficialPureW4();
  if (!layout.ok()) return layout.status();
  auto ledger = QwenInt4LinearShapeLedger::CreateOfficial();
  if (!ledger.ok()) return ledger.status();
  auto disposition = QwenInt4DispositionPlan::CreateOfficialPureW4();
  if (!disposition.ok()) return disposition.status();
  auto binding_digest = source_binding.semantic_digest();
  if (!binding_digest.ok()) return binding_digest.status();
  auto disposition_digest = disposition->semantic_digest();
  if (!disposition_digest.ok()) return disposition_digest.status();
  const QwenInt4ArtifactRoots roots{source_receipt.file_sha256,
                                     *binding_digest, *disposition_digest};

  auto tensor_source = QwenInt4SafetensorsSource::Create(source, source_binding);
  if (!tensor_source.ok()) return tensor_source.status();
  auto stream = QwenInt4ConversionStream::Create(
      layout->records(), ledger->records(), *tensor_source);
  if (!stream.ok()) return stream.status();
  auto scan = stream->scan_payload_digests();
  if (!scan.ok()) return scan.status();
  auto metadata = QwenInt4ArtifactMetadata::Create(
      *layout, roots, std::move(scan->payload_digests));
  if (!metadata.ok()) return metadata.status();

  CanonicalExtentWriteReceipt write_receipt{};
  {
    auto sink = DurableFileSink::CreateExclusive(
        staging_path, QwenInt4ArtifactLayout::kOfficialFileBytes);
    if (!sink.ok()) return sink.status();
    stream->reset();
    auto written = write_qwen_int4_artifact(
        *metadata, *stream, *sink, writer_workspace_bytes);
    if (!written.ok()) return written.status();
    write_receipt = *written;
  }

  QwenInt4ArtifactVerificationReceipt verification{};
  {
    auto staged = MappedFile::OpenReadOnly(
        staging_path, QwenInt4ArtifactLayout::kOfficialFileBytes);
    if (!staged.ok()) return staged.status();
    auto verified = verify_qwen_int4_artifact(std::span<const std::byte>(
        staged->data(), static_cast<std::size_t>(staged->size_bytes())));
    if (!verified.ok()) return verified.status();
    verification = *verified;
  }
  if (verification.file.file_sha256 != write_receipt.file_sha256 ||
      verification.file.file_bytes != write_receipt.file_bytes ||
      verification.roots.source_artifact != roots.source_artifact ||
      verification.roots.source_binding != roots.source_binding ||
      verification.roots.disposition != roots.disposition) {
    return Status::FailedPrecondition(
        "Qwen INT4 independent conversion evidence mismatch");
  }
  auto publication = publish_verified_file_exclusive(
      staging_path, published_path, write_receipt.file_bytes,
      write_receipt.file_sha256);
  if (!publication.ok()) return publication.status();
  return QwenInt4ConversionReceipt{
      *publication, roots, source_binding.records().size(),
      layout->records().size(), layout->payload_record_count(),
      scan->peak_anonymous_workspace_bytes, writer_workspace_bytes};
}

}  // namespace pih
