// Offline plugin-owned conversion implementation.
#include "qwen3_int4_conversion_stream.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>

namespace pih {

Result<QwenInt4ConversionStream> QwenInt4ConversionStream::Create(
    std::vector<QwenInt4ArtifactRecordPlan> artifact_records,
    std::vector<QwenInt4LinearRecordPlan> linear_records,
    QwenInt4ConversionTensorSource& source) {
  std::sort(artifact_records.begin(), artifact_records.end(),
            [](const auto& left, const auto& right) {
              return left.identity < right.identity;
            });
  std::sort(linear_records.begin(), linear_records.end(),
            [](const auto& left, const auto& right) {
              return left.source_name < right.source_name;
            });
  std::set<std::string> target_names;
  for (std::size_t index = 0; index < artifact_records.size(); ++index) {
    const auto& record = artifact_records[index];
    if (record.identity.empty() ||
        (index != 0 && artifact_records[index - 1].identity == record.identity) ||
        !target_names.insert(record.identity).second) {
      return Status::InvalidArgument("Qwen INT4 conversion target is duplicated");
    }
  }
  std::set<std::string> linear_sources;
  for (const auto& linear : linear_records) {
    if (!linear_sources.insert(linear.source_name).second ||
        !target_names.contains(linear.values_name) ||
        !target_names.contains(linear.scales_name)) {
      return Status::InvalidArgument("Qwen INT4 conversion Linear inventory mismatch");
    }
  }
  for (const auto& record : artifact_records) {
    if (record.kind == QwenInt4ArtifactRecordKind::kPackedValues ||
        record.kind == QwenInt4ArtifactRecordKind::kFp16Scales) {
      const auto found = std::find_if(linear_records.begin(), linear_records.end(),
                                      [&record](const auto& linear) {
                                        return linear.source_name == record.source_name;
                                      });
      if (found == linear_records.end() ||
          (record.kind == QwenInt4ArtifactRecordKind::kPackedValues
               ? found->values_name
               : found->scales_name) != record.identity) {
        return Status::InvalidArgument("Qwen INT4 conversion target lacks Linear");
      }
    }
  }
  return QwenInt4ConversionStream(std::move(artifact_records),
                                  std::move(linear_records), &source);
}

const QwenInt4ArtifactRecordPlan* QwenInt4ConversionStream::artifact(
    std::string_view identity) const {
  const auto found = std::lower_bound(
      artifact_records_.begin(), artifact_records_.end(), identity,
      [](const auto& record, std::string_view key) {
        return record.identity < key;
      });
  return found == artifact_records_.end() || found->identity != identity
             ? nullptr
             : &*found;
}

const QwenInt4LinearRecordPlan* QwenInt4ConversionStream::linear_for(
    const QwenInt4ArtifactRecordPlan& record) const {
  const auto found = std::lower_bound(
      linear_records_.begin(), linear_records_.end(), record.source_name,
      [](const auto& linear, std::string_view key) {
        return linear.source_name < key;
      });
  return found == linear_records_.end() ||
                 found->source_name != record.source_name
             ? nullptr
             : &*found;
}

Result<std::span<const std::byte>> QwenInt4ConversionStream::materialize(
    const QwenInt4ArtifactRecordPlan& record) {
  if (record.kind == QwenInt4ArtifactRecordKind::kAlias) {
    return Status::InvalidArgument("Qwen INT4 alias has no payload");
  }
  const auto* linear = linear_for(record);
  if (linear == nullptr) {
    active_linear_payload_.reset();
    active_linear_source_.clear();
    auto source = source_->tensor(record.source_name);
    if (!source.ok()) return source.status();
    if (source->size() != record.logical_bytes) {
      return Status::FailedPrecondition("Qwen INT4 identity payload size mismatch");
    }
    return *source;
  }
  if (!active_linear_payload_.has_value() ||
      active_linear_source_ != linear->source_name) {
    auto source = source_->tensor(linear->source_name);
    if (!source.ok()) return source.status();
    auto payload = QwenInt4TensorPayload::Create(*source, *linear);
    if (!payload.ok()) return payload.status();
    peak_anonymous_workspace_bytes_ =
        std::max(peak_anonymous_workspace_bytes_,
                 payload->peak_anonymous_workspace_bytes());
    active_linear_payload_ = std::move(*payload);
    active_linear_source_ = linear->source_name;
  }
  return record.kind == QwenInt4ArtifactRecordKind::kPackedValues
             ? active_linear_payload_->packed_values()
             : active_linear_payload_->scale_bytes_le();
}

Result<QwenInt4DigestScanReceipt>
QwenInt4ConversionStream::scan_payload_digests() {
  reset();
  std::vector<QwenInt4PayloadDigest> digests;
  for (const auto& record : artifact_records_) {
    if (record.kind == QwenInt4ArtifactRecordKind::kAlias) continue;
    auto payload = materialize(record);
    if (!payload.ok()) {
      reset();
      return payload.status();
    }
    auto digest = sha256(*payload);
    if (!digest.ok()) {
      reset();
      return digest.status();
    }
    digests.push_back({record.identity, *digest});
  }
  const auto peak = peak_anonymous_workspace_bytes_;
  reset();
  return QwenInt4DigestScanReceipt{std::move(digests), peak};
}

Result<std::size_t> QwenInt4ConversionStream::read(
    std::string_view identity, std::uint64_t logical_offset,
    std::span<std::byte> output) {
  const auto* record = artifact(identity);
  if (record == nullptr || output.empty()) {
    return Status::InvalidArgument("Qwen INT4 conversion read is invalid");
  }
  auto payload = materialize(*record);
  if (!payload.ok()) return payload.status();
  if (logical_offset >= payload->size()) {
    return Status::InvalidArgument("Qwen INT4 conversion read is out of range");
  }
  const auto count = std::min<std::size_t>(
      output.size(), payload->size() - static_cast<std::size_t>(logical_offset));
  std::memcpy(output.data(), payload->data() + logical_offset, count);
  return count;
}

void QwenInt4ConversionStream::reset() noexcept {
  active_linear_payload_.reset();
  active_linear_source_.clear();
}

}  // namespace pih
