#include "pih/model/deepseek_rank_artifact_descriptor_batch.h"

#include <limits>
#include <type_traits>
#include <utility>

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

constexpr std::uint32_t kMagic = 0x50524958U;
constexpr std::uint16_t kFrameType = 7;
constexpr std::uint16_t kFrameVersion = 1;
constexpr std::size_t kFixedBytesBeforeRecords =
    8 + (2 * 8) + (8 * 4) + (5 * 8) + (5 * 32);
constexpr std::size_t kTrailingRootBytes = 32;
constexpr std::size_t kRecordFixedBytes =
    4 + 2 + (4 * 8) + 4 + 4 + 32;

Result<Sha256Digest> compile_frame_root(
    const DeepSeekRankArtifactDescriptorBatchFields& fields,
    std::span<const DeepSeekRankArtifactDescriptorExpectation>
        expectations) {
  DeepSeekRankArtifactTransferAckFields validation{
      fields.engine_epoch,
      fields.worker_generation,
      fields.world_size,
      fields.rank,
      fields.batch_index,
      fields.batch_count,
      fields.first_descriptor_ordinal,
      fields.descriptor_count,
      fields.cumulative_descriptor_count,
      fields.final_batch,
      fields.process_manifest_identity,
      fields.process_identity,
      fields.pidfd_identity,
      fields.control_identity,
      fields.challenge_identity,
      fields.transfer_manifest_root,
      fields.artifact_admission_binding_root,
      fields.transfer_transaction_root,
      fields.descriptor_batch_root,
      fields.adopted_descriptor_set_root};
  auto valid = DeepSeekRankArtifactTransferAck::Create(validation);
  if (!valid.ok()) return valid.status();
  if (expectations.size() != fields.descriptor_count) {
    return Status::InvalidArgument(
        "DeepSeek artifact descriptor batch metadata count differs");
  }
  auto descriptor_root =
      compile_deepseek_rank_artifact_transfer_descriptor_batch_root(
          fields.rank, fields.batch_index, expectations);
  if (!descriptor_root.ok()) return descriptor_root.status();
  if (*descriptor_root != fields.descriptor_batch_root) {
    return Status::FailedPrecondition(
        "DeepSeek artifact descriptor batch metadata root differs");
  }
  if (fields.first_descriptor_ordinal == 0) {
    auto adopted_root =
        compile_deepseek_rank_artifact_transfer_adopted_descriptor_set_root(
            fields.rank, expectations);
    if (!adopted_root.ok()) return adopted_root.status();
    if (*adopted_root != fields.adopted_descriptor_set_root) {
      return Status::FailedPrecondition(
          "DeepSeek initial adopted descriptor set root differs");
    }
  }

  std::size_t encoded_bytes = kFixedBytesBeforeRecords + kTrailingRootBytes;
  for (const auto& expectation : expectations) {
    if (expectation.shard_name.size() > 255U ||
        encoded_bytes >
            kDeepSeekRankArtifactDescriptorBatchFrameMaximumBytes -
                kRecordFixedBytes - expectation.shard_name.size()) {
      return Status::ResourceExhausted(
          "DeepSeek artifact descriptor batch frame exceeds budget");
    }
    encoded_bytes += kRecordFixedBytes + expectation.shard_name.size();
  }

  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-descriptor-batch-frame:v1", 20);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u64(1, fields.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, fields.worker_generation);
  if (status.ok()) status = builder->add_u32(3, fields.world_size);
  if (status.ok()) status = builder->add_u32(4, fields.rank);
  if (status.ok()) status = builder->add_u32(5, fields.batch_index);
  if (status.ok()) status = builder->add_u32(6, fields.batch_count);
  if (status.ok()) {
    status = builder->add_u32(7, fields.first_descriptor_ordinal);
  }
  if (status.ok()) status = builder->add_u32(8, fields.descriptor_count);
  if (status.ok()) {
    status = builder->add_u32(9, fields.cumulative_descriptor_count);
  }
  if (status.ok()) status = builder->add_u32(10, fields.final_batch ? 1U : 0U);
  if (status.ok()) {
    status = builder->add_u64(11, fields.process_manifest_identity);
  }
  if (status.ok()) status = builder->add_u64(12, fields.process_identity);
  if (status.ok()) status = builder->add_u64(13, fields.pidfd_identity);
  if (status.ok()) status = builder->add_u64(14, fields.control_identity);
  if (status.ok()) status = builder->add_u64(15, fields.challenge_identity);
  if (status.ok()) {
    status = builder->add_hash(16, fields.transfer_manifest_root);
  }
  if (status.ok()) {
    status = builder->add_hash(17, fields.artifact_admission_binding_root);
  }
  if (status.ok()) {
    status = builder->add_hash(18, fields.transfer_transaction_root);
  }
  if (status.ok()) status = builder->add_hash(19, fields.descriptor_batch_root);
  if (status.ok()) {
    status = builder->add_hash(20, fields.adopted_descriptor_set_root);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

template <class T>
void put(std::vector<std::byte>& output, T value) {
  using U = std::make_unsigned_t<T>;
  const auto bits = static_cast<U>(value);
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    output.push_back(
        static_cast<std::byte>((bits >> (index * 8U)) & 0xffU));
  }
}

void put_digest(std::vector<std::byte>& output,
                const Sha256Digest& value) {
  output.insert(output.end(), value.bytes.begin(), value.bytes.end());
}

template <class T>
bool get(std::span<const std::byte> input, std::size_t& cursor, T* value) {
  if (value == nullptr || cursor > input.size() ||
      input.size() - cursor < sizeof(T)) {
    return false;
  }
  using U = std::make_unsigned_t<T>;
  U bits = 0;
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    bits |= static_cast<U>(std::to_integer<unsigned>(input[cursor++]))
            << (index * 8U);
  }
  *value = static_cast<T>(bits);
  return true;
}

bool get_digest(std::span<const std::byte> input, std::size_t& cursor,
                Sha256Digest* value) {
  if (value == nullptr || cursor > input.size() ||
      input.size() - cursor < value->bytes.size()) {
    return false;
  }
  for (auto& byte : value->bytes) byte = input[cursor++];
  return true;
}

}  // namespace

DeepSeekRankArtifactDescriptorBatch::DeepSeekRankArtifactDescriptorBatch(
    DeepSeekRankArtifactDescriptorBatchFields fields,
    std::vector<DeepSeekRankArtifactDescriptorExpectation> expectations,
    Sha256Digest frame_root) noexcept
    : fields_(std::move(fields)), expectations_(std::move(expectations)),
      frame_root_(frame_root) {}

Result<DeepSeekRankArtifactDescriptorBatch>
DeepSeekRankArtifactDescriptorBatch::Create(
    DeepSeekRankArtifactDescriptorBatchFields fields,
    std::vector<DeepSeekRankArtifactDescriptorExpectation> expectations) {
  auto root = compile_frame_root(fields, expectations);
  if (!root.ok()) return root.status();
  return DeepSeekRankArtifactDescriptorBatch(
      std::move(fields), std::move(expectations), *root);
}

Result<DeepSeekRankArtifactDescriptorBatch>
DeepSeekRankArtifactDescriptorBatch::FromView(
    const DeepSeekRankArtifactDescriptorBatchView& view) {
  DeepSeekRankArtifactDescriptorBatchFields fields{
      view.engine_epoch,
      view.worker_generation,
      view.world_size,
      view.rank,
      view.batch_index,
      view.batch_count,
      view.first_descriptor_ordinal,
      static_cast<std::uint32_t>(view.descriptors.size()),
      view.cumulative_descriptor_count,
      view.final_batch,
      view.process_manifest_identity,
      view.process_identity,
      view.pidfd_identity,
      view.control_identity,
      view.challenge_identity,
      view.transfer_manifest_root,
      view.artifact_admission_binding_root,
      view.transfer_transaction_root,
      view.descriptor_batch_root,
      view.adopted_descriptor_set_root};
  return Create(
      std::move(fields),
      std::vector<DeepSeekRankArtifactDescriptorExpectation>(
          view.expectations.begin(), view.expectations.end()));
}

Result<std::vector<std::byte>> encode_deepseek_rank_artifact_descriptor_batch(
    const DeepSeekRankArtifactDescriptorBatch& batch) {
  auto verified = DeepSeekRankArtifactDescriptorBatch::Create(
      batch.fields(),
      std::vector<DeepSeekRankArtifactDescriptorExpectation>(
          batch.expectations().begin(), batch.expectations().end()));
  if (!verified.ok()) return verified.status();
  if (verified->frame_root() != batch.frame_root()) {
    return Status::FailedPrecondition(
        "DeepSeek artifact descriptor batch object root differs");
  }

  std::vector<std::byte> output;
  output.reserve(kDeepSeekRankArtifactDescriptorBatchFrameMaximumBytes);
  put(output, kMagic);
  put(output, kFrameType);
  put(output, kFrameVersion);
  const auto& fields = batch.fields();
  put(output, fields.engine_epoch);
  put(output, fields.worker_generation);
  put(output, fields.world_size);
  put(output, fields.rank);
  put(output, fields.batch_index);
  put(output, fields.batch_count);
  put(output, fields.first_descriptor_ordinal);
  put(output, fields.descriptor_count);
  put(output, fields.cumulative_descriptor_count);
  put(output, static_cast<std::uint32_t>(fields.final_batch ? 1U : 0U));
  put(output, fields.process_manifest_identity);
  put(output, fields.process_identity);
  put(output, fields.pidfd_identity);
  put(output, fields.control_identity);
  put(output, fields.challenge_identity);
  put_digest(output, fields.transfer_manifest_root);
  put_digest(output, fields.artifact_admission_binding_root);
  put_digest(output, fields.transfer_transaction_root);
  put_digest(output, fields.descriptor_batch_root);
  put_digest(output, fields.adopted_descriptor_set_root);
  for (const auto& expectation : batch.expectations()) {
    put(output, expectation.ordinal);
    put(output, static_cast<std::uint16_t>(expectation.shard_name.size()));
    for (const auto value : expectation.shard_name) {
      output.push_back(static_cast<std::byte>(
          static_cast<unsigned char>(value)));
    }
    put(output, expectation.identity.file_bytes);
    put(output, expectation.identity.filesystem_identity);
    put(output, expectation.identity.file_identity);
    put(output, expectation.identity.data_mtime_seconds);
    put(output, expectation.identity.data_mtime_nanoseconds);
    put(output, static_cast<std::uint32_t>(expectation.immutability_mode));
    put_digest(output, expectation.enforced_digest);
  }
  put_digest(output, batch.frame_root());
  if (output.size() >
      kDeepSeekRankArtifactDescriptorBatchFrameMaximumBytes) {
    return Status::Internal(
        "DeepSeek artifact descriptor batch encoder exceeded budget");
  }
  return output;
}

Result<DeepSeekRankArtifactDescriptorBatch>
decode_deepseek_rank_artifact_descriptor_batch(
    std::span<const std::byte> frame) {
  if (frame.size() < kFixedBytesBeforeRecords + kRecordFixedBytes +
                         kTrailingRootBytes ||
      frame.size() >
          kDeepSeekRankArtifactDescriptorBatchFrameMaximumBytes) {
    return Status::InvalidArgument(
        "DeepSeek artifact descriptor batch frame size is invalid");
  }
  std::size_t cursor = 0;
  std::uint32_t magic = 0;
  std::uint16_t type = 0;
  std::uint16_t version = 0;
  DeepSeekRankArtifactDescriptorBatchFields fields;
  std::uint32_t final_batch = 0;
  if (!get(frame, cursor, &magic) || !get(frame, cursor, &type) ||
      !get(frame, cursor, &version) || magic != kMagic ||
      type != kFrameType || version != kFrameVersion ||
      !get(frame, cursor, &fields.engine_epoch) ||
      !get(frame, cursor, &fields.worker_generation) ||
      !get(frame, cursor, &fields.world_size) ||
      !get(frame, cursor, &fields.rank) ||
      !get(frame, cursor, &fields.batch_index) ||
      !get(frame, cursor, &fields.batch_count) ||
      !get(frame, cursor, &fields.first_descriptor_ordinal) ||
      !get(frame, cursor, &fields.descriptor_count) ||
      !get(frame, cursor, &fields.cumulative_descriptor_count) ||
      !get(frame, cursor, &final_batch) || final_batch > 1U ||
      !get(frame, cursor, &fields.process_manifest_identity) ||
      !get(frame, cursor, &fields.process_identity) ||
      !get(frame, cursor, &fields.pidfd_identity) ||
      !get(frame, cursor, &fields.control_identity) ||
      !get(frame, cursor, &fields.challenge_identity) ||
      !get_digest(frame, cursor, &fields.transfer_manifest_root) ||
      !get_digest(frame, cursor,
                  &fields.artifact_admission_binding_root) ||
      !get_digest(frame, cursor, &fields.transfer_transaction_root) ||
      !get_digest(frame, cursor, &fields.descriptor_batch_root) ||
      !get_digest(frame, cursor, &fields.adopted_descriptor_set_root)) {
    return Status::InvalidArgument(
        "DeepSeek artifact descriptor batch frame header is invalid");
  }
  fields.final_batch = final_batch == 1U;
  if (fields.descriptor_count == 0 ||
      fields.descriptor_count >
          kDeepSeekRankArtifactTransferDescriptorBatchMaximum) {
    return Status::InvalidArgument(
        "DeepSeek artifact descriptor batch count is invalid");
  }

  std::vector<DeepSeekRankArtifactDescriptorExpectation> expectations;
  expectations.reserve(fields.descriptor_count);
  for (std::uint32_t index = 0; index < fields.descriptor_count; ++index) {
    DeepSeekRankArtifactDescriptorExpectation expectation;
    std::uint16_t name_bytes = 0;
    std::uint32_t mode = 0;
    if (!get(frame, cursor, &expectation.ordinal) ||
        !get(frame, cursor, &name_bytes) || name_bytes == 0 ||
        cursor > frame.size() ||
        frame.size() - cursor <
            static_cast<std::size_t>(name_bytes) +
                (kRecordFixedBytes - 6U) + kTrailingRootBytes) {
      return Status::InvalidArgument(
          "DeepSeek artifact descriptor batch record is truncated");
    }
    expectation.shard_name.reserve(name_bytes);
    for (std::uint16_t byte = 0; byte < name_bytes; ++byte) {
      expectation.shard_name.push_back(
          static_cast<char>(std::to_integer<unsigned>(frame[cursor++])));
    }
    if (!get(frame, cursor, &expectation.identity.file_bytes) ||
        !get(frame, cursor, &expectation.identity.filesystem_identity) ||
        !get(frame, cursor, &expectation.identity.file_identity) ||
        !get(frame, cursor, &expectation.identity.data_mtime_seconds) ||
        !get(frame, cursor,
             &expectation.identity.data_mtime_nanoseconds) ||
        !get(frame, cursor, &mode) ||
        mode > static_cast<std::uint32_t>(
                   ArtifactImmutabilityMode::kDmVeritySnapshot) ||
        !get_digest(frame, cursor, &expectation.enforced_digest)) {
      return Status::InvalidArgument(
          "DeepSeek artifact descriptor batch record is invalid");
    }
    expectation.immutability_mode =
        static_cast<ArtifactImmutabilityMode>(mode);
    expectations.push_back(std::move(expectation));
  }
  Sha256Digest expected_root{};
  if (!get_digest(frame, cursor, &expected_root) || cursor != frame.size()) {
    return Status::InvalidArgument(
        "DeepSeek artifact descriptor batch frame length differs");
  }
  auto result = DeepSeekRankArtifactDescriptorBatch::Create(
      std::move(fields), std::move(expectations));
  if (!result.ok()) return result.status();
  if (result->frame_root() != expected_root) {
    return Status::FailedPrecondition(
        "DeepSeek artifact descriptor batch frame root differs");
  }
  return result;
}

static_assert(kFixedBytesBeforeRecords == 256);
static_assert(kRecordFixedBytes == 78);
static_assert(kDeepSeekRankArtifactDescriptorBatchFrameMaximumBytes ==
              kFixedBytesBeforeRecords +
                  (kDeepSeekRankArtifactTransferDescriptorBatchMaximum *
                   (kRecordFixedBytes + 255)) +
                  kTrailingRootBytes);

}  // namespace pih
