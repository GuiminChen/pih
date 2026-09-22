#include "pih/model/deepseek_rank_artifact_transfer_ack.h"

#include <type_traits>
#include <utility>

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

constexpr std::uint32_t kMagic = 0x50524958U;
constexpr std::uint16_t kFrameType = 6;
constexpr std::uint16_t kFrameVersion = 1;

bool nonzero(const Sha256Digest& value) noexcept {
  return value != Sha256Digest{};
}

Result<Sha256Digest> compile_root(
    const DeepSeekRankArtifactTransferAckFields& fields) {
  const auto expected_first =
      fields.batch_index *
      kDeepSeekRankArtifactTransferDescriptorBatchMaximum;
  const auto expected_cumulative =
      fields.first_descriptor_ordinal + fields.descriptor_count;
  const auto expected_final = fields.batch_index + 1U == fields.batch_count;
  if (fields.engine_epoch == 0 || fields.worker_generation == 0 ||
      fields.world_size < 1 || fields.world_size > 4 ||
      fields.rank >= fields.world_size || fields.batch_count == 0 ||
      fields.batch_count >
          (kDeepSeekRankArtifactTransferDescriptorMaximum +
           kDeepSeekRankArtifactTransferDescriptorBatchMaximum - 1U) /
              kDeepSeekRankArtifactTransferDescriptorBatchMaximum ||
      fields.batch_index >= fields.batch_count ||
      fields.first_descriptor_ordinal != expected_first ||
      fields.descriptor_count == 0 ||
      fields.descriptor_count >
          kDeepSeekRankArtifactTransferDescriptorBatchMaximum ||
      (!expected_final &&
       fields.descriptor_count !=
           kDeepSeekRankArtifactTransferDescriptorBatchMaximum) ||
      fields.cumulative_descriptor_count != expected_cumulative ||
      fields.cumulative_descriptor_count >
          kDeepSeekRankArtifactTransferDescriptorMaximum ||
      fields.final_batch != expected_final ||
      fields.process_manifest_identity == 0 || fields.process_identity == 0 ||
      fields.pidfd_identity == 0 || fields.control_identity == 0 ||
      fields.challenge_identity == 0 ||
      !nonzero(fields.transfer_manifest_root) ||
      !nonzero(fields.artifact_admission_binding_root) ||
      !nonzero(fields.transfer_transaction_root) ||
      !nonzero(fields.descriptor_batch_root) ||
      !nonzero(fields.adopted_descriptor_set_root)) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact transfer acknowledgement is invalid");
  }

  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-transfer-ack:v1", 20);
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

template <class T, std::size_t N>
void put(std::array<std::byte, N>& output, std::size_t& cursor, T value) {
  using U = std::make_unsigned_t<T>;
  const auto bits = static_cast<U>(value);
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    output[cursor++] =
        static_cast<std::byte>((bits >> (index * 8U)) & 0xffU);
  }
}

template <class T>
T get(std::span<const std::byte> input, std::size_t& cursor) {
  using U = std::make_unsigned_t<T>;
  U value = 0;
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    value |= static_cast<U>(std::to_integer<unsigned>(input[cursor++]))
             << (index * 8U);
  }
  return static_cast<T>(value);
}

template <std::size_t N>
void put_digest(std::array<std::byte, N>& output, std::size_t& cursor,
                const Sha256Digest& value) {
  for (const auto byte : value.bytes) output[cursor++] = byte;
}

Sha256Digest get_digest(std::span<const std::byte> input,
                        std::size_t& cursor) {
  Sha256Digest value{};
  for (auto& byte : value.bytes) byte = input[cursor++];
  return value;
}

}  // namespace

DeepSeekRankArtifactTransferAck::DeepSeekRankArtifactTransferAck(
    DeepSeekRankArtifactTransferAckFields fields,
    Sha256Digest ack_root) noexcept
    : fields_(std::move(fields)), ack_root_(ack_root) {}

Result<DeepSeekRankArtifactTransferAck>
DeepSeekRankArtifactTransferAck::Create(
    DeepSeekRankArtifactTransferAckFields fields) {
  auto root = compile_root(fields);
  if (!root.ok()) return root.status();
  return DeepSeekRankArtifactTransferAck(std::move(fields), *root);
}

std::array<std::byte, kDeepSeekRankArtifactTransferAckBytes>
encode_deepseek_rank_artifact_transfer_ack(
    const DeepSeekRankArtifactTransferAck& ack) {
  std::array<std::byte, kDeepSeekRankArtifactTransferAckBytes> output{};
  std::size_t cursor = 0;
  put(output, cursor, kMagic);
  put(output, cursor, kFrameType);
  put(output, cursor, kFrameVersion);
  const auto& fields = ack.fields();
  put(output, cursor, fields.engine_epoch);
  put(output, cursor, fields.worker_generation);
  put(output, cursor, fields.world_size);
  put(output, cursor, fields.rank);
  put(output, cursor, fields.batch_index);
  put(output, cursor, fields.batch_count);
  put(output, cursor, fields.first_descriptor_ordinal);
  put(output, cursor, fields.descriptor_count);
  put(output, cursor, fields.cumulative_descriptor_count);
  put(output, cursor, static_cast<std::uint32_t>(fields.final_batch ? 1U : 0U));
  put(output, cursor, fields.process_manifest_identity);
  put(output, cursor, fields.process_identity);
  put(output, cursor, fields.pidfd_identity);
  put(output, cursor, fields.control_identity);
  put(output, cursor, fields.challenge_identity);
  put_digest(output, cursor, fields.transfer_manifest_root);
  put_digest(output, cursor, fields.artifact_admission_binding_root);
  put_digest(output, cursor, fields.transfer_transaction_root);
  put_digest(output, cursor, fields.descriptor_batch_root);
  put_digest(output, cursor, fields.adopted_descriptor_set_root);
  put_digest(output, cursor, ack.ack_root());
  return output;
}

Result<DeepSeekRankArtifactTransferAck>
decode_deepseek_rank_artifact_transfer_ack(
    std::span<const std::byte> frame) {
  if (frame.size() != kDeepSeekRankArtifactTransferAckBytes) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact transfer acknowledgement size is invalid");
  }
  std::size_t cursor = 0;
  if (get<std::uint32_t>(frame, cursor) != kMagic ||
      get<std::uint16_t>(frame, cursor) != kFrameType ||
      get<std::uint16_t>(frame, cursor) != kFrameVersion) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact transfer acknowledgement header is invalid");
  }
  DeepSeekRankArtifactTransferAckFields fields;
  fields.engine_epoch = get<std::uint64_t>(frame, cursor);
  fields.worker_generation = get<std::uint64_t>(frame, cursor);
  fields.world_size = get<std::uint32_t>(frame, cursor);
  fields.rank = get<std::uint32_t>(frame, cursor);
  fields.batch_index = get<std::uint32_t>(frame, cursor);
  fields.batch_count = get<std::uint32_t>(frame, cursor);
  fields.first_descriptor_ordinal = get<std::uint32_t>(frame, cursor);
  fields.descriptor_count = get<std::uint32_t>(frame, cursor);
  fields.cumulative_descriptor_count = get<std::uint32_t>(frame, cursor);
  const auto final_batch = get<std::uint32_t>(frame, cursor);
  if (final_batch > 1U) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact transfer acknowledgement boolean is invalid");
  }
  fields.final_batch = final_batch == 1U;
  fields.process_manifest_identity = get<std::uint64_t>(frame, cursor);
  fields.process_identity = get<std::uint64_t>(frame, cursor);
  fields.pidfd_identity = get<std::uint64_t>(frame, cursor);
  fields.control_identity = get<std::uint64_t>(frame, cursor);
  fields.challenge_identity = get<std::uint64_t>(frame, cursor);
  fields.transfer_manifest_root = get_digest(frame, cursor);
  fields.artifact_admission_binding_root = get_digest(frame, cursor);
  fields.transfer_transaction_root = get_digest(frame, cursor);
  fields.descriptor_batch_root = get_digest(frame, cursor);
  fields.adopted_descriptor_set_root = get_digest(frame, cursor);
  const auto expected_root = get_digest(frame, cursor);
  auto result = DeepSeekRankArtifactTransferAck::Create(std::move(fields));
  if (!result.ok()) return result.status();
  if (result->ack_root() != expected_root) {
    return Status::FailedPrecondition(
        "DeepSeek rank artifact transfer acknowledgement root differs");
  }
  return result;
}

static_assert(kDeepSeekRankArtifactTransferAckBytes ==
              8 + (2 * 8) + (8 * 4) + (5 * 8) + (6 * 32));

}  // namespace pih
