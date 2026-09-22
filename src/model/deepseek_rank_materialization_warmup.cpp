#include "pih/model/deepseek_rank_materialization_warmup.h"

#include <algorithm>
#include <utility>

#include "pih/core/canonical_hash.h"
#include "pih/core/checked_math.h"

namespace pih {
namespace {

bool nonzero(const Sha256Digest& value) noexcept {
  return value != Sha256Digest{};
}

Result<Sha256Digest> compile_root_set(
    std::string_view domain, std::span<const Sha256Digest> roots) {
  if (roots.empty() || roots.size() > 4U) {
    return Status::InvalidArgument(
        "DeepSeek warmup root set geometry is invalid");
  }
  auto builder = CanonicalHashBuilder::Create(
      domain, static_cast<std::uint32_t>(roots.size() + 1U));
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(
      1, static_cast<std::uint32_t>(roots.size()));
  for (std::size_t index = 0; status.ok() && index < roots.size(); ++index) {
    status = builder->add_hash(
        static_cast<std::uint16_t>(index + 2U), roots[index]);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> compile_prefault_rank_layout_set(
    std::span<const Sha256Digest> roots) {
  if (roots.empty() || roots.size() > 4U) {
    return Status::InvalidArgument(
        "DeepSeek prefault rank layout set geometry is invalid");
  }
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-node-prefault-rank-layout-set:v1",
      static_cast<std::uint32_t>(roots.size() + 1U));
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(
      1, static_cast<std::uint32_t>(roots.size()));
  for (std::size_t index = 0; status.ok() && index < roots.size(); ++index) {
    status = builder->add_hash(
        static_cast<std::uint16_t>(index + 100U), roots[index]);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

Status add_total(std::uint64_t& total, std::uint64_t value) {
  auto next = checked_add_u64(total, value);
  if (!next.ok()) return next.status();
  total = *next;
  return Status::Ok();
}

Result<Sha256Digest> compile_weight_record(
    std::uint32_t rank,
    const DeepSeekRankMaterializationCompletionFields& completion) {
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-materialization-warm-weight:v1", 3);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(1, rank);
  if (status.ok()) status = builder->add_hash(2, completion.weight_layout_root);
  if (status.ok()) status = builder->add_hash(3, completion.weight_seal_root);
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> compile_cuda_record(
    std::uint32_t rank,
    const DeepSeekRankMaterializationCompletionFields& completion) {
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-materialization-warm-cuda-allocation:v1", 5);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(1, rank);
  if (status.ok()) status = builder->add_u32(
      2, static_cast<std::uint32_t>(completion.device_ordinal));
  if (status.ok()) {
    status = builder->add_u64(3, completion.fixed_weight_backing_bytes);
  }
  if (status.ok()) {
    status = builder->add_u64(
        4, completion.fixed_weight_allocation_generation);
  }
  if (status.ok()) status = builder->add_hash(5, completion.cuda_allocation_root);
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> compile_pinned_record(
    std::uint32_t rank,
    const DeepSeekRankMaterializationCompletionFields& completion) {
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-materialization-warm-pinned-allocation:v1", 5);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(1, rank);
  if (status.ok()) {
    status = builder->add_u32(
        2, nonzero(completion.pinned_allocation_root) ? 1U : 0U);
  }
  if (status.ok()) status = builder->add_u64(3, completion.pinned_staging_bytes);
  if (status.ok()) {
    status = builder->add_u64(
        4, completion.pinned_staging_allocation_generation);
  }
  if (status.ok()) {
    status = builder->add_bytes(5, completion.pinned_allocation_root.bytes);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

}  // namespace

Result<Sha256Digest>
compile_deepseek_rank_materialization_warm_grant_set_root(
    std::span<const DeepSeekRankMaterializationGrantFields> grants) {
  if (grants.empty() || grants.size() > 4U) {
    return Status::InvalidArgument("DeepSeek warm grant set geometry is invalid");
  }
  std::vector<Sha256Digest> roots;
  roots.reserve(grants.size());
  for (std::uint32_t rank = 0; rank < grants.size(); ++rank) {
    auto root = compile_deepseek_rank_materialization_grant_root(grants[rank]);
    if (!root.ok() || grants[rank].rank != rank ||
        grants[rank].world_size != grants.size()) {
      return Status::FailedPrecondition("DeepSeek warm grant set is invalid");
    }
    roots.push_back(*root);
  }
  return compile_root_set("pih:deepseek-materialization-warm-grant-set:v1",
                          roots);
}

DeepSeekRankMaterializationWarmSeal::DeepSeekRankMaterializationWarmSeal(
    std::uint64_t engine_epoch, std::uint64_t worker_generation,
    std::uint32_t world_size, std::uint64_t completion_monotonic_ns,
    std::uint64_t summed_rank_selected_page_bytes,
    std::uint64_t node_selected_page_union_bytes,
    std::uint64_t node_duplicate_selected_page_bytes,
    std::uint64_t fixed_weight_backing_bytes,
    std::uint64_t pinned_staging_bytes, bool production_eligible,
    bool dspark_enabled, Sha256Digest node_prefault_layout_root,
    Sha256Digest grant_set_root, Sha256Digest completion_set_root,
    Sha256Digest seal_root) noexcept
    : engine_epoch_(engine_epoch), worker_generation_(worker_generation),
      world_size_(world_size),
      completion_monotonic_ns_(completion_monotonic_ns),
      summed_rank_selected_page_bytes_(summed_rank_selected_page_bytes),
      node_selected_page_union_bytes_(node_selected_page_union_bytes),
      node_duplicate_selected_page_bytes_(
          node_duplicate_selected_page_bytes),
      fixed_weight_backing_bytes_(fixed_weight_backing_bytes),
      pinned_staging_bytes_(pinned_staging_bytes),
      production_eligible_(production_eligible),
      dspark_enabled_(dspark_enabled),
      node_prefault_layout_root_(node_prefault_layout_root),
      grant_set_root_(grant_set_root),
      completion_set_root_(completion_set_root), seal_root_(seal_root) {}

Result<DeepSeekRankMaterializationWarmSeal>
DeepSeekRankMaterializationWarmSeal::Compile(
    std::span<const DeepSeekRankMaterializationGrantFields> grants,
    std::span<const DeepSeekRankArtifactPrefaultLayout> rank_layouts,
    const DeepSeekNodeArtifactPrefaultLayout& node_layout,
    const Sha256Digest& metadata_transaction_root,
    std::span<const DeepSeekRankMaterializationCompletionFields>
        completions) {
  if (grants.empty() || grants.size() > 4U ||
      rank_layouts.size() != grants.size() ||
      completions.size() != grants.size() ||
      node_layout.world_size != grants.size() ||
      node_layout.page_bytes != kDeepSeekArtifactPrefaultPageBytes ||
      node_layout.summed_rank_selected_page_bytes == 0 ||
      node_layout.node_selected_page_union_bytes == 0 ||
      node_layout.node_selected_page_union_bytes >
          node_layout.summed_rank_selected_page_bytes ||
      !nonzero(node_layout.rank_layout_set_root) ||
      !nonzero(node_layout.layout_root) ||
      !nonzero(node_layout.node_page_union_root) ||
      !nonzero(metadata_transaction_root)) {
    return Status::InvalidArgument(
        "DeepSeek materialization warm seal geometry is invalid");
  }

  const auto& first = grants.front();
  std::vector<Sha256Digest> grant_roots;
  std::vector<Sha256Digest> completion_roots;
  std::vector<Sha256Digest> rank_layout_roots;
  std::vector<Sha256Digest> weight_roots;
  std::vector<Sha256Digest> cuda_roots;
  std::vector<Sha256Digest> pinned_roots;
  grant_roots.reserve(grants.size());
  completion_roots.reserve(grants.size());
  rank_layout_roots.reserve(grants.size());
  weight_roots.reserve(grants.size());
  cuda_roots.reserve(grants.size());
  pinned_roots.reserve(grants.size());
  std::uint64_t completion_monotonic_ns = 0;
  std::uint64_t summed_mapped_bytes = 0;
  std::uint64_t summed_rank_pages = 0;
  std::uint64_t fixed_backing_bytes = 0;
  std::uint64_t fixed_payload_bytes = 0;
  std::uint64_t pinned_bytes = 0;

  for (std::uint32_t rank = 0; rank < grants.size(); ++rank) {
    const auto& grant = grants[rank];
    const auto& layout = rank_layouts[rank];
    const auto& completion = completions[rank];
    auto grant_root = compile_deepseek_rank_materialization_grant_root(grant);
    if (!grant_root.ok()) return grant_root.status();
    auto completion_root =
        compile_deepseek_rank_materialization_completion_root(completion);
    if (!completion_root.ok()) return completion_root.status();
    if (grant.engine_epoch != first.engine_epoch ||
        grant.worker_generation != first.worker_generation ||
        grant.world_size != grants.size() || grant.rank != rank ||
        grant.gpu_family != first.gpu_family ||
        grant.residency != first.residency ||
        grant.production_eligible != first.production_eligible ||
        grant.dspark_enabled != first.dspark_enabled ||
        grant.deadline_ns != first.deadline_ns ||
        grant.metadata_transaction_root != metadata_transaction_root ||
        layout.rank != rank || layout.interval_count == 0 ||
        layout.page_bytes != node_layout.page_bytes ||
        layout.mapped_interval_bytes == 0 ||
        layout.selected_page_union_bytes < layout.mapped_interval_bytes ||
        (layout.selected_page_union_bytes % layout.page_bytes) != 0 ||
        !nonzero(layout.mapping_plan_root) || !nonzero(layout.layout_root) ||
        completion.engine_epoch != grant.engine_epoch ||
        completion.worker_generation != grant.worker_generation ||
        completion.world_size != grant.world_size ||
        completion.rank != rank ||
        completion.process_manifest_identity !=
            grant.process_manifest_identity ||
        completion.process_identity != grant.process_identity ||
        completion.pidfd_identity != grant.pidfd_identity ||
        completion.control_identity != grant.control_identity ||
        completion.challenge_identity != grant.challenge_identity ||
        completion.device_ordinal != grant.device_ordinal ||
        completion.gpu_family != grant.gpu_family ||
        completion.residency != grant.residency ||
        completion.production_eligible != grant.production_eligible ||
        completion.dspark_enabled != grant.dspark_enabled ||
        completion.deadline_ns != grant.deadline_ns ||
        completion.profile_envelope_root != grant.profile_envelope_root ||
        completion.device_observation_root != grant.device_observation_root ||
        completion.capacity_plan_instance_root !=
            grant.capacity_plan_instance_root ||
        completion.post_mapping_seal_root !=
            grant.post_mapping_seal_root ||
        completion.metadata_transaction_root != metadata_transaction_root ||
        completion.mapping_owner_root != grant.mapping_owner_root ||
        completion.grant_root != *grant_root ||
        completion.prefault_layout_root != layout.layout_root ||
        completion.mapped_interval_bytes != layout.mapped_interval_bytes ||
        completion.selected_page_union_bytes !=
            layout.selected_page_union_bytes) {
      return Status::FailedPrecondition(
          "DeepSeek materialization completion differs from authority");
    }
    auto weight_root = compile_weight_record(rank, completion);
    if (!weight_root.ok()) return weight_root.status();
    auto cuda_root = compile_cuda_record(rank, completion);
    if (!cuda_root.ok()) return cuda_root.status();
    auto pinned_root = compile_pinned_record(rank, completion);
    if (!pinned_root.ok()) return pinned_root.status();
    grant_roots.push_back(*grant_root);
    completion_roots.push_back(*completion_root);
    rank_layout_roots.push_back(layout.layout_root);
    weight_roots.push_back(*weight_root);
    cuda_roots.push_back(*cuda_root);
    pinned_roots.push_back(*pinned_root);
    completion_monotonic_ns = std::max(
        completion_monotonic_ns, completion.completion_monotonic_ns);
    auto status = add_total(summed_mapped_bytes,
                            completion.mapped_interval_bytes);
    if (status.ok()) {
      status = add_total(summed_rank_pages,
                         completion.selected_page_union_bytes);
    }
    if (status.ok()) {
      status = add_total(fixed_backing_bytes,
                         completion.fixed_weight_backing_bytes);
    }
    if (status.ok()) {
      status = add_total(fixed_payload_bytes,
                         completion.fixed_weight_payload_bytes);
    }
    if (status.ok()) {
      status = add_total(pinned_bytes, completion.pinned_staging_bytes);
    }
    if (!status.ok()) return status;
  }

  auto rank_layout_set =
      compile_prefault_rank_layout_set(rank_layout_roots);
  if (!rank_layout_set.ok()) return rank_layout_set.status();
  auto recomposed_node_bytes = checked_add_u64(
      node_layout.node_selected_page_union_bytes,
      node_layout.node_duplicate_selected_page_bytes);
  if (!recomposed_node_bytes.ok()) return recomposed_node_bytes.status();
  if (summed_rank_pages != node_layout.summed_rank_selected_page_bytes ||
      *recomposed_node_bytes != summed_rank_pages ||
      *rank_layout_set != node_layout.rank_layout_set_root) {
    return Status::FailedPrecondition(
        "DeepSeek node prefault projection differs from rank completions");
  }

  auto grant_set_root = compile_root_set(
      "pih:deepseek-materialization-warm-grant-set:v1", grant_roots);
  if (!grant_set_root.ok()) return grant_set_root.status();
  auto completion_set_root = compile_root_set(
      "pih:deepseek-materialization-warm-completion-set:v1",
      completion_roots);
  if (!completion_set_root.ok()) return completion_set_root.status();
  auto weight_set_root = compile_root_set(
      "pih:deepseek-materialization-warm-weight-set:v1", weight_roots);
  if (!weight_set_root.ok()) return weight_set_root.status();
  auto cuda_set_root = compile_root_set(
      "pih:deepseek-materialization-warm-cuda-allocation-set:v1",
      cuda_roots);
  if (!cuda_set_root.ok()) return cuda_set_root.status();
  auto pinned_set_root = compile_root_set(
      "pih:deepseek-materialization-warm-pinned-allocation-set:v1",
      pinned_roots);
  if (!pinned_set_root.ok()) return pinned_set_root.status();

  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-materialization-warm-seal:v1", 23);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u64(1, first.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, first.worker_generation);
  if (status.ok()) status = builder->add_u32(3, first.world_size);
  if (status.ok()) {
    status = builder->add_u32(
        4, static_cast<std::uint32_t>(first.gpu_family));
  }
  if (status.ok()) {
    status = builder->add_u32(
        5, static_cast<std::uint32_t>(first.residency));
  }
  if (status.ok()) {
    status = builder->add_u32(6, first.production_eligible ? 1U : 0U);
  }
  if (status.ok()) {
    status = builder->add_u32(7, first.dspark_enabled ? 1U : 0U);
  }
  if (status.ok()) status = builder->add_u64(8, first.deadline_ns);
  if (status.ok()) status = builder->add_u64(9, completion_monotonic_ns);
  if (status.ok()) status = builder->add_u64(10, summed_mapped_bytes);
  if (status.ok()) status = builder->add_u64(11, summed_rank_pages);
  if (status.ok()) {
    status = builder->add_u64(
        12, node_layout.node_selected_page_union_bytes);
  }
  if (status.ok()) {
    status = builder->add_u64(
        13, node_layout.node_duplicate_selected_page_bytes);
  }
  if (status.ok()) status = builder->add_u64(14, fixed_backing_bytes);
  if (status.ok()) status = builder->add_u64(15, fixed_payload_bytes);
  if (status.ok()) status = builder->add_u64(16, pinned_bytes);
  if (status.ok()) status = builder->add_hash(17, metadata_transaction_root);
  if (status.ok()) status = builder->add_hash(18, node_layout.layout_root);
  if (status.ok()) status = builder->add_hash(19, *grant_set_root);
  if (status.ok()) status = builder->add_hash(20, *completion_set_root);
  if (status.ok()) status = builder->add_hash(21, *weight_set_root);
  if (status.ok()) status = builder->add_hash(22, *cuda_set_root);
  if (status.ok()) status = builder->add_hash(23, *pinned_set_root);
  if (!status.ok()) return status;
  auto seal_root = builder->finalize();
  if (!seal_root.ok()) return seal_root.status();
  return DeepSeekRankMaterializationWarmSeal(
      first.engine_epoch, first.worker_generation, first.world_size,
      completion_monotonic_ns, summed_rank_pages,
      node_layout.node_selected_page_union_bytes,
      node_layout.node_duplicate_selected_page_bytes, fixed_backing_bytes,
      pinned_bytes, first.production_eligible, first.dspark_enabled,
      node_layout.layout_root, *grant_set_root, *completion_set_root,
      *seal_root);
}

DeepSeekRankMaterializationWarmupCoordinator::
    DeepSeekRankMaterializationWarmupCoordinator(
        DeepSeekRankMaterializationGrantCoordinator& grants,
        const DeepSeekRankArtifactMetadataTransferTransaction& metadata,
        DeepSeekRankMaterializationWarmupOperations& operations) noexcept
    : grants_(&grants), metadata_(&metadata), operations_(&operations),
      completions_(metadata.world_size()) {}

Result<DeepSeekRankMaterializationWarmupCoordinator>
DeepSeekRankMaterializationWarmupCoordinator::Create(
    DeepSeekRankMaterializationGrantCoordinator& grants,
    const DeepSeekRankArtifactMetadataTransferTransaction& metadata,
    DeepSeekRankMaterializationWarmupOperations& operations) {
  const auto fail_create = [&](Status cause)
      -> Result<DeepSeekRankMaterializationWarmupCoordinator> {
    const auto* first = grants.grant(0);
    if (first != nullptr) {
      (void)operations.abort_generation(
          first->engine_epoch, first->worker_generation, cause);
    }
    if (grants.supervisor_ != nullptr) (void)grants.fail(cause);
    return cause;
  };
  if (!grants.complete() || grants.poisoned() ||
      grants.acknowledgment_count() == 0 ||
      grants.acknowledgment_count() != metadata.world_size() ||
      !metadata.complete() || metadata.poisoned() ||
      !metadata.retains_descriptor_transaction() ||
      metadata.world_size() < 1 || metadata.world_size() > 4 ||
      grants.grants_.size() != metadata.world_size() ||
      grants.supervisor_ == nullptr || grants.supervisor_->failed()) {
    return fail_create(Status::FailedPrecondition(
        "DeepSeek materialization warmup antecedents are incomplete"));
  }
  for (std::uint32_t rank = 0; rank < metadata.world_size(); ++rank) {
    const auto* grant = grants.grant(rank);
    const auto* ack = grants.acknowledgment(rank);
    if (grant == nullptr || ack == nullptr ||
        grant->metadata_transaction_root != metadata.transaction_root() ||
        grant->mapping_owner_root !=
            metadata.expected_mapping_owner_root(rank)) {
      return fail_create(Status::FailedPrecondition(
          "DeepSeek materialization warmup authority drifted"));
    }
  }
  return DeepSeekRankMaterializationWarmupCoordinator(
      grants, metadata, operations);
}

Status DeepSeekRankMaterializationWarmupCoordinator::fail(
    Status cause) noexcept {
  if (poisoned_) return cause;
  poisoned_ = true;
  seal_.reset();
  if (cause.ok()) {
    cause = Status::Internal(
        "DeepSeek materialization warmup coordination failed");
  }
  const auto* first = grants_->grant(0);
  if (first != nullptr) {
    (void)operations_->abort_generation(
        first->engine_epoch, first->worker_generation, cause);
  }
  (void)grants_->fail(cause);
  return cause;
}

Status DeepSeekRankMaterializationWarmupCoordinator::validate_completion(
    std::uint32_t rank,
    const DeepSeekRankMaterializationCompletionFields& completion) const {
  const auto* grant = grants_->grant(rank);
  if (grant == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek materialization warmup grant disappeared");
  }
  auto grant_root = compile_deepseek_rank_materialization_grant_root(*grant);
  if (!grant_root.ok()) return grant_root.status();
  auto completion_root =
      compile_deepseek_rank_materialization_completion_root(completion);
  if (!completion_root.ok()) return completion_root.status();
  const auto& layout = metadata_->expected_prefault_layout(rank);
  if (completion.engine_epoch != grant->engine_epoch ||
      completion.worker_generation != grant->worker_generation ||
      completion.world_size != grant->world_size || completion.rank != rank ||
      completion.process_manifest_identity !=
          grant->process_manifest_identity ||
      completion.process_identity != grant->process_identity ||
      completion.pidfd_identity != grant->pidfd_identity ||
      completion.control_identity != grant->control_identity ||
      completion.challenge_identity != grant->challenge_identity ||
      completion.device_ordinal != grant->device_ordinal ||
      completion.gpu_family != grant->gpu_family ||
      completion.residency != grant->residency ||
      completion.production_eligible != grant->production_eligible ||
      completion.dspark_enabled != grant->dspark_enabled ||
      completion.deadline_ns != grant->deadline_ns ||
      completion.profile_envelope_root != grant->profile_envelope_root ||
      completion.device_observation_root != grant->device_observation_root ||
      completion.capacity_plan_instance_root !=
          grant->capacity_plan_instance_root ||
      completion.post_mapping_seal_root !=
          grant->post_mapping_seal_root ||
      completion.metadata_transaction_root != metadata_->transaction_root() ||
      completion.mapping_owner_root !=
          metadata_->expected_mapping_owner_root(rank) ||
      completion.grant_root != *grant_root ||
      completion.prefault_layout_root != layout.layout_root ||
      completion.mapped_interval_bytes != layout.mapped_interval_bytes ||
      completion.selected_page_union_bytes !=
          layout.selected_page_union_bytes) {
    return Status::FailedPrecondition(
        "DeepSeek materialization warmup completion is foreign");
  }
  return Status::Ok();
}

std::size_t DeepSeekRankMaterializationWarmupCoordinator::completion_count()
    const noexcept {
  std::size_t count = 0;
  for (const auto& completion : completions_) {
    if (completion) ++count;
  }
  return count;
}

Status DeepSeekRankMaterializationWarmupCoordinator::advance() {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek materialization warmup coordinator is poisoned");
  }
  if (seal_) return Status::Ok();
  const auto ensure_before_deadline = [&]() -> Status {
    auto observed = operations_->monotonic_now_ns();
    if (!observed.ok()) return observed.status();
    return *observed < grants_->deadline_ns()
               ? Status::Ok()
               : Status::DeadlineExceeded(
                     "DeepSeek materialization warmup deadline expired");
  };
  auto status = ensure_before_deadline();
  if (!status.ok()) return fail(status);
  status = grants_->supervisor_->poll();
  if (!status.ok()) return fail(status);
  status = ensure_before_deadline();
  if (!status.ok()) return fail(status);

  bool pending = false;
  for (std::uint32_t rank = 0; rank < completions_.size(); ++rank) {
    if (completions_[rank]) continue;
    const auto* handle = grants_->supervisor_->process_handle(rank);
    if (handle == nullptr) {
      return fail(Status::FailedPrecondition(
          "DeepSeek materialization warmup process disappeared"));
    }
    status = ensure_before_deadline();
    if (!status.ok()) return fail(status);
    auto frame = operations_->poll_completion(*handle);
    if (!frame.ok()) {
      if (frame.status().code() == StatusCode::kUnavailable) {
        pending = true;
        continue;
      }
      return fail(frame.status());
    }
    if (!frame->has_value()) {
      pending = true;
      continue;
    }
    status = ensure_before_deadline();
    if (!status.ok()) return fail(status);
    auto completion =
        decode_deepseek_rank_materialization_completion(**frame);
    if (!completion.ok()) return fail(completion.status());
    status = validate_completion(rank, *completion);
    if (!status.ok()) return fail(status);
    completions_[rank] = std::move(*completion);
  }
  if (pending || completion_count() != completions_.size()) {
    return Status::Unavailable(
        "DeepSeek materialization warmup completion is pending");
  }
  status = ensure_before_deadline();
  if (!status.ok()) return fail(status);

  std::vector<DeepSeekRankMaterializationGrantFields> grants;
  std::vector<DeepSeekRankArtifactPrefaultLayout> layouts;
  std::vector<DeepSeekRankMaterializationCompletionFields> completions;
  grants.reserve(completions_.size());
  layouts.reserve(completions_.size());
  completions.reserve(completions_.size());
  for (std::uint32_t rank = 0; rank < completions_.size(); ++rank) {
    grants.push_back(*grants_->grant(rank));
    layouts.push_back(metadata_->expected_prefault_layout(rank));
    completions.push_back(*completions_[rank]);
  }
  auto seal = DeepSeekRankMaterializationWarmSeal::Compile(
      grants, layouts, metadata_->expected_node_prefault_layout(),
      metadata_->transaction_root(), completions);
  if (!seal.ok()) return fail(seal.status());
  seal_ = std::move(*seal);
  return Status::Ok();
}

}  // namespace pih
