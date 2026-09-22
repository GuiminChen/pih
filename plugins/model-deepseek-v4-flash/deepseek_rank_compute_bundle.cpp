#include "pih/model/deepseek_rank_compute_bundle.h"

#include <new>
#include <unordered_set>
#include <utility>

namespace pih {

DeepSeekRankComputeBundle& DeepSeekRankComputeBundle::operator=(
    DeepSeekRankComputeBundle&& other) noexcept {
  if (this != &other) {
    this->~DeepSeekRankComputeBundle();
    ::new (static_cast<void*>(this))
        DeepSeekRankComputeBundle(std::move(other));
  }
  return *this;
}

namespace {

template <typename T>
Result<std::unique_ptr<T>> own(Result<T> result) {
  if (!result.ok()) return result.status();
  return std::make_unique<T>(std::move(*result));
}

bool same_descriptor(const DeepSeekPipelinePlanDescriptor& left,
                     const DeepSeekPipelinePlanDescriptor& right) {
  return left.engine_epoch == right.engine_epoch &&
         left.plan_sequence == right.plan_sequence && left.phase == right.phase &&
         left.token_count == right.token_count &&
         left.sequence_count == right.sequence_count;
}

bool empty_work(const DeepSeekRankComputePlanWork& work) {
  const bool common_empty =
      work.hash_router.empty() && work.learned_router.empty() &&
      work.decode_attention.empty() && work.chunk_attention.empty() &&
      work.dense_attention.empty() && work.mhc_attention.empty() &&
      work.mhc_feed_forward.empty() && work.endpoint.empty();
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  return common_empty && work.dspark == nullptr && work.dspark_mtp.empty();
#else
  return common_empty;
#endif
}

Result<std::vector<DeepSeekAttentionSequenceTransaction*>>
collect_attention_transactions(const DeepSeekRankComputePlanWork& work,
                               std::uint32_t expected) {
  std::vector<DeepSeekAttentionSequenceTransaction*> transactions;
  std::unordered_set<DeepSeekAttentionSequenceTransaction*> unique;
  const auto add = [&](DeepSeekAttentionSequenceTransaction* transaction) {
    if (transaction == nullptr) {
      return Status::InvalidArgument(
          "DeepSeek compute work contains a null attention transaction");
    }
    if (unique.insert(transaction).second) transactions.push_back(transaction);
    return Status::Ok();
  };
  for (const auto& layer : work.decode_attention) {
    auto status = add(layer.work.transaction);
    if (!status.ok()) return status;
  }
  for (const auto& layer : work.chunk_attention) {
    auto status = add(layer.work.transaction);
    if (!status.ok()) return status;
  }
  for (const auto& layer : work.dense_attention) {
    for (const auto& sequence : layer.sequences) {
      auto status = add(sequence.transaction);
      if (!status.ok()) return status;
    }
  }
  for (const auto& layer : work.mhc_attention) {
    for (const auto& sequence : layer.sequences) {
      auto status = add(sequence.transaction);
      if (!status.ok()) return status;
    }
  }
  for (const auto& layer : work.mhc_feed_forward) {
    for (const auto& sequence : layer.sequences) {
      auto status = add(sequence.transaction);
      if (!status.ok()) return status;
    }
  }
  for (const auto& sequence : work.endpoint) {
    auto status = add(sequence.transaction);
    if (!status.ok()) return status;
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  if (work.dspark != nullptr) {
    auto status = add(work.dspark->transaction);
    if (!status.ok()) return status;
  }
  for (const auto& stage : work.dspark_mtp) {
    auto* transaction = stage.prefill_resources.transaction != nullptr
        ? stage.prefill_resources.transaction
        : stage.resources.transaction;
    auto status = add(transaction);
    if (!status.ok()) return status;
  }
#endif
  if (transactions.size() != expected) {
    return Status::InvalidArgument(
        "DeepSeek compute work transaction set differs from sequence count");
  }
  return transactions;
}

}  // namespace

Result<DeepSeekRankComputeBundle> DeepSeekRankComputeBundle::Create(
    DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
    std::uint32_t maximum_tokens, DeepSeekExpertPager& pager,
    DeepSeekExpertTransferDriver& transfer,
    DeepSeekExpertKernelDriver& kernel,
    std::uintptr_t attention_compute_stream) {
  return CreateConfigured(stage, maximum_sequences, maximum_tokens, &pager,
                          &transfer, kernel, nullptr,
                          attention_compute_stream);
}

Result<DeepSeekRankComputeBundle> DeepSeekRankComputeBundle::CreateResident(
    DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
    std::uint32_t maximum_tokens, DeepSeekExpertKernelDriver& kernel,
    const DeepSeekResidentExpertBindings& resident_experts,
    std::uintptr_t attention_compute_stream) {
  return CreateConfigured(stage, maximum_sequences, maximum_tokens, nullptr,
                          nullptr, kernel, &resident_experts,
                          attention_compute_stream);
}

Result<DeepSeekRankComputeBundle>
DeepSeekRankComputeBundle::CreateConfigured(
    DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
    std::uint32_t maximum_tokens, DeepSeekExpertPager* pager,
    DeepSeekExpertTransferDriver* transfer,
    DeepSeekExpertKernelDriver& kernel,
    const DeepSeekResidentExpertBindings* resident_experts,
    std::uintptr_t attention_compute_stream) {
  const bool paged = pager != nullptr && transfer != nullptr &&
                     resident_experts == nullptr;
  const bool resident = pager == nullptr && transfer == nullptr &&
                        resident_experts != nullptr;
  if (maximum_sequences == 0 || maximum_tokens == 0 ||
      stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43 || (!paged && !resident) ||
      (paged && (pager->owned_layers() != stage.layers || pager->poisoned())) ||
      (resident && resident_experts->generation() == 0)) {
    return Status::InvalidArgument(
        "DeepSeek rank compute bundle topology or capacity is invalid");
  }
  DeepSeekRankComputeBundle bundle;
  bundle.stage_ = stage;
  bundle.attention_compute_stream_ = attention_compute_stream;

  auto expert_store = own(DeepSeekBoundExpertPlanProvider::Create(
      stage.layers, maximum_tokens));
  if (!expert_store.ok()) return expert_store.status();
  bundle.expert_store_ = std::move(*expert_store);

  auto router = own(DeepSeekBoundRouterStageWorkProvider::Create(
      stage.layers, *bundle.expert_store_));
  if (!router.ok()) return router.status();
  bundle.router_ = std::move(*router);

  auto decode = own(DeepSeekBoundDecodeAttentionWorkProvider::Create(
      stage.layers));
  if (!decode.ok()) return decode.status();
  bundle.decode_attention_ = std::move(*decode);

  auto chunk = own(DeepSeekBoundChunkAttentionWorkProvider::Create(
      stage.layers));
  if (!chunk.ok()) return chunk.status();
  bundle.chunk_attention_ = std::move(*chunk);

  auto dense = own(DeepSeekBoundDenseAttentionWorkProvider::Create(
      stage.layers, maximum_sequences));
  if (!dense.ok()) return dense.status();
  bundle.dense_attention_ = std::move(*dense);

  auto mhc = own(DeepSeekBoundMhcStageWorkProvider::Create(
      stage.layers, maximum_sequences));
  if (!mhc.ok()) return mhc.status();
  bundle.mhc_ = std::move(*mhc);

  if (stage.owns_embedding || stage.owns_lm_head) {
    auto endpoint = own(DeepSeekBoundEndpointStageWorkProvider::Create(
        maximum_sequences, stage.owns_embedding, stage.owns_lm_head));
    if (!endpoint.ok()) return endpoint.status();
    bundle.endpoint_ = std::move(*endpoint);
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  if (stage.owns_dspark) {
    auto dspark = own(DeepSeekBoundDsparkStageWorkProvider::Create(
        true, maximum_sequences));
    if (!dspark.ok()) return dspark.status();
    bundle.dspark_ = std::move(*dspark);
    auto mtp_provider = own(DeepSeekBoundDsparkMtpStageWorkProvider::Create(
        true));
    if (!mtp_provider.ok()) return mtp_provider.status();
    bundle.dspark_mtp_provider_ = std::move(*mtp_provider);
    auto mtp_backend = own(DeepSeekBoundDsparkMtpOperatorBackend::Create(
        *bundle.dspark_mtp_provider_));
    if (!mtp_backend.ok()) return mtp_backend.status();
    bundle.dspark_mtp_backend_ = std::move(*mtp_backend);
  }
#else
  if (stage.owns_dspark) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark is not present in this model bundle");
  }
#endif

  DeepSeekStageComputeStackDependencies dependencies;
  dependencies.decode_attention = bundle.decode_attention_.get();
  dependencies.chunk_attention = bundle.chunk_attention_.get();
  dependencies.dense_attention = bundle.dense_attention_.get();
  dependencies.router = bundle.router_.get();
  dependencies.expert_plan = bundle.expert_store_.get();
  dependencies.expert_pager = pager;
  dependencies.expert_transfer = transfer;
  dependencies.resident_experts = resident_experts;
  dependencies.expert_kernel = &kernel;
  dependencies.mhc = bundle.mhc_.get();
  dependencies.endpoint = bundle.endpoint_.get();
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  dependencies.dspark = bundle.dspark_.get();
  dependencies.dspark_mtp = bundle.dspark_mtp_backend_.get();
#endif
  auto stack = own(DeepSeekStageComputeStack::Create(stage, dependencies));
  if (!stack.ok()) return stack.status();
  bundle.stack_ = std::move(*stack);
  return bundle;
}

Status DeepSeekRankComputeBundle::bind_plan(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    DeepSeekRankComputePlanWork work) {
  if (bound_plan_.has_value() || descriptor.engine_epoch == 0 ||
      descriptor.plan_sequence == 0 || descriptor.sequence_count == 0 ||
      (descriptor.phase == DeepSeekPlanPhase::kDrain
           ? descriptor.token_count != 0
           : descriptor.token_count == 0)) {
    return Status::FailedPrecondition(
        "DeepSeek rank compute plan cannot be bound in its current state");
  }
  if (descriptor.phase == DeepSeekPlanPhase::kDrain) {
    if (!empty_work(work)) {
      return Status::InvalidArgument(
          "DeepSeek drain plan must not carry operator work");
    }
    bound_plan_ = descriptor;
    return Status::Ok();
  }

  if (attention_compute_stream_ == 0) {
    return Status::FailedPrecondition(
        "DeepSeek non-drain compute requires an owned attention stream");
  }
  auto transactions = collect_attention_transactions(
      work, descriptor.sequence_count);
  if (!transactions.ok()) return transactions.status();

  const auto fail = [this](Status status) {
    clear_plan();
    return status;
  };
  auto status = expert_store_->begin(descriptor);
  if (!status.ok()) return fail(status);
  status = router_->bind(descriptor, work.hash_router, work.learned_router);
  if (!status.ok()) return fail(status);
  if (descriptor.phase == DeepSeekPlanPhase::kDecode) {
    if (!work.chunk_attention.empty()) {
      return fail(Status::InvalidArgument(
          "DeepSeek decode plan cannot carry chunk attention work"));
    }
    status = decode_attention_->bind(descriptor, work.decode_attention);
  } else {
    if (!work.decode_attention.empty()) {
      return fail(Status::InvalidArgument(
          "DeepSeek chunk plan cannot carry decode attention work"));
    }
    status = chunk_attention_->bind(descriptor, work.chunk_attention);
  }
  if (!status.ok()) return fail(status);
  status = dense_attention_->bind(descriptor, work.dense_attention);
  if (!status.ok()) return fail(status);
  status = mhc_->bind(descriptor, work.mhc_attention, work.mhc_feed_forward);
  if (!status.ok()) return fail(status);
  if (endpoint_ != nullptr) {
    status = endpoint_->bind(descriptor, work.endpoint);
    if (!status.ok()) return fail(status);
  } else if (!work.endpoint.empty()) {
    return fail(Status::InvalidArgument(
        "DeepSeek non-endpoint rank received endpoint work"));
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  const bool runs_dspark = stage_.owns_dspark &&
      (descriptor.phase == DeepSeekPlanPhase::kPrefill ||
       descriptor.phase == DeepSeekPlanPhase::kDecode);
  if (runs_dspark) {
    if (work.dspark == nullptr || dspark_ == nullptr ||
        dspark_mtp_provider_ == nullptr || work.dspark_mtp.size() != 3) {
      return fail(Status::InvalidArgument(
          "DeepSeek drafting rank is missing DSpark work"));
    }
    status = dspark_mtp_provider_->bind(descriptor, work.dspark_mtp);
    if (!status.ok()) return fail(status);
    status = dspark_->bind(descriptor, *work.dspark);
    if (!status.ok()) return fail(status);
  } else if (work.dspark != nullptr || !work.dspark_mtp.empty()) {
    return fail(Status::InvalidArgument(
        "DeepSeek plan received unexpected DSpark work"));
  }
#else
  if (stage_.owns_dspark) {
    return fail(Status::FailedPrecondition(
        "DeepSeek DSpark is not present in this model bundle"));
  }
#endif
  auto attention_driver = DeepSeekAttentionPlanComputeDriver::Create(
      *stack_, *transactions, attention_compute_stream_);
  if (!attention_driver.ok()) return fail(attention_driver.status());
  attention_driver_ = std::make_unique<DeepSeekAttentionPlanComputeDriver>(
      std::move(*attention_driver));
  bound_plan_ = descriptor;
  return Status::Ok();
}

Status DeepSeekRankComputeBundle::launch(
    const DeepSeekPipelinePlanDescriptor& plan,
    const DeepSeekStagePlan& stage) {
  if (!bound_plan_.has_value() || !same_descriptor(*bound_plan_, plan) ||
      stage.rank != stage_.rank || stage.layers != stage_.layers ||
      stage.owns_embedding != stage_.owns_embedding ||
      stage.owns_lm_head != stage_.owns_lm_head ||
      stage.owns_dspark != stage_.owns_dspark) {
    return Status::FailedPrecondition(
        "DeepSeek rank compute launch lacks an exact bound plan");
  }
  auto* driver = attention_driver_ == nullptr
                     ? static_cast<DeepSeekStageComputeDriver*>(stack_.get())
                     : static_cast<DeepSeekStageComputeDriver*>(
                           attention_driver_.get());
  // A failing launch can already have enqueued device work. Only a successful
  // poll proves retirement; never treat a partial launch as unsubmitted.
  launched_ = true;
  return driver->launch(plan, stage);
}

Status DeepSeekRankComputeBundle::abort_bound_plan(
    const DeepSeekPipelinePlanDescriptor& descriptor) noexcept {
  if (!can_abort_bound_plan(descriptor)) {
    return Status::FailedPrecondition(
        "DeepSeek rank compute binding cannot be aborted");
  }
  if (!bound_plan_.has_value()) return Status::Ok();
  clear_plan();
  return Status::Ok();
}

bool DeepSeekRankComputeBundle::can_abort_bound_plan(
    const DeepSeekPipelinePlanDescriptor& descriptor) const noexcept {
  return !bound_plan_.has_value() ||
         (!launched_ && same_descriptor(*bound_plan_, descriptor));
}

Result<DeepSeekStageComputeStatus> DeepSeekRankComputeBundle::poll() {
  if (!bound_plan_.has_value()) {
    return Status::FailedPrecondition(
        "DeepSeek rank compute bundle has no active plan");
  }
  auto* driver = attention_driver_ == nullptr
                     ? static_cast<DeepSeekStageComputeDriver*>(stack_.get())
                     : static_cast<DeepSeekStageComputeDriver*>(
                           attention_driver_.get());
  auto result = driver->poll();
  // A poll error does not prove that submitted device work reached a safe
  // completion frontier. Retain the full bound plan so the engine enters its
  // fail-stop quarantine instead of releasing or reusing borrowed resources.
  if (!result.ok() || *result == DeepSeekStageComputeStatus::kError) {
    return result;
  }
  if (*result == DeepSeekStageComputeStatus::kSuccess) {
    // The transaction driver has observed completion and committed its banks.
    // Reset layer borrowers before releasing the plan that owns transactions.
    auto status = decode_attention_->reset_completed();
    if (!status.ok()) return status;
    status = chunk_attention_->reset_completed();
    if (!status.ok()) return status;
    clear_plan();
  }
  return result;
}

void DeepSeekRankComputeBundle::clear_plan() noexcept {
  attention_driver_.reset();
  expert_store_->clear();
  router_->clear();
  decode_attention_->clear();
  chunk_attention_->clear();
  dense_attention_->clear();
  mhc_->clear();
  if (endpoint_ != nullptr) endpoint_->clear();
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  if (dspark_ != nullptr) dspark_->clear();
  if (dspark_mtp_provider_ != nullptr) dspark_mtp_provider_->clear();
#endif
  bound_plan_.reset();
  launched_ = false;
}

}  // namespace pih
