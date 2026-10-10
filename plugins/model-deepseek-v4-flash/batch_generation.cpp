#include "generation.h"
#include <algorithm>
#include <limits>
#include <thread>
#include <stdexcept>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <atomic>

namespace pih::deepseek_plugin {
struct BatchGenerator::Impl {
  struct Job {
    std::vector<std::uint32_t> prompt;
    std::uint32_t maximum, minimum, capacity, processed = 0;
    std::uint64_t id, generation;
    DeepSeekRequestSamplingConfig sampling;
    std::chrono::steady_clock::time_point deadline;
    std::atomic<bool> cancelled{false};
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<DeepSeekAcceptedTokenSnapshot> outputs;
    DeepSeekAcceptedTokenSnapshot accepted;
    Status status = Status::Ok();
    bool done = false;
  };
  BatchEngine& engine;
  std::uint32_t sequences, tokens, pending;
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::shared_ptr<Job>> queued;
  std::vector<std::shared_ptr<Job>> active;
  std::atomic<bool> closing{false};
  std::thread worker;
  std::uint64_t next_plan = 1;
  Impl(BatchEngine& engine, std::uint32_t sequences, std::uint32_t tokens, std::uint32_t pending)
      : engine(engine), sequences(sequences), tokens(tokens), pending(pending) {
    if (!sequences || !tokens || !pending) throw std::invalid_argument("DeepSeek batch bounds must be positive");
    worker = std::thread([this] { Run(); });
  }
  ~Impl() {
    closing = true; ready.notify_all();
    if (worker.joinable()) worker.join();
  }
  void Complete(const std::shared_ptr<Job>& job, Status status) {
    std::lock_guard lock(job->mutex);
    job->status = std::move(status); job->done = true; job->ready.notify_all();
  }
  Status Interrupted(const Job& job) {
    if (closing || job.cancelled) return Status::Unavailable("DeepSeek generation cancelled");
    if (std::chrono::steady_clock::now() >= job.deadline)
      return Status::DeadlineExceeded("DeepSeek generation deadline elapsed");
    return Status::Ok();
  }
  Status Retire(const std::shared_ptr<Job>& job, bool cancel) {
    auto status = cancel ? engine.cancel_request(job->id,job->generation)
                         : engine.finish_request(job->id,job->generation);
    if (!status.ok()) return status;
    return engine.retire_request(job->id,job->generation);
  }
  void Fail(Status status) {
    engine.fail_all();
    for (auto& job : active) Complete(job,status);
    active.clear();
    std::lock_guard lock(mutex);
    for (auto& job : queued) Complete(job,status);
    queued.clear(); closing = true;
  }
  void Run() noexcept {
    try {
      auto binding = engine.bind_execution_thread();
      if (!binding.ok()) { Fail(binding); return; }
      for (;;) {
        {
          std::unique_lock lock(mutex);
          if (active.empty() && queued.empty() && !closing)
            ready.wait(lock,[this] { return closing || !queued.empty(); });
          while (active.size() < sequences && !queued.empty()) {
            auto job = queued.front(); queued.pop_front();
            auto status = Interrupted(*job);
            if (!status.ok()) { Complete(job,status); continue; }
            status = engine.submit_request(job->id,job->generation);
            if (!status.ok()) { Complete(job,status); continue; }
            status = engine.configure_ledger(job->id,job->generation,job->prompt.size(),job->maximum,job->minimum);
            if (status.ok()) {
              job->sampling.config_id=job->generation;
              status=engine.configure_sampling(job->id,job->generation,job->sampling);
            }
            if (!status.ok()) {
              auto retired=Retire(job,true);
              if (!retired.ok()) { lock.unlock(); Fail(retired); return; }
              Complete(job,status); continue;
            }
            active.push_back(job);
          }
        }
        for (auto iterator=active.begin(); iterator!=active.end();) {
          auto status=Interrupted(**iterator);
          if (!status.ok()) {
            auto retired=Retire(*iterator,true);
            if (!retired.ok()) { Fail(retired); return; }
            Complete(*iterator,status); iterator=active.erase(iterator);
          } else ++iterator;
        }
        if (active.empty()) { if (closing) return; continue; }
        std::vector<std::shared_ptr<Job>> selected;
        bool prefill=false, sample=false;
        DeepSeekProductionRankPlanSeed seed;
        for (const auto& job : active) {
          { std::lock_guard lock(job->mutex); if (job->outputs.size() >= 16) continue; }
          const bool candidate_prefill=job->processed<job->prompt.size();
          const bool candidate_sample=!candidate_prefill || job->processed+1==job->prompt.size();
          if (selected.empty()) { prefill=candidate_prefill; sample=candidate_sample; }
          if (candidate_prefill!=prefill || candidate_sample!=sample) continue;
          if (selected.size() >= std::min(sequences,tokens)) break;
          const auto position=candidate_prefill ? job->processed : job->accepted.model_processed_length;
          if (position >= job->capacity || (!candidate_prefill && !job->accepted.pending_input_token)) {
            Fail(Status::Internal("DeepSeek batch decode frontier invalid")); return;
          }
          selected.push_back(job);
          seed.positions.push_back(static_cast<std::uint32_t>(position));
          seed.token_ids.push_back(candidate_prefill ? job->prompt[job->processed] : *job->accepted.pending_input_token);
          seed.table_position_count=std::max(seed.table_position_count,static_cast<std::uint32_t>(position+1));
        }
        if (selected.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
        if (next_plan==std::numeric_limits<std::uint64_t>::max()) {
          Fail(Status::ResourceExhausted("DeepSeek batch plan identity exhausted")); return;
        }
        const auto plan=next_plan++;
        const auto count=static_cast<std::uint32_t>(selected.size());
        std::vector<DeepSeekRequestIdentity> identities;
        for (const auto& job : selected) identities.push_back({job->id,job->generation,false});
        auto status=engine.prepare_production_pipeline(
          {engine.epoch(),plan,prefill ? DeepSeekPlanPhase::kPrefill : DeepSeekPlanPhase::kDecode,count,count},
          std::move(identities),std::vector<DeepSeekProductionRankPlanSeed>(engine.world_size(),seed),sequences,tokens,129280,sample);
        if (status.ok()) status=engine.commit_pipeline();
        if (!status.ok()) { Fail(status); return; }
        const auto drain_deadline=std::chrono::steady_clock::now()+std::chrono::minutes(5);
        while (engine.execution_live()) {
          if (std::chrono::steady_clock::now()>=drain_deadline) {
            Fail(Status::DeadlineExceeded("DeepSeek batch device drain timed out")); return;
          }
          status=engine.advance_bound_pipeline();
          if (!status.ok() && status.code()!=StatusCode::kUnavailable) { Fail(status); return; }
          if (engine.execution_live()) std::this_thread::yield();
        }
        status=engine.acknowledge_output_plan(plan);
        if (!status.ok()) { Fail(status); return; }
        for (const auto& job : selected) {
          if (prefill) ++job->processed;
          if (!sample) continue;
          auto snapshot=engine.accepted_token_snapshot(job->id,job->generation);
          if (!snapshot.ok()) { Fail(snapshot.status()); return; }
          if (snapshot->accepted_completion_count != job->accepted.accepted_completion_count+1) {
            Fail(Status::Internal("DeepSeek batched token receipt missing")); return;
          }
          job->accepted=std::move(*snapshot);
          status=Interrupted(*job);
          const bool done=job->accepted.finish_reason!=DeepSeekFinishReason::kNone;
          if (done || !status.ok()) {
            auto retired=Retire(job,!status.ok());
            if (!retired.ok()) { Fail(retired); return; }
            active.erase(std::find(active.begin(),active.end(),job));
          }
          if (status.ok()) {
            std::lock_guard lock(job->mutex);
            job->outputs.push_back(job->accepted); job->ready.notify_all();
          }
          if (done || !status.ok()) Complete(job,status);
        }
        if (!active.empty()) std::rotate(active.begin(),active.begin()+1,active.end());
      }
    } catch (...) { Fail(Status::Internal("DeepSeek batch executor failed")); }
  }
};
BatchGenerator::BatchGenerator(std::unique_ptr<BatchEngine> engine,std::uint32_t sequences,std::uint32_t tokens,std::uint32_t pending)
    : engine_(std::move(engine)), impl_(std::make_unique<Impl>(*engine_,sequences,tokens,pending)) {}
BatchGenerator::~BatchGenerator() = default;
std::uint32_t BatchGenerator::maximum_sequences() const noexcept { return impl_->sequences; }
Result<DeepSeekAcceptedTokenSnapshot> BatchGenerator::Generate(
    std::span<const std::uint32_t> prompt,std::uint32_t maximum,std::uint32_t minimum,
    DeepSeekRequestSamplingConfig sampling,std::uint32_t capacity,std::uint64_t id,std::uint64_t generation,
    std::chrono::steady_clock::time_point deadline,const GenerationCallbacks& callbacks) {
  if (prompt.empty() || prompt.size()>capacity || !maximum || maximum>capacity-prompt.size() || minimum>maximum ||
      std::any_of(prompt.begin(),prompt.end(),[](auto value) { return value>=129280; }))
    return Status::InvalidArgument("DeepSeek batched request token bounds invalid");
  auto job=std::make_shared<Impl::Job>();
  job->prompt.assign(prompt.begin(),prompt.end()); job->maximum=maximum; job->minimum=minimum;
  job->capacity=capacity; job->id=id; job->generation=generation; job->sampling=sampling; job->deadline=deadline;
  {
    std::lock_guard lock(impl_->mutex);
    if (impl_->closing) return Status::Unavailable("DeepSeek executor closing");
    if (impl_->queued.size()>=impl_->pending) return Status::ResourceExhausted("DeepSeek pending queue full");
    impl_->queued.push_back(job); impl_->ready.notify_one();
  }
  bool callback_failed=false;
  for (;;) {
    try { if (callbacks.cancelled && callbacks.cancelled()) job->cancelled=true; }
    catch (...) { callback_failed=true; job->cancelled=true; }
    std::unique_lock lock(job->mutex);
    if (!job->done && job->outputs.empty()) job->ready.wait_for(lock,std::chrono::milliseconds(10));
    if (!job->outputs.empty()) {
      auto output=std::move(job->outputs.front()); job->outputs.pop_front(); lock.unlock();
      if (!job->cancelled && std::chrono::steady_clock::now()<deadline) {
        try { if (callbacks.on_token && !callbacks.on_token(output)) job->cancelled=true; }
        catch (...) { callback_failed=true; job->cancelled=true; }
      }
      continue;
    }
    if (job->done) {
      if (callback_failed) return Status::Internal("DeepSeek caller callback failed");
      if (job->cancelled) return Status::Unavailable("DeepSeek generation cancelled");
      if (std::chrono::steady_clock::now()>=deadline) return Status::DeadlineExceeded("DeepSeek generation deadline elapsed");
      if (!job->status.ok()) return job->status;
      return job->accepted;
    }
  }
}
} // namespace pih::deepseek_plugin

