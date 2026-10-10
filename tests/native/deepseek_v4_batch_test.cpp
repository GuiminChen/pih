#include "../../plugins/model-deepseek-v4-flash/generation.h"
#include <cstdlib>
#include <atomic>
#include <barrier>
#include <map>
#include <thread>
static void Require(bool condition) { if (!condition) std::abort(); }
using namespace pih;
using namespace pih::deepseek_plugin;
class Simulation final : public BatchEngine {
 public:
  struct Ledger { uint64_t generation,prompt; uint32_t maximum; DeepSeekAcceptedTokenSnapshot accepted; };
  std::map<uint64_t,Ledger> jobs;
  std::vector<DeepSeekRequestIdentity> current;
  DeepSeekProductionRankPlanSeed seed;
  DeepSeekPipelinePlanDescriptor descriptor;
  std::atomic<uint32_t> largest{0}, cancellations{0}, acknowledgements{0};
  std::atomic<bool> permit_advance{true};
  std::atomic<uint32_t> commits{0};
  std::atomic<bool> permit_bind{true};
  Status bind_execution_thread() override {
    while(!permit_bind.load()) std::this_thread::yield(); return Status::Ok();
  }
  bool live=false, sample=false;
  Status submit_request(uint64_t id,uint64_t generation) override { Require(!jobs.contains(id)); jobs[id].generation=generation; return Status::Ok(); }
  Status configure_ledger(uint64_t id,uint64_t generation,uint64_t prompt,uint32_t maximum,uint32_t) override {
    Require(jobs[id].generation==generation); jobs[id].prompt=prompt; jobs[id].maximum=maximum; return Status::Ok();
  }
  Status configure_sampling(uint64_t,uint64_t,DeepSeekRequestSamplingConfig) override { return Status::Ok(); }
  Status cancel_request(uint64_t id,uint64_t generation) override { Require(!live && jobs[id].generation==generation); ++cancellations; return Status::Ok(); }
  Status finish_request(uint64_t id,uint64_t generation) override { Require(!live && jobs[id].generation==generation); return Status::Ok(); }
  Status retire_request(uint64_t id,uint64_t generation) override { Require(!live && jobs[id].generation==generation); jobs.erase(id); return Status::Ok(); }
  Status prepare_production_pipeline(DeepSeekPipelinePlanDescriptor plan,std::vector<DeepSeekRequestIdentity> identities,
      std::vector<DeepSeekProductionRankPlanSeed> seeds,uint32_t sequences,uint32_t tokens,uint32_t vocabulary,bool publish) override {
    Require(!live && plan.sequence_count==identities.size() && plan.token_count==plan.sequence_count);
    Require(plan.sequence_count<=sequences && plan.token_count<=tokens && vocabulary==129280 && seeds.size()==2);
    Require(seeds[0].positions==seeds[1].positions && seeds[0].token_ids==seeds[1].token_ids);
    for(size_t i=0;i<identities.size();++i) Require(jobs[identities[i].request_id].generation==identities[i].request_generation);
    descriptor=plan; current=std::move(identities); seed=seeds[0]; sample=publish;
    largest=std::max(largest.load(),plan.sequence_count); return Status::Ok();
  }
  Status commit_pipeline() override { live=true; ++commits; return Status::Ok(); }
  bool execution_live() const override { return live; }
  Status advance_bound_pipeline() override {
    while (!permit_advance) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    if(sample) for(size_t i=0;i<current.size();++i) {
      auto& ledger=jobs[current[i].request_id]; auto& accepted=ledger.accepted;
      accepted.token_ids.push_back(static_cast<uint32_t>(current[i].request_id+100));
      ++accepted.accepted_completion_count;
      accepted.model_processed_length=seed.positions[i]+1;
      accepted.pending_input_token=accepted.token_ids.back();
      if(accepted.accepted_completion_count==ledger.maximum) accepted.finish_reason=DeepSeekFinishReason::kLength;
    }
    live=false; return Status::Ok();
  }
  Status acknowledge_output_plan(uint64_t sequence) override { Require(!live && descriptor.plan_sequence==sequence); ++acknowledgements; return Status::Ok(); }
  Result<DeepSeekAcceptedTokenSnapshot> accepted_token_snapshot(uint64_t id,uint64_t generation) override { Require(jobs[id].generation==generation); return jobs[id].accepted; }
  void fail_all() override { Require(false); }
  uint64_t epoch() const override { return 1; }
  uint32_t world_size() const override { return 2; }
};
int main() {
  for(const uint32_t batch : {1U,2U,4U}) {
    auto fake=std::make_unique<Simulation>(); auto* state=fake.get();
    state->permit_bind=false;
    BatchGenerator executor(std::move(fake),batch,batch,8);
    // Each caller reaches this barrier only after Generate has enqueued its job.
    // The executor starts after all requests really exist, independent of OS
    // thread scheduling delays. The shared-plan size assertion remains strict.
    std::barrier enqueued(static_cast<ptrdiff_t>(batch),[&]() noexcept { state->permit_bind=true; });
    std::barrier start(static_cast<ptrdiff_t>(batch));
    std::vector<std::thread> callers;
    std::atomic<uint32_t> completed{0};
    for(uint32_t index=0;index<batch;++index) callers.emplace_back([&,index] {
      start.arrive_and_wait();
      const std::vector<uint32_t> prompt(index+2,7);
      const auto caller=std::this_thread::get_id();
      GenerationCallbacks callbacks;
      uint32_t received=0; bool announced=false;
      callbacks.cancelled=[&] { if(!announced) { announced=true; enqueued.arrive_and_wait(); } return false; };
      callbacks.on_token=[&](const DeepSeekAcceptedTokenSnapshot& receipt) {
        Require(std::this_thread::get_id()==caller);
        Require(receipt.token_ids.back()==index+101);
        ++received;
        if(batch>1 && index==0 && received==8) return false;
        if(batch>1 && index==1 && received==1) std::this_thread::sleep_for(std::chrono::milliseconds(120));
        return true;
      };
      auto result=executor.Generate(prompt,32,0,{},128,index+1,index+1,
        std::chrono::steady_clock::now()+std::chrono::seconds(10),callbacks);
      if(batch>1 && index==0) Require(!result.ok() && result.status().code()==StatusCode::kUnavailable);
      else { Require(result.ok() && result->accepted_completion_count==32); ++completed; }
    });
    for(auto& caller:callers) caller.join();
    Require(completed==(batch==1?1:batch-1));
    Require(state->largest==batch);
    Require(state->acknowledgements>0 && state->jobs.empty());
    Require(state->cancellations==(batch==1?0:1));
  }
  {
    auto fake=std::make_unique<Simulation>(); auto* state=fake.get();
    state->permit_advance=false;
    BatchGenerator executor(std::move(fake),1,1,1);
    const std::array<uint32_t,1> prompt{7};
    std::atomic<uint32_t> successful{0}, expired{0};
    auto first=std::thread([&] {
      auto result=executor.Generate(prompt,3,0,{},128,1,1,
          std::chrono::steady_clock::now()+std::chrono::seconds(10),{});
      Require(result.ok()); ++successful;
    });
    while (!state->commits) std::this_thread::yield();
    auto second=std::thread([&] {
      auto result=executor.Generate(prompt,3,0,{},128,2,2,
          std::chrono::steady_clock::now()+std::chrono::milliseconds(20),{});
      Require(!result.ok() && result.status().code()==StatusCode::kDeadlineExceeded); ++expired;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto rejected=executor.Generate(prompt,3,0,{},128,3,3,
        std::chrono::steady_clock::now()+std::chrono::seconds(10),{});
    Require(!rejected.ok() && rejected.status().code()==StatusCode::kResourceExhausted);
    state->permit_advance=true;
    first.join(); second.join();
    Require(successful==1 && expired==1 && state->jobs.empty());
  }
}

