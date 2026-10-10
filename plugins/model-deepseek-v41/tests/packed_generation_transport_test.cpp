// Portable fake OS boundaries around the real packed session, collectors,
// commit gates, ledgers and output credit queues. No GPU/IPC qualification.
#include "../packed_generation_session.h"
#include "../supervisor_request.h"
#include <bit>
#include <cstdlib>
#include <iostream>
#include <thread>
using namespace pih::deepseek_v41;
namespace {
void Check(bool ok, unsigned line) { if (!ok) { std::cerr << "assertion failed at line " << line << '\n'; std::abort(); } }
#define Require(condition) Check((condition), __LINE__)
struct Frame { SamplingIdentity identity; SamplingParameters parameters; unsigned processed; bool terminal; };
std::array<std::deque<Frame>,8> frames;
std::vector<PackedRequest> history;
bool receive_ready = true, processes_live = true;
unsigned process_checks = 0;
std::array<std::uint8_t,256> Encode(const Frame& frame,unsigned rank,unsigned world) {
  std::array<std::uint8_t,256> out{};
  const auto put=[&](unsigned at,std::uint64_t value,unsigned bytes) {
    for(unsigned i=0;i<bytes;++i) out[at+i]=static_cast<std::uint8_t>(value>>(8*i));
  };
  put(0,0x31524b4e41524950ULL,8); put(8,1,4); put(12,256,4);
  put(16,rank,4); put(20,world,4); put(24,frame.processed,4); put(28,rank+1==world?1:0,4);
  put(32,frame.identity.epoch,8); put(40,frame.identity.plan_seq,8);
  put(48,frame.identity.sequence_generation,8); put(56,frame.identity.sampling_config_id,8);
  put(64,frame.parameters.ordinal,8);
  if(rank+1==world) {
    put(72,42,4); put(76,frame.parameters.temperature ? SamplingPhiloxWord(frame.parameters.seed,frame.parameters.ordinal):0,4);
    put(88,std::bit_cast<std::uint32_t>(frame.parameters.logprobs ? -1.0F : 0.0F),4);
  }
  return out;
}
void Reset() { for(auto& queue:frames) queue.clear(); history.clear(); receive_ready=processes_live=true; process_checks=0; }
}
// Tokenization/authentication is outside this session test. Supply a dense byte
// table through its ordinary immutable adapter, and no regex allocations.
namespace pih::plugin_text {
Tokenizer::Tokenizer(const std::filesystem::path&,TokenizerFamily family):family_(family) {}
}
namespace pih::deepseek_v41 {
Result<std::unique_ptr<V41Tokenizer>> V41Tokenizer::Open(const std::filesystem::path&,std::uint64_t) {
  auto tokenizer=std::unique_ptr<V41Tokenizer>(new V41Tokenizer(plugin_text::Tokenizer("fixture",plugin_text::TokenizerFamily::DeepSeekV41Flash)));
  tokenizer->bytes_.resize(FlashConfig::kVocabularySize,"x");
  for(const auto& bytes:tokenizer->bytes_) tokenizer->views_.push_back(bytes);
  return tokenizer;
}
// The production deployment is the friend that creates request owners. This
// test deployment bypasses only tokenizer authentication, never ledger logic.
class SupervisorDeployment {
 public:
  static std::unique_ptr<SupervisorRequest> Request(unsigned generation,unsigned prompt,unsigned maximum,unsigned slots=2,bool stop_at_candidate=false) {
    auto tokenizer=V41Tokenizer::Open({},0); Require(tokenizer.ok());
    auto request=std::unique_ptr<SupervisorRequest>(new SupervisorRequest);
    request->tokenizer_=std::shared_ptr<const V41Tokenizer>(std::move(*tokenizer));
    request->prompt_.assign(prompt,11);
    request->output_=TokenOutputQueue::Create(slots,slots*sizeof(TokenPublication)).value();
    SamplingParameters parameters; parameters.temperature=1; parameters.seed=generation;
    TokenStopConfig stop; stop.maximum=maximum;
    if(stop_at_candidate) { stop.token_count=1; stop.tokens[0]=42; }
    request->ledger_=TokenLedger::Create({1,0,generation,generation+100},prompt,parameters,std::move(stop),
        maximum*sizeof(AcceptedToken),*request->output_).value();
    return request;
  }
};
RankReceiptChannel::~RankReceiptChannel()=default;
RankReceiptChannel::RankReceiptChannel(RankReceiptChannel&& other) noexcept {
  fd_=other.fd_; peer_pid_=other.peer_pid_; rank_=other.rank_; world_=other.world_; direction_=other.direction_;
  other.fd_=-1;
}
Result<RankReceiptChannel> RankReceiptChannel::Attach(int fd,std::int32_t pid,std::uint32_t,unsigned rank,unsigned world,RankChannelDirection direction) {
  RankReceiptChannel channel; channel.fd_=fd; channel.peer_pid_=pid; channel.rank_=rank; channel.world_=world; channel.direction_=direction;
  return channel;
}
Result<bool> RankReceiptChannel::PollReceive(RankCommitGate& gate,Clock::time_point) {
  if(!receive_ready || frames[rank_].empty()) return false;
  auto frame=frames[rank_].front(); frames[rank_].pop_front();
  if(frame.terminal) return Status::FailedPrecondition("unexpected terminal");
  const auto result=gate.SubmitWire(rank_,Encode(frame,rank_,world_)); if(!result.ok()) return result;
  return true;
}
Result<bool> RankReceiptChannel::PollTerminal(const SamplingIdentity& id,std::uint64_t ordinal,unsigned processed,Clock::time_point) {
  if(!receive_ready || frames[rank_].empty()) return false;
  const auto frame=frames[rank_].front();
  if(!frame.terminal || frame.identity.epoch!=id.epoch || frame.identity.plan_seq!=id.plan_seq ||
      frame.identity.sequence_generation!=id.sequence_generation || frame.identity.sampling_config_id!=id.sampling_config_id ||
      frame.parameters.ordinal!=ordinal || frame.processed!=processed)
    return Status::FailedPrecondition("terminal identity/frontier mismatch");
  frames[rank_].pop_front(); return true;
}
Status RankReceiptChannel::CheckQuiet(RankCommitGate& gate) {
  if(!frames[rank_].empty()) { gate.Fail(); return Status::FailedPrecondition("undrained batch frame at commit"); }
  return Status::Ok();
}
Result<RankRequestChannel> RankRequestChannel::Attach(int fd,std::int32_t pid,std::uint32_t uid,unsigned rank,unsigned world,RankChannelDirection direction) {
  auto channel=RankReceiptChannel::Attach(fd,pid,uid,rank,world,direction); if(!channel.ok()) return channel.status();
  return RankRequestChannel(std::move(*channel));
}
Status RankRequestChannel::QueuePacked(const PackedRequest& plan) {
  if(queued_) return Status::FailedPrecondition("fake request send still queued");
  queued_=true;
  if(rank()==0) history.push_back(plan);
  for(unsigned i=0;i<plan.spans().size();++i) {
    const auto& span=plan.spans()[i];
    frames[rank()].push_back({span.identity,plan.parameters()[i],span.position+span.token_count,span.finish});
  }
  return Status::Ok();
}
Result<bool> RankRequestChannel::PollSendPacked(Clock::time_point) { queued_=false; return true; }
RankProcessWatch::~RankProcessWatch()=default;
RankProcessWatch::RankProcessWatch(RankProcessWatch&& other) noexcept { world_=other.world_; ranks_=other.ranks_; other.world_=0; }
Result<RankProcessWatch> RankProcessWatch::Attach(std::span<const RankProcessBinding> ranks) {
  RankProcessWatch watch; watch.world_=ranks.size(); std::copy(ranks.begin(),ranks.end(),watch.ranks_.begin()); return watch;
}
Status RankProcessWatch::CheckLive() { ++process_checks; return processes_live ? Status::Ok():Status::Unavailable("fake worker exited"); }
}
struct Harness {
  std::vector<RankRequestChannel> requests;
  std::vector<RankReceiptChannel> receipts;
  std::vector<RankRequestChannel*> request_views;
  std::vector<RankReceiptChannel*> receipt_views;
  RankProcessWatch processes;
  std::unique_ptr<PackedGenerationSession> session;
  static RankProcessWatch Bind(unsigned world) {
    std::array<RankProcessBinding,8> bindings{};
    for(unsigned rank=0;rank<world;++rank) bindings[rank]={-1,static_cast<std::int32_t>(100+rank)};
    return RankProcessWatch::Attach({bindings.data(),world}).value();
  }
  Harness(unsigned sequences,unsigned tokens,unsigned world=2):processes(Bind(world)) {
    for(unsigned rank=0;rank<world;++rank) {
      requests.push_back(RankRequestChannel::Attach(rank,100+rank,0,rank,world,RankChannelDirection::kSend).value());
      receipts.push_back(RankReceiptChannel::Attach(rank,100+rank,0,rank,world,RankChannelDirection::kReceive).value());
    }
    for(auto& channel:requests) request_views.push_back(&channel);
    for(auto& channel:receipts) receipt_views.push_back(&channel);
    session=PackedGenerationSession::Create(request_views,receipt_views,processes,1,sequences,tokens,4).value();
  }
  void Poll() { auto status=session->Poll(); if(!status.ok()) std::cerr<<status.message()<<'\n'; Require(status.ok()); }
  unsigned Outputs(bool release=true) {
    unsigned count=0;
    for(;;) {
      auto output=session->TakeOutput(); Require(output.ok()); if(!*output) break;
      auto publication=(**output).request->output().Read((**output).lease); Require(publication.ok());
      Require(publication->record.observation.candidate.token_id==42);
      if(release) Require((**output).request->output().Release((**output).lease).ok());
      ++count;
    }
    return count;
  }
  void Drain() {
    for(unsigned i=0;i<200 && session->HasWork();++i) { Poll(); Outputs(); }
    Require(!session->HasWork());
  }
};
int main() {
  using Clock=PackedGenerationSession::Clock;
  for(unsigned world:{2U,4U,8U}) for(unsigned size:{1U,2U,4U}) {
    Reset(); std::vector<std::unique_ptr<SupervisorRequest>> requests;
    Harness h(size,32,world);
    for(unsigned i=0;i<size;++i) {
      requests.push_back(SupervisorDeployment::Request(size-i,i+1,2));
      Require(h.session->Admit(*requests.back(),Clock::now()+std::chrono::seconds(5)).ok());
    }
    h.Poll(); Require(history.size()==1 && history[0].spans().size()==size);
    Require(history[0].tokens().size()==size*(size+1)/2);
    for(unsigned i=0;i<size;++i) Require(history[0].spans()[i].identity.sequence_generation==i+1);
    h.Drain();
    for(auto& request:requests) Require(request->ledger().records().size()==2 && !request->ledger().failed());
    unsigned completed=0; while(h.session->TakeCompleted().value()) ++completed; Require(completed==size);
    Require(process_checks>size);
  }
  // A new full-budget prefill joins a surviving decode request without starving.
  Reset(); auto first=SupervisorDeployment::Request(1,2,4); auto joined=SupervisorDeployment::Request(2,4,2);
  { Harness h(2,4); Require(h.session->Admit(*first,Clock::now()+std::chrono::seconds(5)).ok());
    h.Poll(); h.Poll(); h.Poll(); Require(h.Outputs()==1);
    Require(h.session->Admit(*joined,Clock::now()+std::chrono::seconds(5)).ok());
    h.Poll(); Require(history.back().spans()[0].identity.sequence_generation==1);
    h.Drain(); bool shared=false; for(const auto& plan:history) shared|=plan.spans().size()==2 && !plan.spans()[0].finish;
    Require(shared); Require(first->ledger().records().size()==4 && joined->ledger().records().size()==2);
  }
  // Full-budget short arrivals cannot monopolize compute while an older decode
  // survives. Terminal-only plans must preserve the next compute priority.
  Reset(); auto survivor=SupervisorDeployment::Request(1,1,8);
  std::vector<std::unique_ptr<SupervisorRequest>> arrivals;
  { Harness h(2,4); Require(h.session->Admit(*survivor,Clock::now()+std::chrono::seconds(5)).ok());
    for(unsigned i=0;i<3;++i) { h.Poll(); h.Outputs(); }
    for(unsigned generation=2;generation<=5;++generation) {
      arrivals.push_back(SupervisorDeployment::Request(generation,4,1));
      Require(h.session->Admit(*arrivals.back(),Clock::now()+std::chrono::seconds(5)).ok());
    }
    bool decode_before_last_prefill=false;
    for(unsigned i=0;i<100 && h.session->HasWork();++i) {
      h.Poll(); h.Outputs();
      if(survivor->ledger().records().size()>=3 && arrivals.back()->ledger().records().empty()) decode_before_last_prefill=true;
    }
    Require(decode_before_last_prefill); Require(!h.session->HasWork());
    Require(survivor->ledger().records().size()==8);
    for(const auto& request:arrivals) Require(request->ledger().records().size()==1);
  }
  // Cancel while the shared plan is in flight: drain it, discard only that
  // request's publication, and keep the peer decoding through terminal receipts.
  Reset(); auto cancelled=SupervisorDeployment::Request(1,1,5); auto peer=SupervisorDeployment::Request(2,2,3);
  { Harness h(2,8); Require(h.session->Admit(*cancelled,Clock::now()+std::chrono::seconds(5)).ok());
    Require(h.session->Admit(*peer,Clock::now()+std::chrono::seconds(5)).ok());
    receive_ready=false; h.Poll(); h.Poll(); h.Poll();
    Require(h.session->Cancel(1).ok()); receive_ready=true; h.Poll(); Require(h.Outputs()==1);
    h.Drain(); Require(cancelled->ledger().records().size()==1 && peer->ledger().records().size()==3);
    bool terminal=false; for(const auto& plan:history) if(plan.spans().size()==1 && plan.spans()[0].finish && plan.spans()[0].identity.sequence_generation==1) terminal=true;
    Require(terminal); auto done=h.session->TakeCompleted().value(); Require(done && done->request==cancelled.get() && !done->status.ok());
  }
  // A request's model stop token completes only that ledger and frees its slot.
  Reset(); auto stopped=SupervisorDeployment::Request(1,1,8,2,true);
  auto continuing=SupervisorDeployment::Request(2,2,4);
  { Harness h(2,8); Require(h.session->Admit(*stopped,Clock::now()+std::chrono::seconds(5)).ok());
    Require(h.session->Admit(*continuing,Clock::now()+std::chrono::seconds(5)).ok()); h.Drain();
    Require(stopped->ledger().finish()==TokenFinish::kStopToken && stopped->ledger().records().size()==1);
    Require(continuing->ledger().finish()==TokenFinish::kLength && continuing->ledger().records().size()==4);
    auto done=h.session->TakeCompleted().value(); Require(done && done->request==stopped.get() && done->status.ok());
  }
  // Exhausted output credit skips only the originating sequence.
  Reset(); auto slow=SupervisorDeployment::Request(1,1,4,1);
  auto fast=SupervisorDeployment::Request(2,1,4);
  { Harness h(2,8); Require(h.session->Admit(*slow,Clock::now()+std::chrono::seconds(5)).ok());
    Require(h.session->Admit(*fast,Clock::now()+std::chrono::seconds(5)).ok());
    for(unsigned i=0;i<3;++i) h.Poll();
    auto held=h.session->TakeOutput().value(); Require(held && held->request==slow.get()); h.Outputs();
    for(unsigned i=0;i<50 && fast->ledger().records().size()<4;++i) { h.Poll(); h.Outputs(); }
    Require(fast->ledger().records().size()==4 && slow->ledger().records().size()==1);
    Require(slow->output().Release(held->lease).ok()); Require(h.session->Cancel(1).ok()); h.Drain();
  }
  // Expiring an in-flight member drains its shared step and preserves its peer.
  Reset(); auto expired=SupervisorDeployment::Request(1,1,4);
  auto timely=SupervisorDeployment::Request(2,1,4);
  { Harness h(2,8); Require(h.session->Admit(*expired,Clock::now()+std::chrono::milliseconds(20)).ok());
    Require(h.session->Admit(*timely,Clock::now()+std::chrono::seconds(5)).ok());
    receive_ready=false; h.Poll(); h.Poll(); std::this_thread::sleep_for(std::chrono::milliseconds(25));
    receive_ready=true; h.Drain(); Require(expired->ledger().records().size()<=1 && timely->ledger().records().size()==4);
    auto done=h.session->TakeCompleted().value(); Require(done && done->request==expired.get() && done->status.code()==pih::StatusCode::kDeadlineExceeded);
  }
  // Admission refuses the seventh member while keeping the six owned requests.
  Reset(); std::vector<std::unique_ptr<SupervisorRequest>> bounded;
  { Harness h(2,8);
    for(unsigned i=1;i<=7;++i) {
      bounded.push_back(SupervisorDeployment::Request(i,1,1));
      auto status=h.session->Admit(*bounded.back(),Clock::now()+std::chrono::seconds(5));
      if(i<=6) Require(status.ok()); else Require(status.code()==pih::StatusCode::kResourceExhausted);
    }
    h.Poll(); Require(history.back().spans().size()==2); h.Drain();
    for(unsigned i=0;i<6;++i) Require(bounded[i]->ledger().records().size()==1);
    Require(bounded.back()->ledger().records().empty());
  }
  // A process fault observed before commit publishes no token for either member.
  Reset(); auto a=SupervisorDeployment::Request(1,1,2); auto b=SupervisorDeployment::Request(2,1,2);
  { Harness h(2,4); Require(h.session->Admit(*a,Clock::now()+std::chrono::seconds(5)).ok());
    Require(h.session->Admit(*b,Clock::now()+std::chrono::seconds(5)).ok()); h.Poll(); h.Poll();
    processes_live=false; Require(!h.session->Poll().ok()); Require(h.Outputs()==0);
    Require(a->ledger().records().empty() && b->ledger().records().empty());
  }
  std::cout<<"packed session plans 2/4, dynamic joins, independent cancellation and process-fault gate passed\n";
}
