#include "../packed_request.h"
#include "../packed_request_ledger.h"
#include "../packed_rank_worker_loop.h"
#include "../kernel_wire.h"
#include <cstdlib>

using namespace pih::deepseek_v41;
static void Require(bool condition) { if (!condition) std::abort(); }
static PackedRequestMember Member(unsigned generation, unsigned slot, unsigned count, unsigned position = 0) {
  PackedRequestMember member;
  member.cache_slot = slot;
  auto& r = member.request;
  r.sampling.identity = {1, 7, generation, generation + 10};
  r.sampling.processed_length = position + count;
  r.count = count;
  r.sampling.parameters.temperature = 0;
  for (unsigned i = 0; i < count; ++i) r.tokens[i] = generation * 10 + i;
  return member;
}
int main() {
  std::array members{Member(4, 3, 1, 20), Member(2, 1, 3), Member(1, 0, 2), Member(3, 2, 1, 9)};
  auto plan = PackedRequest::Create(members, 4, 7);
  Require(plan.ok() && plan->spans().size() == 4 && plan->tokens().size() == 7);
  Require(plan->spans()[0].identity.sequence_generation == 1);
  Require(plan->spans()[1].token_offset == 2 && plan->spans()[1].token_count == 3);
  Require(plan->spans()[2].position == 9 && plan->spans()[3].position == 20);
  Require(plan->tokens()[0] == 10 && plan->tokens()[2] == 20 && plan->tokens()[6] == 40);
  auto head_rows = plan->LastTokenRows(); Require(head_rows.ok());
  Require((*head_rows == std::vector<std::uint32_t>{1, 4, 5, 6}));
  auto wire = plan->Encode(); Require(wire.ok());
  auto received = PackedRequest::Decode(*wire, 4, 7); Require(received.ok());
  Require(std::equal(plan->tokens().begin(), plan->tokens().end(), received->tokens().begin()));
  Require(received->spans()[3].position == 20 && received->spans()[3].cache_slot == 3);
  auto invalid = *wire; invalid[24] = 1;
  Require(!PackedRequest::Decode(invalid, 4, 7).ok());
  Require(!PackedRequest::Decode(*wire, 4, 6).ok());
  Require(!PackedRequest::Decode(*wire, 2, 7).ok());
  std::reverse(members.begin(), members.end());
  auto other_rank = PackedRequest::Create(members, 4, 7);
  Require(other_rank.ok());
  Require(std::equal(plan->tokens().begin(), plan->tokens().end(), other_rank->tokens().begin()));
  Require(!PackedRequest::Create(members, 4, 6).ok());
  Require(!PackedRequest::Create(members, 2, 7).ok());
  members[1].cache_slot = members[0].cache_slot;
  Require(!PackedRequest::Create(members, 4, 7).ok());
  auto single = Member(1, 0, 1);
  Require(PackedRequest::Create({&single, 1}, 1, 1).ok());
  std::array pair{Member(1, 0, 1), Member(2, 1, 1)};
  Require(PackedRequest::Create(pair, 2, 2).ok());
  pair[1].request.sampling.identity.plan_seq++;
  Require(!PackedRequest::Create(pair, 2, 2).ok());
  // Actual worker admission/retirement path: different prompt lengths share
  // one plan, then one member terminates while its peer decodes and a new
  // sequence reuses the released slot with a fresh position/RNG ledger.
  PackedRequestLedger ledger(1, 4);
  auto first_a = Member(1, 0, 2); auto first_b = Member(2, 1, 3);
  first_b.request.sampling.parameters.temperature = 1;
  std::array first_members{first_a, first_b};
  auto first_plan = PackedRequest::Create(first_members, 4, 7); Require(first_plan.ok());
  Require(ledger.Admit(*first_plan).ok()); Require(!ledger.Admit(*first_plan).ok());
  Require(ledger.slot(0).position == 0); // GPU work has not retired.
  Require(ledger.CommitRetired(*first_plan).ok());
  Require(ledger.slot(0).position == 2 && ledger.slot(1).position == 3 && ledger.slot(1).ordinal == 1);
  auto finished = first_a; finished.request.tokens.fill(0); finished.request.count = 0; finished.request.finish = true;
  finished.request.sampling.processed_length = 2; finished.request.sampling.identity.plan_seq = 8;
  auto decode = Member(2, 1, 1, 3); decode.request.sampling.identity.plan_seq = 8;
  decode.request.sampling.parameters.temperature = 1; decode.request.sampling.parameters.ordinal = 1;
  std::array second_members{finished, decode};
  auto second_plan = PackedRequest::Create(second_members, 4, 7); Require(second_plan.ok());
  Require(ledger.Admit(*second_plan).ok()); Require(ledger.CommitRetired(*second_plan).ok());
  Require(ledger.slot(0).identity.sequence_generation == 0 && ledger.slot(1).position == 4 && ledger.slot(1).ordinal == 2);
  auto replacement = Member(3, 0, 2); replacement.request.sampling.identity.plan_seq = 9;
  decode = Member(2, 1, 1, 4); decode.request.sampling.identity.plan_seq = 9;
  decode.request.sampling.parameters.temperature = 1; decode.request.sampling.parameters.ordinal = 2;
  std::array third_members{replacement, decode};
  auto third_plan = PackedRequest::Create(third_members, 4, 7); Require(third_plan.ok());
  Require(ledger.Admit(*third_plan).ok()); Require(ledger.CommitRetired(*third_plan).ok());
  Require(ledger.slot(0).identity.sequence_generation == 3 && ledger.slot(0).position == 2 && ledger.slot(1).position == 5);
  Require(!ledger.Admit(*third_plan).ok()); // Replayed plan cannot consume RNG twice.
  auto stale = Member(3, 0, 1, 2); stale.request.sampling.identity.plan_seq = 10;
  stale.request.sampling.identity.sampling_config_id++;
  auto stale_plan = PackedRequest::Create({&stale, 1}, 4, 7); Require(stale_plan.ok());
  Require(!ledger.Admit(*stale_plan).ok());
  PackedRequestLedger minimum_stop_ledger(1, 1);
  auto suppressed_prefill = Member(5, 0, 2);
  suppressed_prefill.request.sampling.parameters.suppressed_count = 1;
  suppressed_prefill.request.sampling.parameters.suppressed[0] = 1;
  auto suppressed_plan = PackedRequest::Create({&suppressed_prefill, 1}, 1, 2); Require(suppressed_plan.ok());
  Require(minimum_stop_ledger.Admit(*suppressed_plan).ok());
  Require(minimum_stop_ledger.CommitRetired(*suppressed_plan).ok());
  auto final_a = Member(3, 0, 1, 1); final_a.request.tokens.fill(0); final_a.request.count = 0;
  final_a.request.finish = true; final_a.request.sampling.processed_length = 2; final_a.request.sampling.identity.plan_seq = 10;
  auto final_b = Member(2, 1, 1, 4); final_b.request.tokens.fill(0); final_b.request.count = 0;
  final_b.request.finish = true; final_b.request.sampling.processed_length = 5; final_b.request.sampling.identity.plan_seq = 10;
  final_b.request.sampling.parameters.temperature = 1; final_b.request.sampling.parameters.ordinal = 3;
  std::array final_members{final_a, final_b};
  auto final_plan = PackedRequest::Create(final_members, 4, 7); Require(final_plan.ok());
  Require(ledger.Admit(*final_plan).ok()); Require(ledger.CommitRetired(*final_plan).ok()); Require(ledger.idle());
  Require(!PackedRankWorkerLoop::CanRetire(RankWorkerState::kSending, ledger));
  Require(PackedRankWorkerLoop::CanRetire(RankWorkerState::kReceiving, ledger));
  auto prompt_chunk = Member(6, 0, 2);
  auto peer_decode = Member(7, 1, 1, 12);
  std::array continuation_members{prompt_chunk, peer_decode};
  auto continuation_plan = PackedRequest::Create(continuation_members, 2, 3); Require(continuation_plan.ok());
  Require(PackedRankWorkerLoop::NeedsPrefillWorkspace(*continuation_plan));
  prompt_chunk = Member(6, 0, 1, 10);
  std::array decode_members{prompt_chunk, peer_decode};
  auto decode_plan = PackedRequest::Create(decode_members, 2, 2); Require(decode_plan.ok());
  Require(!PackedRankWorkerLoop::NeedsPrefillWorkspace(*decode_plan));
  // Legacy op33 retains its exact word geometry; packed op41 adds row metadata.
  HeadProjectionLaunch scalar_head;
  kernel_wire::Encoder scalar_wire; scalar_wire.One(scalar_head);
  Require(scalar_wire.valid && scalar_wire.command.word_count == 11);
  kernel_wire::PackedHeadLaunch packed_head;
  packed_head.projection.row_indices = {0x1000, 16}; packed_head.projection.sequences = 4; packed_head.projection.input_rows = 7;
  kernel_wire::Encoder packed_wire; packed_wire.One(packed_head);
  Require(packed_wire.valid && packed_wire.command.word_count == 15);
  kernel_wire::PackedHeadLaunch decoded_head;
  kernel_wire::Decoder head_decoder{packed_wire.command}; head_decoder.One(decoded_head);
  Require(head_decoder.valid && head_decoder.cursor == 15 && decoded_head.projection.sequences == 4 &&
      decoded_head.projection.input_rows == 7 && decoded_head.projection.row_indices.address == 0x1000);
}
