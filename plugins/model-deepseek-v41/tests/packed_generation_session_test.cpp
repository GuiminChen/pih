#include "../packed_generation_session.h"
#include <cstdlib>
using namespace pih::deepseek_v41;
static void Require(bool condition) { if (!condition) std::abort(); }
static PackedGenerationInput Input(unsigned generation, unsigned slot, unsigned processed) {
  PackedGenerationInput input;
  input.sampling.identity = {1, 9, generation, generation + 10};
  input.sampling.processed_length = processed;
  input.cache_slot = slot;
  return input;
}
int main() {
  const std::array<std::uint32_t, 3> prompt{11,12,13};
  auto prefill = Input(4, 1, 3); prefill.first = true; prefill.prompt = prompt;
  auto decode = Input(2, 0, 17); decode.pending_input = 42;
  std::array inputs{prefill, decode};
  auto plan = BuildPackedGenerationPlan(inputs, 2, 4);
  Require(plan.ok());
  Require(plan->spans()[0].identity.sequence_generation == 2);
  Require(plan->spans()[0].position == 16 && plan->spans()[0].token_count == 1);
  Require(plan->spans()[1].position == 0 && plan->spans()[1].token_count == 3);
  Require((std::vector<std::uint32_t>(plan->tokens().begin(),plan->tokens().end()) == std::vector<std::uint32_t>{42,11,12,13}));
  Require(!BuildPackedGenerationPlan(inputs,2,3).ok());
  auto terminal = Input(2,0,17); terminal.finish = true;
  std::array ending{prefill,terminal};
  auto finishing = BuildPackedGenerationPlan(ending,2,3);
  Require(finishing.ok());
  Require(finishing->spans()[0].finish && finishing->spans()[0].token_count == 0);
  Require(finishing->spans()[0].position == 17 && finishing->tokens().size() == 3);
  inputs[1].pending_input.reset(); Require(!BuildPackedGenerationPlan(inputs,2,4).ok());
  inputs[1]=decode; inputs[1].cache_slot=1; Require(!BuildPackedGenerationPlan(inputs,2,4).ok());
  inputs[1]=decode; ++inputs[1].sampling.identity.plan_seq; Require(!BuildPackedGenerationPlan(inputs,2,4).ok());
  prefill.sampling.processed_length=4;
  Require(!BuildPackedGenerationPlan({&prefill,1},2,4).ok());
}
