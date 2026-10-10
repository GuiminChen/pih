#include "../../../plugins/model-qwen3/batch_generation.h"
#include <gtest/gtest.h>
#include <future>
#include <set>
#include <atomic>
#include <algorithm>
#ifndef PIH_QWEN_BATCH_OWNER_ONLY
#include "pih/scheduler/controller_runtime.h"
#endif

namespace pih::qwen_plugin {
namespace {
// A packed backend whose single plan contains one real token from every
// admitted sequence. It exposes the plan membership, not just call counts.
struct PackedEngine {
  std::uint32_t expected;
  std::uint64_t next = 1;
  std::vector<std::uint64_t> admitted, plan;
  std::deque<NvidiaQwenPackedOutputEvent> events;
  std::uint32_t acknowledgements = 0;
  bool executed = false;
  bool reject_first = false;
  bool dynamic = false;
  std::vector<std::vector<std::uint64_t>> plan_history;
  std::atomic<std::uint32_t> submitted_count{0};
  std::set<std::uint64_t> cancelled;
  Result<std::uint64_t> submit_packed(std::span<const std::int64_t>,
      std::uint32_t, const ControllerRequestSampling&) {
    if (reject_first) {
      reject_first = false; emit(ControllerOutputEventKind::kFailed, next);
      ++submitted_count; return next++;
    }
    admitted.push_back(next); emit(ControllerOutputEventKind::kAdmitted, next);
    ++submitted_count; return next++;
  }
  Status cancel_packed(std::uint64_t id) {
    cancelled.insert(id); emit(ControllerOutputEventKind::kDraining, id);
    return Status::Ok();
  }
  void emit(ControllerOutputEventKind kind, std::uint64_t id) {
    NvidiaQwenPackedOutputEvent e{}; e.event.kind = kind;
    e.event.request_generation = id;
    e.event.finish_reason = ControllerFinishReason::kLength;
    events.push_back(e);
  }
  Result<NvidiaQwenPackedStep> drive_packed() {
    if (!executed && admitted.size() == expected) {
      executed = true; plan = admitted; plan_history.push_back(plan);
      for (std::uint32_t i = 0; i < plan.size(); ++i) {
        emit(ControllerOutputEventKind::kTokenCommitted, plan[i]);
        auto& e = events.back().event; e.token_id = 42 + i;
        e.token_ordinal = 1; e.plan_sequence = 1;
        e.plan_event_index = i; e.plan_event_count = expected;
      }
      for (auto id : plan) emit(ControllerOutputEventKind::kDraining, id);
    }
    return NvidiaQwenPackedStep::kIdle;
  }
  Result<std::optional<NvidiaQwenPackedOutputEvent>> try_take_packed_event() {
    if (events.empty()) return std::optional<NvidiaQwenPackedOutputEvent>{};
    auto e = events.front(); events.pop_front();
    return std::optional<NvidiaQwenPackedOutputEvent>{e};
  }
  Status acknowledge_packed_output(std::uint64_t p) {
    if (p != 1) return Status::Internal("wrong plan");
    ++acknowledgements; return Status::Ok();
  }
  Status drain_packed(std::uint64_t id) {
    emit(cancelled.contains(id) ? ControllerOutputEventKind::kCancelled :
        ControllerOutputEventKind::kCompleted, id);
    if (dynamic) {
      admitted.erase(std::find(admitted.begin(), admitted.end(), id));
      if (admitted.empty()) executed = false;
    }
    return Status::Ok();
  }
};
struct VariableEngine : PackedEngine {
  VariableEngine() : PackedEngine{2} {}
  std::map<std::uint64_t, std::uint32_t> maximum, ordinals;
  std::uint64_t plan_sequence = 0;
  Result<std::uint64_t> submit_packed(std::span<const std::int64_t> prompt,
      std::uint32_t limit, const ControllerRequestSampling& sampling) {
    auto id = PackedEngine::submit_packed(prompt, limit, sampling);
    maximum[*id] = limit; ordinals[*id] = 0; return id;
  }
  Result<NvidiaQwenPackedStep> drive_packed() {
    // Ensure both initial requests share their first real plan, and ensure a
    // new request can join while the surviving initial request is decoding.
    if (next < 3 || (maximum.size() == 1 && next == 3)) return NvidiaQwenPackedStep::kIdle;
    std::vector<std::uint64_t> members;
    for (auto& [id, limit] : maximum) if (ordinals[id] < limit) members.push_back(id);
    if (members.empty()) return NvidiaQwenPackedStep::kIdle;
    plan_history.push_back(members); ++plan_sequence;
    for (std::uint32_t i = 0; i < members.size(); ++i) {
      auto id = members[i]; emit(ControllerOutputEventKind::kTokenCommitted, id);
      auto& e = events.back().event; e.token_id = 42 + id;
      e.token_ordinal = ++ordinals[id]; e.plan_sequence = plan_sequence;
      e.plan_event_index = i; e.plan_event_count = static_cast<std::uint32_t>(members.size());
    }
    for (auto id : members) if (ordinals[id] == maximum[id]) emit(ControllerOutputEventKind::kDraining, id);
    return NvidiaQwenPackedStep::kBatchCompleted;
  }
  Status acknowledge_packed_output(std::uint64_t id) {
    if (id != plan_sequence) return Status::Internal("wrong variable plan");
    ++acknowledgements; return Status::Ok();
  }
  Status drain_packed(std::uint64_t id) {
    maximum.erase(id); ordinals.erase(id);
    return PackedEngine::drain_packed(id);
  }
};
TEST(QwenBatchGeneration, RealPlansRouteTokensAndAcknowledgeOnce) {
  for (auto size : {1U, 2U, 4U}) {
    PackedEngine engine{size};
    BatchGeneration owner(engine, size, 8);
    std::vector<std::future<Result<Generation>>> callers;
    for (std::uint32_t i = 0; i < size; ++i) {
      callers.push_back(std::async(std::launch::async, [&] {
        const std::vector<std::int64_t> prompt{1, 2};
        const auto caller = std::this_thread::get_id();
        return owner.Generate(prompt, 1, {},
            std::chrono::steady_clock::now() + std::chrono::seconds(5),
            [caller](std::int64_t) {
              EXPECT_EQ(std::this_thread::get_id(), caller);
              return TokenAction::kContinue;
            }, {});
      }));
    }
    std::set<std::int64_t> tokens;
    for (auto& caller : callers) {
      auto result = caller.get(); ASSERT_TRUE(result.ok());
      ASSERT_EQ(result->tokens.size(), 1); tokens.insert(result->tokens[0]);
      EXPECT_FALSE(result->cancelled);
    }
    EXPECT_EQ(tokens.size(), size);
    EXPECT_EQ(engine.plan.size(), size);
    EXPECT_EQ(engine.acknowledgements, 1);
  }
}
TEST(QwenBatchGeneration, SlowCallbackDoesNotBlockPeerAndStopIsIndependent) {
  PackedEngine engine{2}; BatchGeneration owner(engine, 2, 4);
  std::promise<void> entered, release;
  auto gate = release.get_future().share();
  const std::vector<std::int64_t> prompt{1};
  auto slow = std::async(std::launch::async, [&] {
    return owner.Generate(prompt, 1, {}, std::chrono::steady_clock::now() + std::chrono::seconds(5),
        [&](std::int64_t) { entered.set_value(); gate.wait(); return TokenAction::kStop; }, {});
  });
  auto peer = std::async(std::launch::async, [&] {
    return owner.Generate(prompt, 1, {}, std::chrono::steady_clock::now() + std::chrono::seconds(5), {}, {});
  });
  entered.get_future().wait();
  const auto ready = peer.wait_for(std::chrono::seconds(1));
  release.set_value();
  EXPECT_EQ(ready, std::future_status::ready);
  auto p = peer.get(); auto s = slow.get();
  ASSERT_TRUE(p.ok()); ASSERT_TRUE(s.ok());
  EXPECT_FALSE(p->cancelled); EXPECT_FALSE(p->stopped_by_callback);
  ASSERT_TRUE(s->stopped_by_callback);
}
TEST(QwenBatchGeneration, DeadlineCancelsOnlyItsOwnSequence) {
  PackedEngine engine{99}; BatchGeneration owner(engine, 2, 2);
  const std::vector<std::int64_t> prompt{1};
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(30);
  auto result = owner.Generate(prompt, 1, {}, deadline, {}, {});
  ASSERT_TRUE(result.ok()); ASSERT_TRUE(result->cancelled);
  ASSERT_TRUE(result->deadline_elapsed);
}
// Keep the first sequence live until cancellation while every other sequence
// completes normally. This exercises retirement and reuse under a live peer.
struct IsolatedCancellationEngine : PackedEngine {
  IsolatedCancellationEngine() : PackedEngine{2} { dynamic = true; }
  std::set<std::uint64_t> published;
  std::uint64_t frontier = 0;
  Result<NvidiaQwenPackedStep> drive_packed() {
    for (auto id : admitted) {
      if (id == 1 || published.contains(id)) continue;
      published.insert(id);
      emit(ControllerOutputEventKind::kTokenCommitted, id);
      auto& event = events.back().event;
      event.token_id = 42 + id;
      event.token_ordinal = 1;
      event.plan_sequence = ++frontier;
      event.plan_event_count = 1;
      emit(ControllerOutputEventKind::kDraining, id);
      break;
    }
    return NvidiaQwenPackedStep::kIdle;
  }
  Status acknowledge_packed_output(std::uint64_t sequence) {
    if (sequence != frontier) return Status::Internal("wrong isolated frontier");
    ++acknowledgements;
    return Status::Ok();
  }
};
TEST(QwenBatchGeneration, DeadlineAndCancellationPreservePeerAndRetireState) {
  for (bool expire : {false, true}) {
    IsolatedCancellationEngine engine;
    BatchGeneration owner(engine, 2, 1);
    const std::vector<std::int64_t> prompt{1};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    std::atomic<bool> cancel{false};
    auto victim = std::async(std::launch::async, [&] {
      return owner.Generate(prompt, 1, {}, expire ?
          std::chrono::steady_clock::now() + std::chrono::milliseconds(100) : deadline,
          {}, [&] { return cancel.load(); });
    });
    while (!engine.submitted_count.load() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    auto peer = owner.Generate(prompt, 1, {}, deadline, {}, {});
    cancel = !expire;
    auto cancelled = victim.get();
    ASSERT_TRUE(peer.ok());
    EXPECT_FALSE(peer->cancelled);
    EXPECT_EQ(peer->tokens, (std::vector<std::int64_t>{44}));
    ASSERT_TRUE(cancelled.ok());
    EXPECT_TRUE(cancelled->cancelled);
    EXPECT_EQ(cancelled->deadline_elapsed, expire);
    auto reused = owner.Generate(prompt, 1, {}, deadline, {}, {});
    ASSERT_TRUE(reused.ok());
    EXPECT_FALSE(reused->cancelled);
    EXPECT_EQ(reused->tokens, (std::vector<std::int64_t>{45}));
    EXPECT_EQ(engine.cancelled, (std::set<std::uint64_t>{1}));
    EXPECT_TRUE(engine.admitted.empty());
    EXPECT_TRUE(engine.events.empty());
    EXPECT_EQ(owner.pending_requests(), 0U);
    EXPECT_EQ(engine.acknowledgements, 2U);
  }
}
TEST(QwenBatchGeneration, KvAdmissionRejectDoesNotPoisonEngine) {
  PackedEngine engine{1}; engine.reject_first = true;
  BatchGeneration owner(engine, 2, 2);
  const std::vector<std::int64_t> prompt{1};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  auto rejected = owner.Generate(prompt, 1, {}, deadline, {}, {});
  EXPECT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), StatusCode::kResourceExhausted);
  auto next = owner.Generate(prompt, 1, {}, deadline, {}, {});
  ASSERT_TRUE(next.ok()); EXPECT_FALSE(next->cancelled);
}
TEST(QwenBatchGeneration, BoundedQueueRejectsAndCancellationRetiresIndependently) {
  PackedEngine engine{99}; BatchGeneration owner(engine, 1, 1);
  std::atomic<bool> cancel{false};
  const std::vector<std::int64_t> prompt{1};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  auto request = [&] { return owner.Generate(prompt, 1, {}, deadline, {}, [&] { return cancel.load(); }); };
  auto first = std::async(std::launch::async, request);
  while (engine.submitted_count.load() != 1 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  auto second = std::async(std::launch::async, request);
  while (owner.pending_requests() != 1 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  auto third = request();
  cancel = true;
  EXPECT_FALSE(third.ok()); EXPECT_EQ(third.status().code(), StatusCode::kResourceExhausted);
  auto a = first.get(); auto b = second.get();
  ASSERT_TRUE(a.ok()); ASSERT_TRUE(b.ok());
  ASSERT_TRUE(a->cancelled); ASSERT_TRUE(b->cancelled);
}
TEST(QwenBatchGeneration, CallbackExceptionIsIsolated) {
  PackedEngine engine{2}; BatchGeneration owner(engine, 2, 2);
  const std::vector<std::int64_t> prompt{1};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  auto broken = std::async(std::launch::async, [&] {
    return owner.Generate(prompt, 1, {}, deadline,
        [](std::int64_t) -> TokenAction { throw 1; }, {});
  });
  auto peer = owner.Generate(prompt, 1, {}, deadline, {}, {});
  auto failed = broken.get(); ASSERT_TRUE(peer.ok()); ASSERT_TRUE(failed.ok());
  ASSERT_TRUE(failed->callback_failed); EXPECT_FALSE(peer->cancelled);
}
TEST(QwenBatchGeneration, DynamicAdmissionReusesSlotsAfterCompletion) {
  PackedEngine engine{2}; engine.dynamic = true;
  BatchGeneration owner(engine, 2, 2);
  for (std::uint32_t wave = 0; wave < 3; ++wave) {
    const std::vector<std::int64_t> prompt(wave + 1, 1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    auto a = std::async(std::launch::async, [&] { return owner.Generate(prompt, 1, {}, deadline, {}, {}); });
    auto b = owner.Generate(prompt, 1, {}, deadline, {}, {});
    ASSERT_TRUE(a.get().ok()); ASSERT_TRUE(b.ok());
  }
  EXPECT_EQ(engine.plan_history.size(), 3U);
  EXPECT_EQ(engine.plan_history[0], (std::vector<std::uint64_t>{1, 2}));
  EXPECT_EQ(engine.plan_history[2], (std::vector<std::uint64_t>{5, 6}));
}
TEST(QwenBatchGeneration, NewSequenceJoinsSurvivingVariableLengthDecode) {
  VariableEngine engine; BatchGeneration owner(engine, 2, 4);
  const std::vector<std::int64_t> prompt{1};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  auto short_request = std::async(std::launch::async, [&] { return owner.Generate(prompt, 1, {}, deadline, {}, {}); });
  while (engine.submitted_count.load() != 1) std::this_thread::yield();
  auto long_request = std::async(std::launch::async, [&] { return owner.Generate(prompt, 8, {}, deadline, {}, {}); });
  auto first = short_request.get(); ASSERT_TRUE(first.ok()); EXPECT_EQ(first->tokens.size(), 1U);
  const std::vector<std::int64_t> longer_prompt{2, 3, 4};
  auto joined = owner.Generate(longer_prompt, 3, {}, deadline, {}, {});
  auto survived = long_request.get(); ASSERT_TRUE(joined.ok()); ASSERT_TRUE(survived.ok());
  EXPECT_EQ(joined->tokens.size(), 3U); EXPECT_EQ(survived->tokens.size(), 8U);
  ASSERT_TRUE(std::find(engine.plan_history.begin(), engine.plan_history.end(),
      std::vector<std::uint64_t>{2, 3}) != engine.plan_history.end());
  EXPECT_EQ(engine.acknowledgements, engine.plan_history.size());
}
#ifndef PIH_QWEN_BATCH_OWNER_ONLY
TEST(QwenBatchGeneration, NativeControllerPacksFourVariableLengthRequests) {
  Sha256Digest resource{}; resource.bytes[0] = std::byte{1};
  auto created = ControllerRuntime::Create({1, 4, {{16, 32}, {4, 8, 16}},
      {4, 4, 16, 4, 8, 100}, {4, 16}, 4, false, "qwen-batch-test", resource});
  ASSERT_TRUE(created.ok());
  auto controller = std::move(*created);
  for (std::uint32_t i = 0; i < 4; ++i) {
    std::vector<std::uint32_t> prompt(i + 1, i + 1);
    ASSERT_TRUE(controller.submit_admit(i + 1, prompt, 4 - i).ok());
    ASSERT_TRUE(controller.step(i + 1).ok());
    ASSERT_TRUE(controller.try_take_event().value().has_value());
  }
  auto prepared = controller.prepare_next_plan(10);
  ASSERT_TRUE(prepared.ok()); ASSERT_TRUE(prepared->has_value());
  const auto& plan = *(**prepared).plan;
  EXPECT_EQ(plan.sequence_count(), 4U);
  EXPECT_EQ(plan.total_real_tokens(), 10U);
  EXPECT_EQ(plan.real_token_counts(), (std::vector<std::uint32_t>{1, 2, 3, 4}));
  EXPECT_EQ(plan.ordered_sequence_generations(), (std::vector<std::uint64_t>{1, 2, 3, 4}));
}
#endif
}
}
