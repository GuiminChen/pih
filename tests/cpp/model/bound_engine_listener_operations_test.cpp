#include "pih/model/bound_engine_listener_operations.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

class ListenerKernelBackend final : public EngineListenerKernelBackend {
 public:
  Result<EngineListenerKernelState> inspect(std::int32_t descriptor) override {
    ++calls;
    last_descriptor = descriptor;
    return result;
  }
  int calls = 0;
  std::int32_t last_descriptor = -1;
  Result<EngineListenerKernelState> result =
      EngineListenerKernelState{true, true, true, 9, 90};
};

class AcceptAuthority final : public EngineListenerAcceptAuthority {
 public:
  Result<bool> enabled(std::uint64_t listener_identity) override {
    ++calls;
    last_identity = listener_identity;
    return result;
  }
  int calls = 0;
  std::uint64_t last_identity = 0;
  Result<bool> result = false;
};

TEST(BoundEngineListenerOperationsTest, JoinsKernelAndAuthorityObservation) {
  ListenerKernelBackend kernel;
  AcceptAuthority authority;
  const std::array bindings{EngineListenerHandleBinding{30, 7}};
  auto operations = BoundEngineListenerOperations::Create(
      bindings, kernel, authority).value();

  auto observation = operations.observe(30);
  ASSERT_TRUE(observation.ok());
  EXPECT_TRUE(observation->socket);
  EXPECT_TRUE(observation->kernel_listening);
  EXPECT_FALSE(observation->accept_authority);
  EXPECT_EQ(kernel.last_descriptor, 7);
  EXPECT_EQ(authority.last_identity, 30U);
}

TEST(BoundEngineListenerOperationsTest, ClosedSocketSkipsAuthority) {
  ListenerKernelBackend kernel;
  AcceptAuthority authority;
  kernel.result = EngineListenerKernelState{};
  const std::array bindings{EngineListenerHandleBinding{30, 7}};
  auto operations = BoundEngineListenerOperations::Create(
      bindings, kernel, authority).value();

  auto observation = operations.observe(30);
  ASSERT_TRUE(observation.ok());
  EXPECT_FALSE(observation->open);
  EXPECT_EQ(authority.calls, 0);
}

TEST(BoundEngineListenerOperationsTest, PreservesEitherObservationFailure) {
  ListenerKernelBackend kernel;
  AcceptAuthority authority;
  const std::array bindings{EngineListenerHandleBinding{30, 7}};
  auto operations = BoundEngineListenerOperations::Create(
      bindings, kernel, authority).value();
  kernel.result = Status::Unavailable("socket stat unavailable");
  EXPECT_EQ(operations.observe(30).status().code(), StatusCode::kUnavailable);
  kernel.result = EngineListenerKernelState{true, true, true, 9, 90};
  authority.result = Status::Unavailable("accept authority unavailable");
  EXPECT_EQ(operations.observe(30).status().code(), StatusCode::kUnavailable);
}

TEST(BoundEngineListenerOperationsTest, RejectsUnknownAndInvalidBindings) {
  ListenerKernelBackend kernel;
  AcceptAuthority authority;
  std::array bindings{EngineListenerHandleBinding{30, 7},
                      EngineListenerHandleBinding{31, 8}};
  auto operations = BoundEngineListenerOperations::Create(
      bindings, kernel, authority).value();
  EXPECT_FALSE(operations.observe(32).ok());
  EXPECT_EQ(kernel.calls, 0);

  bindings[1].listener_identity = 30;
  EXPECT_FALSE(BoundEngineListenerOperations::Create(
                   bindings, kernel, authority).ok());
  bindings = {EngineListenerHandleBinding{30, 7},
              EngineListenerHandleBinding{31, 7}};
  EXPECT_FALSE(BoundEngineListenerOperations::Create(
                   bindings, kernel, authority).ok());
}

}  // namespace
}  // namespace pih
