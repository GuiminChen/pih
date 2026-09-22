#include "pih/model/engine_termination_event_gate.h"

#include <gtest/gtest.h>

namespace pih {

TEST(EngineTerminationEventGateTest, ClassifiesFirstAndRepeatedRequests) {
  auto gate = EngineTerminationEventGate::Create(7).value();
  ASSERT_EQ(*gate.accept({7, 1, EngineTerminationEventKind::kSigterm}),
            EngineTerminationEventDisposition::kFirstRequest);
  ASSERT_EQ(*gate.accept({7, 2, EngineTerminationEventKind::kSigint}),
            EngineTerminationEventDisposition::kRepeatedRequest);
  ASSERT_EQ(*gate.accept({7, 3, EngineTerminationEventKind::kSigterm}),
            EngineTerminationEventDisposition::kRepeatedRequest);
}

TEST(EngineTerminationEventGateTest, ReplayAndGenerationDriftPoisonGate) {
  for (int mutation = 0; mutation < 5; ++mutation) {
    auto gate = EngineTerminationEventGate::Create(7).value();
    ASSERT_TRUE(gate.accept({7, 1, EngineTerminationEventKind::kSigterm}).ok());
    EngineTerminationEvent value{7, 2, EngineTerminationEventKind::kSigterm};
    if (mutation == 0) value.engine_generation = 8;
    if (mutation == 1) value.event_identity = 1;
    if (mutation == 2) value.event_identity = 0;
    if (mutation == 3) value.event_identity = 3;
    if (mutation == 4)
      value.kind = static_cast<EngineTerminationEventKind>(99);
    EXPECT_FALSE(gate.accept(value).ok()) << mutation;
    EXPECT_FALSE(gate.accept({7, 3, EngineTerminationEventKind::kSigterm}).ok())
        << mutation;
  }
}

TEST(EngineTerminationEventGateTest, RejectsInvalidGeneration) {
  EXPECT_FALSE(EngineTerminationEventGate::Create(0).ok());
}

}  // namespace pih
