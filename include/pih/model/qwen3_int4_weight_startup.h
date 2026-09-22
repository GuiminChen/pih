#pragma once
#include "pih/model/qwen3_int4_resident_weights.h"
namespace pih {
class QwenInt4StartupClock{public:virtual ~QwenInt4StartupClock()=default;virtual Result<std::uint64_t> now_ns()=0;};
class QwenInt4StartupWaiter{public:virtual ~QwenInt4StartupWaiter()=default;virtual Status wait()=0;};
class QwenInt4WeightStartup final{public:
 static Status Publish(QwenInt4ResidentWeights& resident,TypedCopyDriver& copies,
  CompletionEventDriver& events,CompletionEvidenceProvider& evidence,
  QwenInt4StartupClock& clock,QwenInt4StartupWaiter& waiter);
};
}
