#ifndef LLVM_ANALYSIS_UNROLLADVISOR_H
#define LLVM_ANALYSIS_UNROLLADVISOR_H

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

#include <cassert>
#include <memory>
#include <optional>

namespace llvm {
class BasicBlock;
class CallBase;
class Function;
class Module;
class OptimizationRemark;
class OptimizationRemarkEmitter;
class UnrollCostEstimator;

enum class UnrollAdvisorMode : int { Default, Release, Development };

struct UnrollAdviceInfo {
  const unsigned TripCount;
  const UnrollCostEstimator &UCE;
  TargetTransformInfo::UnrollingPreferences &UP;
  ScalarEvolution &SE;
  LoopInfo &LI;
  Loop &L;
};

class UnrollAdvisor;

class UnrollAdvice {
public:
  UnrollAdvice(UnrollAdvisor *Advisor, std::optional<unsigned> Factor)
      : Advisor(Advisor), Factor(Factor) {}

  UnrollAdvice(UnrollAdvice &&) = delete;
  UnrollAdvice(const UnrollAdvice &) = delete;
  virtual ~UnrollAdvice() {
    assert(Recorded && "UnrollAdvice should have been informed of the "
                       "inliner's decision in all cases");
  }

  struct InstrumentationNames {
    std::string BeginName, EndName;
  };

  using InstrumentationInfo = std::optional<InstrumentationNames>;

  InstrumentationInfo recordUnrolling(const LoopUnrollResult &Result) {
    markRecorded();
    return recordUnrollingImpl();
  }

  InstrumentationInfo
  recordUnsuccessfulUnrolling(const LoopUnrollResult &Result) {
    markRecorded();
    return recordUnsuccessfulUnrollingImpl(Result);
  }

  InstrumentationInfo recordUnattemptedUnrolling() {
    markRecorded();
    return recordUnattemptedUnrollingImpl();
  }

  std::optional<unsigned> getRecommendedUnrollFactor() const { return Factor; }

protected:
  virtual InstrumentationInfo recordUnrollingImpl() { return std::nullopt; }
  virtual InstrumentationInfo
  recordUnsuccessfulUnrollingImpl(const LoopUnrollResult &Result) {
    return std::nullopt;
  }
  virtual InstrumentationInfo recordUnattemptedUnrollingImpl() {
    return std::nullopt;
  }

  UnrollAdvisor *const Advisor;
  const std::optional<unsigned> Factor;

private:
  void markRecorded() {
    assert(!Recorded && "Recording should happen exactly once");
    Recorded = true;
  }
  void recordUnrollStatsIfNeeded();

  bool Recorded = false;
};

/// Interface for deciding whether to inline a call site or not.
class UnrollAdvisor {
public:
  UnrollAdvisor(UnrollAdvisor &&) = delete;
  virtual ~UnrollAdvisor() {}

  std::unique_ptr<UnrollAdvice> getAdvice(UnrollAdviceInfo UAI);

protected:
  UnrollAdvisor() {}
  virtual std::unique_ptr<UnrollAdvice> getAdviceImpl(UnrollAdviceInfo UAI) = 0;

private:
  friend class UnrollAdvice;
};

UnrollAdvisor &getUnrollAdvisor();

std::unique_ptr<UnrollAdvisor> getDefaultModeUnrollAdvisor();
std::unique_ptr<UnrollAdvisor> getDevelopmentModeUnrollAdvisor();

/// The default heuristic
std::optional<unsigned>
shouldPartialUnroll(const unsigned LoopSize, const unsigned TripCount,
                    const UnrollCostEstimator UCE,
                    const TargetTransformInfo::UnrollingPreferences &UP);

} // namespace llvm
#endif // LLVM_ANALYSIS_UNROLLADVISOR_H
