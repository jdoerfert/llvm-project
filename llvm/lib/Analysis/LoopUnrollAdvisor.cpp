//===-- UnrollLoopAdvisor.cpp - Loop unrolling utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopPropertiesAnalysis.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"
#include "llvm/Analysis/ReleaseModeModelRunner.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Analysis/UnrollAdvisor.h"
#include "llvm/Analysis/Utils/TrainingLogger.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <memory>

#define DEBUG_TYPE "loop-unroll-advisor"

using namespace llvm;

static cl::opt<UnrollAdvisorMode> ClUnrollAdvisorMode(
    "mlgo-loop-unroll-advisor-mode", cl::desc("Loop unroll ML mode"), cl::Hidden,
    cl::init(UnrollAdvisorMode::Default),
    cl::values(clEnumValN(UnrollAdvisorMode::Default, "default", ""),
               clEnumValN(UnrollAdvisorMode::Release, "release", ""),
               clEnumValN(UnrollAdvisorMode::Development, "development", "")));

namespace llvm {

std::optional<unsigned>
shouldPartialUnroll(const unsigned LoopSize, const unsigned TripCount,
                    const UnrollCostEstimator UCE,
                    const TargetTransformInfo::UnrollingPreferences &UP) {

  if (!TripCount)
    return std::nullopt;

  if (!UP.Partial) {
    LLVM_DEBUG(dbgs() << "  will not try to unroll partially because "
                      << "-unroll-allow-partial not given\n");
    return 0;
  }

  unsigned count = UP.Count;
  if (count == 0)
    count = TripCount;
  if (UP.PartialThreshold != LoopUnrollNoThreshold) {
    // Reduce unroll count to be modulo of TripCount for partial unrolling.
    if (UCE.getUnrolledLoopSize(UP, count) > UP.PartialThreshold)
      count = (std::max(UP.PartialThreshold, UP.BEInsns + 1) - UP.BEInsns) /
              (LoopSize - UP.BEInsns);
    if (count > UP.MaxCount)
      count = UP.MaxCount;
    while (count != 0 && TripCount % count != 0)
      count--;
    if (UP.AllowRemainder && count <= 1) {
      // If there is no Count that is modulo of TripCount, set Count to
      // largest power-of-two factor that satisfies the threshold limit.
      // As we'll create fixup loop, do the type of unrolling only if
      // remainder loop is allowed.
      count = UP.DefaultUnrollRuntimeCount;
      while (count != 0 &&
             UCE.getUnrolledLoopSize(UP, count) > UP.PartialThreshold)
        count >>= 1;
    }
    if (count < 2) {
      count = 0;
    }
  } else {
    count = TripCount;
  }
  if (count > UP.MaxCount)
    count = UP.MaxCount;

  LLVM_DEBUG(dbgs() << "  partially unrolling with count: " << count << "\n");

  return count;
}

class DefaultUnrollAdvisor : public UnrollAdvisor {
public:
  DefaultUnrollAdvisor() {}
  ~DefaultUnrollAdvisor() {}

protected:
  std::unique_ptr<UnrollAdvice> getAdviceImpl(UnrollAdviceInfo UAI) override {
    return std::make_unique<UnrollAdvice>(
        this,
        shouldPartialUnroll(UAI.UCE.getRolledLoopSize(), UAI.TripCount, UAI.UCE, UAI.UP));
  }
};

std::unique_ptr<UnrollAdvice> UnrollAdvisor::getAdvice(UnrollAdviceInfo UAI) {
  return getAdviceImpl(UAI);
}

std::unique_ptr<UnrollAdvisor> getDefaultModeUnrollAdvisor() {
  return std::make_unique<DefaultUnrollAdvisor>();
}

UnrollAdvisor &getUnrollAdvisor() {
  static std::unique_ptr<UnrollAdvisor> Advisor =
      []() -> std::unique_ptr<UnrollAdvisor> {
    switch (ClUnrollAdvisorMode) {
    case UnrollAdvisorMode::Default:
      return getDefaultModeUnrollAdvisor();
    case UnrollAdvisorMode::Release:
      llvm_unreachable("Release mode for UnrollAdvisor not yet implemented");
    case UnrollAdvisorMode::Development:
      return getDevelopmentModeUnrollAdvisor();
    }
    llvm_unreachable("Unknown mode");
  }();
  return *Advisor;
}

} // namespace llvm
