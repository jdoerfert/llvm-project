//===- ArgumentPromotion.h - Promote by-reference arguments -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_ARGUMENTPROMOTION_H
#define LLVM_TRANSFORMS_IPO_ARGUMENTPROMOTION_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/CallSite.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
class AAResults;
class TargetTransformInfo;

struct ArgumentPromoter {

  template <typename T> using GetterTy = std::function<T &(Function &F)>;
  using AARGetterTy = GetterTy<AAResults>;
  using TTIGetterTy = GetterTy<const TargetTransformInfo>;

  using CallSiteReplacerTy = std::function<void(CallSite, CallSite)>;
  using FunctionReplacerTy = std::function<void(Function&, Function&)>;

  // TODO: Eliminate the function and call site replacer once the old PM is gone
  //       and there is only one way of updating the call graph.
  ArgumentPromoter(unsigned MaxElements, AARGetterTy &AARGetter,
                   TTIGetterTy &TTIGetter, FunctionReplacerTy &FunctionReplacer,
                   CallSiteReplacerTy *CallSiteReplacer)
      : MaxElements(MaxElements), AARGetter(AARGetter), TTIGetter(TTIGetter),
        FunctionReplacer(FunctionReplacer), CallSiteReplacer(CallSiteReplacer) {}

  void analyze(Function &F);
  void analyze(Module &M) {
    for (Function &F : M)
      analyze(F);
  }

  bool promoteArguments();

  struct LoadAtOffset {
    LoadInst *L;
    int64_t Offset;
  };
  struct OffsetAndLength {
    int64_t Offset;
    uint64_t Length;
  };
  struct ArgumentPromotionInfo {
    SmallVector<OffsetAndLength, 2> LoadableRange;
    SmallVector<LoadAtOffset, 2> Loads;
    bool KeepArgument;
  };
  struct FunctionPromotionInfo {
    SmallVector<ArgumentPromotionInfo, 8> ArgInfos;
    SmallPtrSet<FunctionType *, 8> SeenTypes;
  };

private:

  bool allowsArgumentPromotion(Function &F);

  bool canPromoteArgument(Argument &Arg);

  Function *promoteArguments(SmallPtrSetImpl<Argument *> &ArgsToPromote);

  DenseMap<Function *, FunctionPromotionInfo> FunctionPromotionInfoMap;

  void collectDereferenceableOffsets(Function &F, FunctionPromotionInfo &FPI);

  const unsigned MaxElements;
  AARGetterTy &AARGetter;
  TTIGetterTy &TTIGetter;
  FunctionReplacerTy &FunctionReplacer;
  CallSiteReplacerTy *CallSiteReplacer;
};

/// Argument promotion pass.
///
/// This pass walks the functions in each SCC and for each one tries to
/// transform it and all of its callers to replace indirect arguments with
/// direct (by-value) arguments.
class ArgumentPromotionPass : public PassInfoMixin<ArgumentPromotionPass> {
  unsigned MaxElements;

public:
  ArgumentPromotionPass(unsigned MaxElements = 3u) : MaxElements(MaxElements) {}

  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_ARGUMENTPROMOTION_H
