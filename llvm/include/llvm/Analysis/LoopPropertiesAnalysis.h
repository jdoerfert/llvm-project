//=- LoopPropertiesAnalysis.h - Loop Properties Analysis --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LoopPropertiesInfo and LoopPropertiesAnalysis
// classes used to extract loop properties.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOPPROPERTIESANALYSIS_H
#define LLVM_ANALYSIS_LOOPPROPERTIESANALYSIS_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/InstructionCost.h"

#include <map>

namespace llvm {

class LPMUpdater;
class Loop;
class raw_ostream;

class LoopPropertiesInfo {
public:
  static LoopPropertiesInfo get(Loop &L, LoopInfo &LI, ScalarEvolution &SE,
                                TargetTransformInfo *TTI);

  void print(raw_ostream &OS) const;

#define PROPERTY(TY, NAME, DEFAULT) TY NAME = DEFAULT;

  /// Loop block sizes (block size -> count)
  /// Ignoring blocks for subloops
  std::map<unsigned, unsigned> LoopBlocksizes;

  /// Access sizes (access size -> count)
  /// Ignoring blocks for subloops
  std::map<unsigned, unsigned> AccessSizes;

  /// Accessed pointers mapped to (unroll) iteration and original pointer.
  /// Ignoring blocks for subloops
  DenseMap<const SCEV *, SmallVector<std::pair<unsigned, Value *>>> PtrSCEVs;

  /// Instruction costs (cost -> count)
  /// Ignoring blocks for subloops
#define INSTCOST(KIND)                                                         \
  std::map<unsigned, unsigned> InstructionCosts##KIND;                         \
  PROPERTY(InstructionCost, LoopInsts##KIND, InstructionCost())

  INSTCOST(RecipThroughput)
  INSTCOST(Latency)
  INSTCOST(CodeSize)

#undef INSTCOST

  PROPERTY(APInt, LoopBackEdgeCount, APInt())
  PROPERTY(bool, HasLoopPreheader, false)
  PROPERTY(bool, IsCountableLoop, false)
  PROPERTY(bool, IsLoopBackEdgeConstant, false)
  PROPERTY(uint64_t, PreheaderBlocksize, 0)
  PROPERTY(uint64_t, BasicBlockAllCount, 0)
  PROPERTY(uint64_t, BasicBlockCount, 0)
  PROPERTY(uint64_t, LoopDepth, 0)
  PROPERTY(uint64_t, NumInnerLoops, 0)
  PROPERTY(uint64_t, LoopLatchCount, 0)
  PROPERTY(uint64_t, LoadInstCount, 0)
  PROPERTY(uint64_t, LoadedBytes, 0)
  PROPERTY(uint64_t, StoreInstCount, 0)
  PROPERTY(uint64_t, StoredBytes, 0)
  PROPERTY(uint64_t, AtomicCount, 0)
  PROPERTY(uint64_t, FloatArithCount, 0)
  PROPERTY(uint64_t, IntArithCount, 0)
  PROPERTY(uint64_t, FloatDivRemCount, 0)
  PROPERTY(uint64_t, IntDivRemCount, 0)
  PROPERTY(uint64_t, LogicalInstCount, 0)
  PROPERTY(uint64_t, ExpensiveCastInstCount, 0)
  PROPERTY(uint64_t, FreeCastInstCount, 0)
  PROPERTY(uint64_t, AlmostFreeCastInstCount, 0)
  PROPERTY(uint64_t, FloatCmpCount, 0)
  PROPERTY(uint64_t, IntCmpCount, 0)
  PROPERTY(uint64_t, CondBrCount, 0)
  PROPERTY(uint64_t, VectorInstCount, 0)
  PROPERTY(uint64_t, InstCount, 0)
  PROPERTY(uint64_t, DirectCallDefCount, 0)
  PROPERTY(uint64_t, DirectCallDeclCount, 0)
  PROPERTY(uint64_t, IndirectCall, 0)
  PROPERTY(uint64_t, IntrinsicCount, 0)
  PROPERTY(std::string, ParentLoop, "")

#undef PROPERTY
};

// Analysis pass
class LoopPropertiesAnalysis
    : public AnalysisInfoMixin<LoopPropertiesAnalysis> {
  friend AnalysisInfoMixin<LoopPropertiesAnalysis>;
  static AnalysisKey Key;

public:
  using Result = const LoopPropertiesInfo;

  Result run(Loop &L, LoopAnalysisManager &AM, LoopStandardAnalysisResults &AR);
};

/// Printer pass for the LoopPropertiesAnalysis results.
class LoopPropertiesPrinterPass
    : public PassInfoMixin<LoopPropertiesPrinterPass> {
  raw_ostream &OS;

public:
  explicit LoopPropertiesPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};

} // namespace llvm
#endif // LLVM_ANALYSIS_LOOPPROPERTIESANALYSIS_H
