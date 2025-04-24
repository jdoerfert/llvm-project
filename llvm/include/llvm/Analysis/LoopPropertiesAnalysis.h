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
#include "llvm/IR/PassManager.h"

#include <map>

namespace llvm {

class LPMUpdater;
class Loop;
class raw_ostream;

class LoopPropertiesInfo {
public:
  static LoopPropertiesInfo get(Loop *L, LoopInfo *LI, ScalarEvolution *SE);

  void print(raw_ostream &OS) const;

  APInt LoopBackEdgeCount;

  /// Loop Block Sizes (block size, loop count)
  /// Ignoring blocks for subloops
  std::map<unsigned, unsigned> LoopBlocksizes;

  bool HasLoopPreheader = false;
  bool IsCountableLoop = false;
  bool IsLoopBackEdgeConstant = false;
  uint64_t PreheaderBlocksize = 0;
  uint64_t BasicBlockAllCount = 0;
  uint64_t BasicBlockCount = 0;
  uint64_t LoopDepth = 0;
  uint64_t NumInnerLoops = 0;
  uint64_t LoopLatchCount = 0;
  uint64_t LoadInstCount = 0;
  uint64_t LoadedBytes = 0;
  uint64_t StoreInstCount = 0;
  uint64_t StoredBytes = 0;
  uint64_t AtomicCount = 0;
  uint64_t FloatArithCount = 0;
  uint64_t IntArithCount = 0;
  uint64_t FloatDivRemCount = 0;
  uint64_t IntDivRemCount = 0;
  uint64_t LogicalInstCount = 0;
  uint64_t ExpensiveCastInstCount = 0;
  uint64_t FreeCastInstCount = 0;
  uint64_t AlmostFreeCastInstCount = 0;
  uint64_t FloatCmpCount = 0;
  uint64_t IntCmpCount = 0;
  uint64_t CondBrCount = 0;
  uint64_t VectorInstCount = 0;
  uint64_t InstCount = 0;
  uint64_t DirectCallDefCount = 0;
  uint64_t DirectCallDeclCount = 0;
  uint64_t IndirectCall = 0;
  uint64_t IntrinsicCount = 0;
};

// Analysis pass
class LoopPropertiesAnalysis
    : public AnalysisInfoMixin<LoopPropertiesAnalysis> {
  friend AnalysisInfoMixin<LoopPropertiesAnalysis>;
  static AnalysisKey Key;

public:
  using Result = const LoopPropertiesInfo;

  LoopPropertiesInfo run(Loop &L, LoopAnalysisManager &AM,
                         LoopStandardAnalysisResults &AR);
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
