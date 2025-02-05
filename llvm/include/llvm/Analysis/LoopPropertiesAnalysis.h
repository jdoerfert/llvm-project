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
  static LoopPropertiesInfo getLoopPropertiesInfo(Loop *L, LoopInfo *LI,
                                                  ScalarEvolution *SE);

  void print(raw_ostream &OS) const;

  /// Is Innermost Loop
  bool IsInnerMostLoop = false;
  uint64_t LoopDepth = 0;

  /// Preheader Block Size (by instructions)
  bool HasLoopPreheader = false;
  uint64_t PreheaderBlocksize = 0;

  /// Is Countable Loop
  bool IsCountableLoop = false;

  /// Loop Backedge Count (if countable)
  bool IsLoopBackEdgeConstant = false;
  APInt LoopBackEdgeCount;

  /// Number of basic blocks
  /// Ignoring blocks for subloops
  uint64_t BasicBlockCount = 0;

  /// Loop Block Sizes (block size, loop count)
  /// Ignoring blocks for subloops
  std::map<unsigned, unsigned> LoopBlocksizes;

  /// Number of loop latches
  uint64_t LoopLatchCount = 0;

  /// Load Instruction Count
  uint64_t LoadInstCount = 0;

  /// Store Instruction Count
  uint64_t StoreInstCount = 0;

  /// Binary instructions Count
  uint64_t BinaryInstCount = 0;

  /// Logical Instruction Count
  uint64_t LogicalInstCount = 0;

  /// Cast Instruction Count
  uint64_t CastInstCount = 0;
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
