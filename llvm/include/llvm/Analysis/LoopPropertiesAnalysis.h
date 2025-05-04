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

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/FMF.h"
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
                                const TargetTransformInfo *TTI,
                                TargetLibraryInfo *TLI = nullptr,
                                AAResults *AA = nullptr,
                                DominatorTree *DT = nullptr,
                                AssumptionCache *AC = nullptr);

  void print(raw_ostream &OS) const;

  /// Accessed pointers mapped to (unroll) iteration and original pointer.
  DenseMap<const SCEV *, SmallVector<std::pair<unsigned, Value *>>>
      TemporalPtrSCEVs;

#define MAP_UINT_UINT_PROPERTY(NAME, DEFAULT)                                  \
  std::map<unsigned, unsigned> NAME = DEFAULT;
#define MAP_UINT64_UINT64_PROPERTY(NAME, DEFAULT)                              \
  std::map<uint64_t, uint64_t> NAME = DEFAULT;
#define BOOL_PROPERTY(NAME, DEFAULT) bool NAME = DEFAULT;
#define UINT64_PROPERTY(NAME, DEFAULT) uint64_t NAME = DEFAULT;
#define STRING_PROPERTY(NAME, DEFAULT) std::string NAME = DEFAULT;
#define APINT_PROPERTY(NAME, DEFAULT) APInt NAME = DEFAULT;
#define INSTCOST_PROPERTY(NAME, DEFAULT) InstructionCost NAME = DEFAULT;
#include "LoopProperties.def"
#undef INSTCOST_PROPERTY
#undef BOOL_PROPERTY
#undef UINT64_PROPERTY
#undef STRING_PROPERTY
#undef APINT_PROPERTY
#undef MAP_UINT_UINT_PROPERTY
#undef MAP_UINT64_UINT64_PROPERTY
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
