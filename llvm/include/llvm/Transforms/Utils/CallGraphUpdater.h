//===- CallGraphUpdater.h - A (lazy) call graph update helper ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides interfaces used to manipulate a call graph, regardless
/// if it is a "old style" CallGraph or an "new style" LazyCallGraph.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_UTILS_GENERIC_CALL_GRAPH_H
#define LLVM_ANALYSIS_UTILS_GENERIC_CALL_GRAPH_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/LazyCallGraph.h"

namespace llvm {

/// Wrapper to unify "old style" CallGraph and "new style" LazyCallGraph. This
/// simplifies the interface and the call sites, e.g., new and old pass manager
/// passes can share the same code.
class CallGraphUpdater {
  CallGraph *const CG = nullptr;
  LazyCallGraph *const LCG = nullptr;
  CGSCCAnalysisManager *AM = nullptr;
  CGSCCUpdateResult *UR = nullptr;

public:
  CallGraphUpdater() {}
  CallGraphUpdater(CallGraph &CG) : CG(&CG) {}
  CallGraphUpdater(LazyCallGraph &LCG, CGSCCAnalysisManager &AM,
                   CGSCCUpdateResult &UR)
      : LCG(&LCG), AM(&AM), UR(&UR) {}

  /// Remove \p Fn from the call graph.
  void removeFunction(Function &Fn);

  /// After an CG-SCC pass changes a function in ways that affect the call
  /// graph, this method can be called to update it.
  void reanalyzeFunction(Function &Fn);

  /// If a new function was created by outlining, this method can be called
  /// to update the call graph for the new function. Note that the old one
  /// still needs to be re-analyzed or manually updated.
  void registerOutlinedFunction(Function &NewFn);
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_UTILS_GENERIC_CALL_GRAPH_H
