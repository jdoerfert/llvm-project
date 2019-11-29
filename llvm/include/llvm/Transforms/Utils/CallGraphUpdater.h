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

#ifndef LLVM_CALL_GRAPH_UPDATER_H
#define LLVM_CALL_GRAPH_UPDATER_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/LazyCallGraph.h"

namespace llvm {

/// Wrapper to unify "old style" CallGraph and "new style" LazyCallGraph. This
/// simplifies the interface and the call sites, e.g., new and old pass manager
/// passes can share the same code.
class CallGraphUpdater {
  SmallPtrSet<Function *, 16> ReplacedFunctions;
  SmallVector<Function *, 16> DeadFunctions;
  SmallVector<Function *, 16> DeadFunctionsInComdats;

  /// Old PM variables
  ///{
  CallGraph *CG = nullptr;
  CallGraphSCC *CGSCC = nullptr;
  ///}

  /// New PM variables
  ///{
  LazyCallGraph *LCG = nullptr;
  LazyCallGraph::SCC *SCC = nullptr;
  CGSCCAnalysisManager *AM = nullptr;
  CGSCCUpdateResult *UR = nullptr;
  ///}

public:
  CallGraphUpdater() {}
  ~CallGraphUpdater() { finalize(); }

  /// Initializers for usage outside of a CG-SCC pass, inside a CG-SCC pass in
  /// the old and new pass manager (PM).
  ///{
  void initialize(CallGraph &CG, CallGraphSCC &SCC) {
    this->CG = &CG;
    this->CGSCC = &SCC;
  }
  void initialize(LazyCallGraph &LCG, LazyCallGraph::SCC &SCC,
                  CGSCCAnalysisManager &AM, CGSCCUpdateResult &UR) {
    this->LCG = &LCG;
    this->SCC = &SCC;
    this->AM = &AM;
    this->UR = &UR;
  }
  ///}

  /// Finalizer that will trigger actions like function removal from the CG.
  bool finalize();

  /// Remove \p Fn from the call graph.
  void removeFunction(Function &Fn);

  /// After an CG-SCC pass changes a function in ways that affect the call
  /// graph, this method can be called to update it.
  void reanalyzeFunction(Function &Fn);

  /// If a new function was created by outlining, this method can be called
  /// to update the call graph for the new function. Note that the old one
  /// still needs to be re-analyzed or manually updated.
  void registerOutlinedFunction(Function &NewFn);

  /// Replace \p OldFn in the call graph (and SCC) with \p NewFn. The uses
  /// outside the call graph and the function \p OldFn are not modified.
  void replaceFunctionWith(Function &OldFn, Function &NewFn);

  /// Replace \p OldCS with the new call site \p NewCS.
  /// \return True if the replacement was successful, otherwise False. In the
  /// latter case the parent function of \p OldCB needs to be re-analyzed.
  bool replaceCallSite(CallBase &OldCS, CallBase &NewCS);
};

} // end namespace llvm

#endif // LLVM_CALL_GRAPH_UPDATER_H
