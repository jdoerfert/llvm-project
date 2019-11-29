//===- CallGraphUpdater.cpp - A (lazy) call graph update helper -----------===//
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

#include "llvm/Transforms/Utils/CallGraphUpdater.h"

using namespace llvm;

void CallGraphUpdater::reanalyzeFunction(Function &Fn) {
  if (CG) {
    CallGraphNode *OldCGN = (*CG)[&Fn];
    CG->getExternalCallingNode()->removeAnyCallEdgeTo(OldCGN);
    OldCGN->removeAllCalledFunctions();
    CG->addToCallGraph(&Fn);
  } else if (LCG) {
    LazyCallGraph::Node &N = LCG->get(Fn);
    LazyCallGraph::SCC *C = LCG->lookupSCC(N);
    updateCGAndAnalysisManagerForCGSCCPass(*LCG, *C, N, *AM, *UR);
  }
}

void CallGraphUpdater::registerOutlinedFunction(Function &NewFn) {
  if (CG) {
    CG->addToCallGraph(&NewFn);
  } else if (LCG) {
    LazyCallGraph::Node &CCNode = LCG->get(NewFn);
    CCNode.populate();
  }
}

void CallGraphUpdater::removeFunction(Function &Fn) {
  if (CG) {
    CG->removeFunctionFromModule((*CG)[&Fn]);
  } else if (LCG) {
    // TODO
  }
}
