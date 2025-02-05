//===- LoopPropertiesAnalysis.cpp - Function Properties Analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LoopPropertiesInfo and LoopPropertiesAnalysis
// classes used to extract function properties.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopPropertiesAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

LoopPropertiesInfo
LoopPropertiesInfo::getLoopPropertiesInfo(Loop *L, LoopInfo *LI,
                                          ScalarEvolution *SE) {

  LoopPropertiesInfo LPI;

  LPI.IsInnerMostLoop = L->isInnermost();
  LPI.LoopDepth = L->getLoopDepth();

  if (BasicBlock *Preheader = L->getLoopPreheader()) {
    LPI.HasLoopPreheader = true;
    LPI.PreheaderBlocksize = Preheader->size();
  }

  if (SE->hasLoopInvariantBackedgeTakenCount(L)) {
    LPI.IsCountableLoop = true;
    const SCEV *BECount = SE->getBackedgeTakenCount(L);
    if (const SCEVConstant *BEConst = dyn_cast<SCEVConstant>(BECount)) {
      LPI.IsLoopBackEdgeConstant = true;
      LPI.LoopBackEdgeCount = BEConst->getAPInt();
    }
  }

  for (BasicBlock *BB : L->getBlocks()) {
    // Ignore blocks in subloops.
    if (LI->getLoopFor(BB) != L)
      continue;

    ++LPI.BasicBlockCount;
    ++LPI.LoopBlocksizes[BB->size()];

    if (L->isLoopLatch(BB))
      ++LPI.LoopLatchCount;

    for (Instruction &I : *BB) {
      unsigned Opcode = I.getOpcode();
      if (Opcode == Instruction::Load) {
        ++LPI.LoadInstCount;
      } else if (Opcode == Instruction::Store) {
        ++LPI.StoreInstCount;
      } else if (Instruction::Add <= Opcode && Opcode <= Instruction::FRem) {
        ++LPI.BinaryInstCount;
      } else if (Instruction::Shl <= Opcode && Opcode <= Instruction::Xor) {
        ++LPI.LogicalInstCount;
      } else if (Instruction::Trunc <= Opcode &&
                 Opcode <= Instruction::AddrSpaceCast) {
        ++LPI.CastInstCount;
      }
    }
  }

  return LPI;
}

void LoopPropertiesInfo::print(raw_ostream &OS) const {
  OS << "IsInnerMostLoop: " << IsInnerMostLoop << "\n"
     << "LoopDepth: " << LoopDepth << "\n"
     << "HasLoopPreheader: " << HasLoopPreheader << "\n"
     << "PreheaderBlocksize: " << PreheaderBlocksize << "\n"
     << "IsCountableLoop: " << IsCountableLoop << "\n"
     << "IsLoopBackEdgeConstant: " << IsLoopBackEdgeConstant << "\n"
     << "LoopBackEdgeCount: " << LoopBackEdgeCount << "\n"
     << "BasicBlockCount: " << BasicBlockCount << "\n"
     << "LoopBlocksizes: ";
  for (auto Pair : LoopBlocksizes) {
    OS << "{" << Pair.first << ", " << Pair.second << "} ";
  }
  OS << "\n";
  OS << "LoopLatchCount: " << LoopLatchCount << "\n"
     << "LoadInstCount: " << LoadInstCount << "\n"
     << "StoreInstCount: " << StoreInstCount << "\n"
     << "BinaryInstCount: " << BinaryInstCount << "\n"
     << "LogicalInstCount: " << LogicalInstCount << "\n"
     << "CastInstCount: " << CastInstCount << "\n\n";
}

AnalysisKey LoopPropertiesAnalysis::Key;

LoopPropertiesInfo
LoopPropertiesAnalysis::run(Loop &L, LoopAnalysisManager &AM,
                            LoopStandardAnalysisResults &AR) {
  return LoopPropertiesInfo::getLoopPropertiesInfo(&L, &AR.LI, &AR.SE);
}

PreservedAnalyses
LoopPropertiesPrinterPass::run(Loop &L, LoopAnalysisManager &AM,
                               LoopStandardAnalysisResults &AR, LPMUpdater &U) {
  OS << "Printing analysis results for Loop "
     << "'" << L.getName() << "':"
     << "\n";
  AM.getResult<LoopPropertiesAnalysis>(L, AR).print(OS);
  // AM.getResult<IVUsersAnalysis>(L, AR).print(OS);
  // AM.getResult<LoopAccessAnalysis>(*L, LAR);
  return PreservedAnalyses::all();
}
