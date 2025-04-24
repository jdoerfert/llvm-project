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
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

using namespace llvm;

LoopPropertiesInfo LoopPropertiesInfo::get(Loop *L, LoopInfo *LI,
                                           ScalarEvolution *SE) {

  auto &DL = L->getHeader()->getModule()->getDataLayout();

  LoopPropertiesInfo LPI;

  LPI.LoopDepth = L->getLoopDepth();
  LPI.NumInnerLoops = L->getLoopsInPreorder().size();

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
    if (LI->getLoopFor(BB) == L)
      ++LPI.BasicBlockCount;

    ++LPI.BasicBlockAllCount;
    ++LPI.LoopBlocksizes[BB->size()];

    if (L->isLoopLatch(BB))
      ++LPI.LoopLatchCount;

    for (Instruction &I : *BB) {
      unsigned Opcode = I.getOpcode();
      Type *AccessTy = I.getAccessType();
      uint64_t AccessSize = 0;
      if (AccessTy)
        AccessSize = DL.getTypeAllocSize(AccessTy);
      if (Opcode == Instruction::Load) {
        ++LPI.LoadInstCount;
        LPI.LoadedBytes += AccessSize;
      } else if (Opcode == Instruction::Store) {
        ++LPI.StoreInstCount;
        LPI.StoredBytes += AccessSize;
      } else if (Opcode == Instruction::AtomicRMW ||
                 Opcode == Instruction::AtomicCmpXchg) {
        ++LPI.StoreInstCount;
        ++LPI.LoadInstCount;
        LPI.StoredBytes += AccessSize;
        LPI.LoadedBytes += AccessSize;
        if (Opcode == Instruction::AtomicRMW) {
        }
      } else if (Instruction::Shl <= Opcode && Opcode <= Instruction::Xor) {
        ++LPI.LogicalInstCount;
      } else if (Instruction::GetElementPtr == Opcode) {
        ++LPI.IntArithCount;
      } else if (I.isBinaryOp()) {
        if (I.getType()->isIntOrIntVectorTy()) {
          if (I.isIntDivRem())
            ++LPI.IntDivRemCount;
          else
            ++LPI.IntArithCount;
        } else {
          if (I.isFPDivRem())
            ++LPI.FloatDivRemCount;
          else
            ++LPI.FloatArithCount;
        }
        if (I.getType()->isVectorTy())
          ++LPI.VectorInstCount;
      } else if (Instruction::Trunc <= Opcode && Opcode <= Instruction::SExt) {
        ++LPI.AlmostFreeCastInstCount;
      } else if (Instruction::PtrToInt <= Opcode &&
                 Opcode <= Instruction::AddrSpaceCast) {
        ++LPI.FreeCastInstCount;
      } else if (Instruction::FPToUI <= Opcode &&
                 Opcode <= Instruction::FPExt) {
        ++LPI.ExpensiveCastInstCount;
      } else if (Instruction::FCmp == Opcode) {
        ++LPI.FloatCmpCount;
      } else if (Instruction::ICmp == Opcode) {
        ++LPI.IntCmpCount;
      } else if (auto *Br = dyn_cast<BranchInst>(&I)) {
        if (Br->isConditional())
          ++LPI.CondBrCount;
      } else if (auto *CB = dyn_cast<CallBase>(&I)) {
        if (auto *II = dyn_cast<IntrinsicInst>(CB)) {
          if (!II->isAssumeLikeIntrinsic())
            ++LPI.IntrinsicCount;
        } else if (Function *F = CB->getCalledFunction()) {
          if (F->isDeclaration()) {
            ++LPI.DirectCallDeclCount;
          } else {
            ++LPI.DirectCallDefCount;
          }
        } else {
          ++LPI.IndirectCall;
        }
      }

      if (!I.isDebugOrPseudoInst() && !I.isLifetimeStartOrEnd()) {
        ++LPI.InstCount;
      }

      if (I.isAtomic()) {
        ++LPI.AtomicCount;
      }
    }
  }

  return LPI;
}

void LoopPropertiesInfo::print(raw_ostream &OS) const {}

AnalysisKey LoopPropertiesAnalysis::Key;

LoopPropertiesInfo
LoopPropertiesAnalysis::run(Loop &L, LoopAnalysisManager &AM,
                            LoopStandardAnalysisResults &AR) {
  return LoopPropertiesInfo::get(&L, &AR.LI, &AR.SE);
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
