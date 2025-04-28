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
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

using namespace llvm;

static cl::opt<bool> PrintZeroValues(
    "loop-propertiesprint-non-zero-values", cl::Hidden, cl::init(false),
    cl::desc("Whether or not to print non-zero loop property values."));

LoopPropertiesInfo LoopPropertiesInfo::get(Loop &L, LoopInfo &LI,
                                           ScalarEvolution &SE,
                                           TargetTransformInfo *TTI) {

  auto &DL = L.getHeader()->getModule()->getDataLayout();

  LoopPropertiesInfo LPI;

  if (auto *ParentLoop = L.getParentLoop())
    LPI.ParentLoop = ParentLoop->getName();

  LPI.LoopDepth = L.getLoopDepth();
  LPI.NumInnerLoops = L.getLoopsInPreorder().size();

  if (BasicBlock *Preheader = L.getLoopPreheader()) {
    LPI.HasLoopPreheader = true;
    LPI.PreheaderBlocksize = Preheader->size();
  }

  if (SE.hasLoopInvariantBackedgeTakenCount(&L)) {
    LPI.IsCountableLoop = true;
    const SCEV *BECount = SE.getBackedgeTakenCount(&L);
    if (const SCEVConstant *BEConst = dyn_cast<SCEVConstant>(BECount)) {
      LPI.IsLoopBackEdgeConstant = true;
      LPI.LoopBackEdgeCount = BEConst->getAPInt();
    }
  }

  for (BasicBlock *BB : L.getBlocks()) {
    if (LI.getLoopFor(BB) == &L)
      ++LPI.BasicBlockCount;

    ++LPI.BasicBlockAllCount;
    ++LPI.LoopBlocksizes[BB->size()];

    if (L.isLoopLatch(BB))
      ++LPI.LoopLatchCount;

    for (Instruction &I : *BB) {
      if (TTI) {
#define INSTCOST(KIND)                                                         \
  {                                                                            \
    const auto &ICost =                                                        \
        TTI->getInstructionCost(&I, TargetTransformInfo::TCK_##KIND);          \
    LPI.LoopInsts##KIND += ICost;                                              \
    ++LPI.InstructionCosts##KIND[ICost.getValue().value_or(-1)];               \
  }
        INSTCOST(RecipThroughput)
        INSTCOST(Latency)
        INSTCOST(CodeSize)
#undef INSTCOST
      }

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

void LoopPropertiesInfo::print(raw_ostream &OS) const {

#define PROPERTY(TY, NAME, DEFAULT)                                            \
  if (PrintZeroValues || (NAME != DEFAULT))                                    \
    OS << #NAME << " (" << #TY << ") " << NAME << "\n";

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

#define INSTCOST(KIND)                                                         \
  OS << "Loop instruction costs (" #KIND "): ";                                \
  if (LoopInsts##KIND.isValid())                                               \
    OS << *LoopInsts##KIND.getValue() << "\n";                                 \
  else                                                                         \
    OS << "<invalid>\n";                                                       \
  for (auto &It : InstructionCosts##KIND) {                                    \
    OS << It.first << ": " << It.second << "\n";                               \
  }

  INSTCOST(RecipThroughput)
  INSTCOST(Latency)
  INSTCOST(CodeSize)
#undef INSTCOST

  OS << "Block sizes:\n";
  for (auto &It : LoopBlocksizes) {
    OS << It.first << ": " << It.second << "\n";
  }
}

AnalysisKey LoopPropertiesAnalysis::Key;

LoopPropertiesInfo
LoopPropertiesAnalysis::run(Loop &L, LoopAnalysisManager &AM,
                            LoopStandardAnalysisResults &AR) {
  return LoopPropertiesInfo::get(L, AR.LI, AR.SE, &AR.TTI);
}

PreservedAnalyses
LoopPropertiesPrinterPass::run(Loop &L, LoopAnalysisManager &AM,
                               LoopStandardAnalysisResults &AR, LPMUpdater &U) {
  OS << "Printing analysis results for Loop "
     << "'" << L.getName() << "':"
     << "\n";
  AM.getResult<LoopPropertiesAnalysis>(L, AR).print(OS);
  // AM.getResult<IVUsersAnalysis>(L, AR).print(OS);
  // AM.getResult<LoopAccessAnalysis>(L, AR);
  return PreservedAnalyses::all();
}
