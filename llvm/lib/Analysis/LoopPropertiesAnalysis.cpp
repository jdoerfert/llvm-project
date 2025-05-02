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
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include <string>

using namespace llvm;

static cl::opt<bool> PrintZeroValues(
    "loop-propertiesprint-print-zero-values", cl::Hidden, cl::init(false),
    cl::desc("Whether or not to print non-zero loop property values."));

static cl::opt<bool> PrintAccessedPointers(
    "loop-propertiesprint-print-accessed-pointers", cl::Hidden, cl::init(false),
    cl::desc("Whether or not to print accessed pointers."));

static cl::opt<bool> PrintSingletonPointers(
    "loop-propertiesprint-print-singleton-pointers", cl::Hidden,
    cl::init(false), cl::desc("Whether or not to print singleton pointers."));

static cl::opt<int>
    NumUnrollIterations("loop-propertiesprint-unroll-iterations", cl::Hidden,
                        cl::init(32),
                        cl::desc("Number of iterations that are (virtually) "
                                 "unrolled to detect common pointers."));

LoopPropertiesInfo
LoopPropertiesInfo::get(Loop &L, LoopInfo &LI, ScalarEvolution &SE,
                        TargetTransformInfo *TTI, TargetLibraryInfo *TLI,
                        AAResults *AA, DominatorTree *DT, AssumptionCache *AC) {

  auto &DL = L.getHeader()->getModule()->getDataLayout();

  LoopPropertiesInfo LPI;

  LPI.HasRotatedForm = L.isRotatedForm();

  if (auto *ParentLoop = L.getParentLoop()) {
    LPI.ParentLoop = ParentLoop->getName();
    LPI.IsPerfectlyNested =
        (L.getUniqueExitBlock() == ParentLoop->getHeader() ||
         L.getUniqueExitBlock() == ParentLoop->getUniqueLatchExitBlock()) &&
        all_of(ParentLoop->blocks(), [&](BasicBlock *BB) {
          return L.contains(BB) || ParentLoop->getHeader() == BB ||
                 ParentLoop->getUniqueLatchExitBlock();
        });
  }

  LPI.LoopDepth = L.getLoopDepth();
  LPI.NumInnerLoops = L.getLoopsInPreorder().size() - 1;

  if (BasicBlock *Preheader = L.getLoopPreheader()) {
    LPI.HasLoopPreheader = true;
    LPI.PreheaderBlocksize = Preheader->size();
  }

  SmallVector<BasicBlock *> ExitBBs, ExitingBBs;
  L.getExitingBlocks(ExitingBBs);
  L.getExitBlocks(ExitBBs);

  LPI.NumExitBlocks = ExitBBs.size();
  LPI.NumExitingBlocks = ExitingBBs.size();

  for (auto *ExitBB : ExitBBs) {
    auto *TI = ExitBB->getTerminator();
    LPI.NumEarlyReturnExits += isa<ReturnInst>(TI);
    LPI.NumUnreachableExits += isa<UnreachableInst>(TI);
  }

  std::optional<Loop::LoopBounds> LoopBounds = L.getBounds(SE);
  if (LoopBounds) {
    LPI.BoundsAreSimple = true;
    LPI.IsInitialValueConstant = isa<Constant>(LoopBounds->getInitialIVValue());
    LPI.IsStepConstant = isa<Constant>(LoopBounds->getStepValue());
    LPI.IsFinalValueConstant = isa<Constant>(LoopBounds->getFinalIVValue());
  }

  if (SE.hasLoopInvariantBackedgeTakenCount(&L)) {
    LPI.IsCountableLoop = true;
    const SCEV *BECount = SE.getBackedgeTakenCount(&L);
    if (const SCEVConstant *BEConst = dyn_cast<SCEVConstant>(BECount)) {
      LPI.IsLoopBackEdgeCountConstant = true;
      LPI.LoopBackEdgeCount = BEConst->getAPInt();
    } else {
      SmallPtrSet<const Loop *, 4> UsedLoops;
      SE.getUsedLoops(BECount, UsedLoops);
      if (UsedLoops.empty())
        LPI.IsLoopBackEdgeCountFixed = true;
      else
        LPI.IsLoopBackEdgeCountLoopCarried = true;
    }
  }

  LoopAccessInfo LAI(&L, &SE, TTI, TLI, AA, DT, &LI);
  LPI.HasLoadStoreDepWithInvariantAddr =
      LAI.hasLoadStoreDependenceInvolvingLoopInvariantAddress();
  LPI.HasStoreStoreDepWithInvariantAddr =
      LAI.hasStoreStoreDependenceInvolvingLoopInvariantAddress();
  LPI.HasConvergentOp = LAI.hasConvergentOp();
  LPI.NumRequiredRuntimePointerChecks = LAI.getNumRuntimePointerChecks();
  LPI.CanVectorizeMemory = LAI.canVectorizeMemory();
  auto &DepChecker = LAI.getDepChecker();
  LPI.MaxSaveVectorWidthInBits = DepChecker.getMaxSafeVectorWidthInBits();
  if (auto *Dependences = DepChecker.getDependences())
    for (auto &Dep : *Dependences)
      ++LPI.DependenceInfos[Dep.Type];

  SmallPtrSet<const SCEV *, 8> BasePointers;
  DenseMap<const SCEV *, SmallVector<const SCEV *>> SpacialPtrSCEVs;

  auto *HeaderBB = L.getHeader();
  auto GetChainCost =
      [&](Instruction *InitialI,
          TargetTransformInfo::TargetCostKind CostKind) -> uint64_t {
    SmallVector<Instruction *> Worklist;
    SmallPtrSet<Instruction *, 8> Visited;
    Worklist.push_back(InitialI);
    uint64_t Cost = 0;
    while (!Worklist.empty()) {
      auto *I = Worklist.pop_back_val();
      if (!Visited.insert(I).second)
        continue;
      if (!L.contains(I) || (isa<PHINode>(I) && I->getParent() == HeaderBB))
        continue;
      const auto &ICost = TTI->getInstructionCost(I, CostKind);
      if (!ICost.isValid())
        return -1;
      for (auto *Op : I->operand_values())
        if (auto *OpI = dyn_cast<Instruction>(Op))
          Worklist.push_back(OpI);
      Cost += *ICost.getValue();
    }
    return Cost;
  };

  RecurrenceDescriptor RD;
  for (auto &I : *HeaderBB) {
    auto *PHI = dyn_cast<PHINode>(&I);
    if (!PHI)
      break;
    if (RecurrenceDescriptor::isReductionPHI(PHI, &L, RD, /*DB=*/nullptr, AC,
                                             DT, &SE)) {
      ++LPI.RecurranceInfos[unsigned(RD.getRecurrenceKind())];
      ++LPI.NumReductionPHIs;
    } else {
      ++LPI.NumNonReductionPHIs;
    }
    for (auto *PredBB : predecessors(HeaderBB)) {
      if (!L.contains(PredBB))
        continue;
      if (auto *RecI =
              dyn_cast<Instruction>(PHI->getIncomingValueForBlock(PredBB))) {
        if (!L.contains(RecI))
          continue;
        ++LPI.NumPHIChains;
        if (TTI) {
          LPI.MaxPHIChainLatency =
              std::max(LPI.MaxPHIChainLatency,
                       GetChainCost(RecI, TargetTransformInfo::TCK_Latency));
          LPI.MaxPHIChainRecipThroughput = std::max(
              LPI.MaxPHIChainRecipThroughput,
              GetChainCost(RecI, TargetTransformInfo::TCK_RecipThroughput));
        }
      }
    }
  }

  DenseMap<std::pair<BasicBlock *, Instruction *>, Instruction *>
      LastUserInBlockMap;
  DenseMap<BasicBlock *, SmallPtrSet<Instruction *, 8>> LifeValues;
  DenseMap<Instruction *, unsigned> ScalarLifeMap, VectorLifeMap;

  for (BasicBlock *BB : L.getBlocks()) {
    if (LI.getLoopFor(BB) == &L)
      ++LPI.BasicBlockCount;

    ++LPI.BasicBlockAllCount;
    ++LPI.LoopBlocksizes[BB->size()];

    if (L.isLoopLatch(BB)) {
      ++LPI.LoopLatchCount;
      LPI.ContinueLatchCount += all_of(successors(BB), [&](BasicBlock *SuccBB) {
        return LI.getLoopFor(SuccBB) == &L;
      });
    }

    auto MarkLifeBlocks = [&](BasicBlock *SrcBB, BasicBlock *UseBB,
                              Instruction *I) {
      SmallVector<BasicBlock *> Worklist;
      SmallPtrSet<BasicBlock *, 8> Visited;
      auto AddPredecessors = [&](BasicBlock *BB) {
        for (auto *PredBB : predecessors(BB))
          if (PredBB != SrcBB && !DT->dominates(BB, PredBB))
            Worklist.push_back(PredBB);
      };
      AddPredecessors(UseBB);
      while (!Worklist.empty()) {
        auto *BB = Worklist.pop_back_val();
        if (!Visited.insert(BB).second)
          continue;
        LifeValues[BB].insert(I);
        AddPredecessors(BB);
      }
    };

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

      if (!I.user_empty()) {
        if (I.getType()->isVectorTy())
          ++VectorLifeMap[&I];
        else
          ++ScalarLifeMap[&I];
      }
      DenseMap<PHINode *, unsigned> PHISkipMap;
      for (auto *Usr : I.users()) {
        auto *UsrI = cast<Instruction>(Usr);
        if (auto *PHI = dyn_cast<PHINode>(UsrI)) {
          unsigned Skips = PHISkipMap[PHI];
          ++PHISkipMap[PHI];
          for (unsigned U = 0, E = PHI->getNumIncomingValues(); U < E; U++)
            if (PHI->getIncomingValue(U) == &I && (Skips--) == 0)
              UsrI = PHI->getIncomingBlock(U)->getTerminator();
        }
        auto *UsrBB = UsrI->getParent();
        auto *&LastUserInBlock = LastUserInBlockMap[{UsrBB, &I}];
        if (!LastUserInBlock || DT->dominates(LastUserInBlock, UsrI))
          LastUserInBlock = UsrI;
        if (UsrBB == BB)
          continue;
        MarkLifeBlocks(BB, UsrI->getParent(), &I);
      }
      SmallVector<Instruction *> Worklist;
      append_range(Worklist, map_range(I.operand_values(), [](Value *V) {
                     return dyn_cast<Instruction>(V);
                   }));
      while (!Worklist.empty()) {
        auto *OpI = Worklist.pop_back_val();
        if (!OpI || L.contains(OpI))
          continue;
        if (isa<GetElementPtrInst>(OpI)) {
          append_range(Worklist, map_range(OpI->operand_values(), [](Value *V) {
                         return dyn_cast<Instruction>(V);
                       }));
          continue;
        }
        MarkLifeBlocks(HeaderBB, BB, OpI);
        if (BB != HeaderBB)
          LifeValues[HeaderBB].insert(OpI);
        auto *&LastUserInBlock = LastUserInBlockMap[{BB, OpI}];
        if (!LastUserInBlock || DT->dominates(LastUserInBlock, &I))
          LastUserInBlock = &I;
      }

      unsigned Opcode = I.getOpcode();
      Type *AccessTy = I.getAccessType();
      uint64_t AccessSize = 0;
      Value *Ptr = nullptr;
      if (AccessTy) {
        AccessSize = DL.getTypeAllocSize(AccessTy);
        ++LPI.AccessSizes[AccessSize];
      }
      if (Opcode == Instruction::Load) {
        ++LPI.LoadInstCount;
        LPI.LoadedBytes += AccessSize;
        auto &LI = cast<LoadInst>(I);
        Ptr = LI.getPointerOperand();
        ++LPI.AccessAlignments[LI.getAlign().value()];
      } else if (Opcode == Instruction::Store) {
        ++LPI.StoreInstCount;
        LPI.StoredBytes += AccessSize;
        auto &SI = cast<StoreInst>(I);
        Ptr = SI.getPointerOperand();
        ++LPI.AccessAlignments[SI.getAlign().value()];
      } else if (Opcode == Instruction::AtomicRMW ||
                 Opcode == Instruction::AtomicCmpXchg) {
        ++LPI.StoreInstCount;
        ++LPI.LoadInstCount;
        LPI.StoredBytes += AccessSize;
        LPI.LoadedBytes += AccessSize;
        if (auto *RMWI = dyn_cast<AtomicRMWInst>(&I)) {
          Ptr = RMWI->getPointerOperand();
          ++LPI.AccessAlignments[RMWI->getAlign().value()];
        } else if (auto *CXI = dyn_cast<AtomicCmpXchgInst>(&I)) {
          Ptr = CXI->getPointerOperand();
          ++LPI.AccessAlignments[CXI->getAlign().value()];
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
          if (II->getIntrinsicID() == Intrinsic::fmuladd)
            ++LPI.NumFMA;
        } else if (Function *F = CB->getCalledFunction()) {
          if (F->isDeclaration()) {
            ++LPI.DirectCallDeclCount;
          } else {
            ++LPI.DirectCallDefCount;
          }
          if (CB->mayHaveSideEffects())
            ++LPI.NonPureDirectCallCount;
        } else {
          ++LPI.IndirectCall;
        }
      }

      if (!isAssumeLikeIntrinsic(&I)) {
        ++LPI.InstCount;
      }

      if (I.isAtomic()) {
        ++LPI.AtomicCount;
      }

      if (Ptr) {
        auto *PtrSCEV = SE.getSCEV(Ptr);
        if (!SE.hasComputableLoopEvolution(PtrSCEV, &L)) {
          ++LPI.UnknownPtrStrides;
          ++LPI.UnknownBasePointers;
          continue;
        }
        auto *BasePtrSCEV = SE.getPointerBase(PtrSCEV);
        BasePointers.insert(BasePtrSCEV);
        if (!isa<SCEVCouldNotCompute>(BasePtrSCEV)) {
          SpacialPtrSCEVs[BasePtrSCEV].push_back(SE.removePointerBase(PtrSCEV));
        }

        SmallPtrSet<const Loop *, 4> UsedLoops;
        SE.getUsedLoops(PtrSCEV, UsedLoops);
        if (!UsedLoops.count(&L)) {
          ++LPI.PtrStides[0];
          continue;
        }
        auto [InitSCEV, PostIncSCEV] = SE.SplitIntoInitAndPostInc(&L, PtrSCEV);
        auto *PostIncLoopSCEV = SE.getMinusSCEV(PostIncSCEV, InitSCEV);
        auto [LoopIncSCEV, _] = SE.SplitIntoInitAndPostInc(&L, PostIncLoopSCEV);

        if (auto *StrideConst = dyn_cast<SCEVConstant>(LoopIncSCEV)) {
          auto AccessStride = StrideConst->getAPInt();
          if (AccessStride.isNegative())
            AccessStride.negate();
          AccessStride = AccessStride.udiv(AccessSize);
          ++LPI.PtrStides[AccessStride.getZExtValue()];
        } else
          ++LPI.ComplexPtrStrides;

        if (!NumUnrollIterations)
          continue;

        auto *IterationSCEV = InitSCEV;
        for (auto I = 0; I <= NumUnrollIterations; ++I) {
          auto &Iterations = LPI.TemporalPtrSCEVs[IterationSCEV];
          Iterations.push_back({I, Ptr});
          IterationSCEV = SE.getAddExpr(IterationSCEV, LoopIncSCEV);
        }
      }
    }
  }

  unsigned MaxLifeScalars = 0, MaxLifeVectors = 0;
  ReversePostOrderTraversal<Function *> RPO(HeaderBB->getParent());
  unsigned NumLoopBlocks = 0;
  for (auto *BB : RPO) {
    if (!NumLoopBlocks && BB != HeaderBB)
      continue;
    if (NumLoopBlocks == LPI.BasicBlockAllCount)
      break;
    NumLoopBlocks += L.contains(BB);
    DenseMap<Instruction *, unsigned> ScalarKillMap, VectorKillMap;
    unsigned NumLifeScalars = 0, NumLifeVectors = 0;
    for (auto *I : LifeValues.lookup(BB)) {
      if (I->getType()->isVectorTy())
        ++NumLifeVectors;
      else
        ++NumLifeScalars;
    }
    for (auto &It : LastUserInBlockMap) {
      if (It.first.first != BB ||
          LifeValues.lookup(It.first.first).count(It.first.second))
        continue;
      if (It.first.second->getParent() != BB) {
        if (It.first.second->getType()->isVectorTy())
          ++NumLifeVectors;
        else
          ++NumLifeScalars;
      }
      assert(It.second->getParent() == BB);
      if (It.first.second->getType()->isVectorTy())
        ++VectorKillMap[It.second];
      else
        ++ScalarKillMap[It.second];
    }
    for (auto &I : *BB) {
      NumLifeScalars += ScalarLifeMap.lookup(&I);
      NumLifeVectors += VectorLifeMap.lookup(&I);
      NumLifeScalars -= ScalarKillMap.lookup(&I);
      NumLifeVectors -= VectorKillMap.lookup(&I);
      MaxLifeScalars = std::max(MaxLifeScalars, NumLifeScalars);
      MaxLifeVectors = std::max(MaxLifeVectors, NumLifeVectors);
    }
  }
  LPI.MaxLifeScalars = MaxLifeScalars;
  LPI.MaxLifeVectors = MaxLifeVectors;

  for (auto &It : SpacialPtrSCEVs) {
    for (unsigned I1 = 0, E = It.second.size(); I1 < E; ++I1)
      for (unsigned I2 = I1 + 1; I2 < E; ++I2) {
        auto *S1 = It.second[I1];
        auto *S2 = It.second[I2];
        auto *Diff = SE.getMinusSCEV(S1, S2);
        if (auto *ConstDiff = dyn_cast<SCEVConstant>(Diff)) {
          APInt ConstVal = ConstDiff->getAPInt();
          if (ConstVal.isNegative())
            ConstVal.negate();
          ++LPI.SpacialReuseDistance[ConstVal.getZExtValue()];
        } else {
          SmallPtrSet<const Loop *, 4> UsedLoops;
          SE.getUsedLoops(Diff, UsedLoops);
          if (UsedLoops.empty())
            ++LPI.ParametricSpacialReuseDistance;
          else
            ++LPI.LoopCarriedSpacialReuseDistance;
        }
      }
  }

  for (auto *BasePtrSCEV : BasePointers) {
    auto *BasePtrValueSCEV = dyn_cast<SCEVUnknown>(BasePtrSCEV);
    if (!BasePtrValueSCEV) {
      ++LPI.ComplexBasePointers;
      continue;
    }
    if (isa<GlobalValue>(BasePtrValueSCEV->getValue())) {
      ++LPI.GlobalBasePointers;
      continue;
    }
    if (isa<Argument>(BasePtrValueSCEV->getValue())) {
      ++LPI.ArgumentBasePointers;
      continue;
    }
    if (isa<PHINode>(BasePtrValueSCEV->getValue()) ||
        isa<SelectInst>(BasePtrValueSCEV->getValue())) {
      ++LPI.VariableBasePointers;
      continue;
    }
    auto *BasePtrInst = dyn_cast<Instruction>(BasePtrValueSCEV->getValue());
    if (!BasePtrInst) {
      ++LPI.ComplexBasePointers;
      continue;
    }
    auto *BasePtrLoop = LI.getLoopFor(BasePtrInst->getParent());
    if (!BasePtrLoop) {
      ++LPI.ParameterBasePointers;
      continue;
    }
    if (BasePtrLoop == &L) {
      ++LPI.LoopBasePointers;
      continue;
    }
    ++LPI.OuterLoopBasePointers;
  }

  return LPI;
}

void LoopPropertiesInfo::print(raw_ostream &OS) const {

  OS << "LoopBackEdgeCount (APInt) ";
  LoopBackEdgeCount.print(OS, /*isSigned=*/false);
  OS << "\n";

#define PROPERTY(TY, NAME, DEFAULT)                                            \
  if (PrintZeroValues || (NAME != DEFAULT))                                    \
    OS << #NAME << " (" << #TY << ") " << NAME << "\n";

  PROPERTY(bool, HasRotatedForm, false)
  PROPERTY(bool, HasLoopPreheader, false)
  PROPERTY(bool, IsPerfectlyNested, false)
  PROPERTY(bool, IsCountableLoop, false)
  PROPERTY(bool, IsLoopBackEdgeCountConstant, false)
  PROPERTY(bool, IsLoopBackEdgeCountFixed, false)
  PROPERTY(bool, IsLoopBackEdgeCountLoopCarried, false)
  PROPERTY(bool, BoundsAreSimple, false)
  PROPERTY(bool, IsInitialValueConstant, false)
  PROPERTY(bool, IsStepConstant, false)
  PROPERTY(bool, IsFinalValueConstant, false)
  PROPERTY(bool, HasLoadStoreDepWithInvariantAddr, false)
  PROPERTY(bool, HasStoreStoreDepWithInvariantAddr, false)
  PROPERTY(bool, HasConvergentOp, false)
  PROPERTY(bool, NumRequiredRuntimePointerChecks, false)
  PROPERTY(bool, CanVectorizeMemory, false)
  PROPERTY(uint64_t, MaxSaveVectorWidthInBits, 0)
  PROPERTY(uint64_t, NumReductionPHIs, 0)
  PROPERTY(uint64_t, NumNonReductionPHIs, 0)
  PROPERTY(uint64_t, NumPHIChains, 0)
  PROPERTY(uint64_t, MaxPHIChainLatency, 0)
  PROPERTY(uint64_t, MaxPHIChainRecipThroughput, 0)
  PROPERTY(uint64_t, PreheaderBlocksize, 0)
  PROPERTY(uint64_t, NumExitBlocks, 0)
  PROPERTY(uint64_t, NumExitingBlocks, 0)
  PROPERTY(uint64_t, NumUnreachableExits, 0)
  PROPERTY(uint64_t, NumEarlyReturnExits, 0)
  PROPERTY(uint64_t, BasicBlockAllCount, 0)
  PROPERTY(uint64_t, BasicBlockCount, 0)
  PROPERTY(uint64_t, LoopDepth, 0)
  PROPERTY(uint64_t, NumInnerLoops, 0)
  PROPERTY(uint64_t, LoopLatchCount, 0)
  PROPERTY(uint64_t, ContinueLatchCount, 0)
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
  PROPERTY(uint64_t, NonPureDirectCallCount, 0)
  PROPERTY(uint64_t, IndirectCall, 0)
  PROPERTY(uint64_t, IntrinsicCount, 0)
  PROPERTY(uint64_t, UnknownBasePointers, 0)
  PROPERTY(uint64_t, ComplexBasePointers, 0)
  PROPERTY(uint64_t, GlobalBasePointers, 0)
  PROPERTY(uint64_t, ArgumentBasePointers, 0)
  PROPERTY(uint64_t, VariableBasePointers, 0)
  PROPERTY(uint64_t, ParameterBasePointers, 0)
  PROPERTY(uint64_t, LoopBasePointers, 0)
  PROPERTY(uint64_t, OuterLoopBasePointers, 0)
  PROPERTY(uint64_t, MaxLifeScalars, 0)
  PROPERTY(uint64_t, MaxLifeVectors, 0)
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
  for (auto &It : LoopBlocksizes)
    OS << It.first << ": " << It.second << "\n";

  if (!AccessSizes.empty()) {
    OS << "Access sizes:\n";
    for (auto &It : AccessSizes)
      OS << It.first << ": " << It.second << "\n";
  }

  if (!AccessAlignments.empty()) {
    OS << "Access alignments:\n";
    for (auto &It : AccessAlignments)
      OS << It.first << ": " << It.second << "\n";
  }

  if (PrintAccessedPointers)
    OS << "Accessed pointers (" << TemporalPtrSCEVs.size() << "):\n";

  std::map<unsigned, SmallSet<std::pair<Value *, Value *>, 8>>
      TemporalReuseDistances;
  for (auto &It : TemporalPtrSCEVs) {
    if (!PrintSingletonPointers && It.second.size() <= 1)
      continue;

    SmallVector<std::pair<unsigned, Value *>> Iterations = It.second;
    sort(Iterations);

    for (unsigned I1 = 0, E = Iterations.size(); I1 < E; ++I1)
      for (unsigned I2 = I1 + 1; I2 < E; ++I2) {
        auto Distance = Iterations[I2].first - Iterations[I1].first;
        if (Distance)
          TemporalReuseDistances[Distance].insert(
              {Iterations[I2].second, Iterations[I1].second});
      }

    if (PrintAccessedPointers)
      OS << *It.first << ":  "
         << join(map_range(Iterations,
                           [](auto &It) { return std::to_string(It.first); }),
                 ",")
         << "\n";
  }

  OS << "Temporal reuse distances (" << TemporalReuseDistances.size() << "):\n";
  for (auto &It : TemporalReuseDistances)
    OS << "- " << It.first << " : " << It.second.size() << "\n";

  OS << "Spacial reuse distance (" << SpacialReuseDistance.size() << "):\n";
  for (auto &It : SpacialReuseDistance)
    OS << "- " << It.first << " : " << It.second << "\n";
  if (ParametricSpacialReuseDistance || PrintZeroValues)
    OS << "- <parametric> : " << ParametricSpacialReuseDistance << "\n";
  if (LoopCarriedSpacialReuseDistance || PrintZeroValues)
    OS << "- <carried> : " << LoopCarriedSpacialReuseDistance << "\n";
  if (UnknownSpacialReuseDistance || PrintZeroValues)
    OS << "- <unknown> : " << UnknownSpacialReuseDistance << "\n";

  OS << "Pointer strides (" << PtrStides.size() << "):\n";
  for (auto &It : PtrStides)
    OS << "- " << It.first << " : " << It.second << "\n";
  if (ComplexPtrStrides || PrintZeroValues)
    OS << "- <complex> : " << ComplexPtrStrides << "\n";
  if (UnknownPtrStrides || PrintZeroValues)
    OS << "- <unknown> : " << UnknownPtrStrides << "\n";

  auto RecurKindToString = [](RecurKind RK) {
    switch (RK) {
#define RK_CASE(NAME)                                                          \
  case RecurKind::NAME:                                                        \
    return #NAME;
      RK_CASE(None)
      RK_CASE(Add)
      RK_CASE(Mul)
      RK_CASE(Or)
      RK_CASE(And)
      RK_CASE(Xor)
      RK_CASE(SMin)
      RK_CASE(SMax)
      RK_CASE(UMin)
      RK_CASE(UMax)
      RK_CASE(FAdd)
      RK_CASE(FMul)
      RK_CASE(FMin)
      RK_CASE(FMax)
      RK_CASE(FMinimum)
      RK_CASE(FMaximum)
      RK_CASE(FMulAdd)
      RK_CASE(IAnyOf)
      RK_CASE(FAnyOf)
      RK_CASE(IFindLastIV)
      RK_CASE(FFindLastIV);
#undef RK_CASE
    };
  };

  if (!RecurranceInfos.empty()) {
    OS << "Recurrance info:\n";
    for (auto &It : RecurranceInfos)
      OS << RecurKindToString(RecurKind(It.first)) << ": " << It.second << "\n";
  }

  if (!DependenceInfos.empty()) {
    OS << "Dependence info:\n";
    for (auto &It : DependenceInfos)
      OS << MemoryDepChecker::Dependence::DepName[It.first] << ": " << It.second
         << "\n";
  }
}

AnalysisKey LoopPropertiesAnalysis::Key;

const LoopPropertiesInfo
LoopPropertiesAnalysis::run(Loop &L, LoopAnalysisManager &AM,
                            LoopStandardAnalysisResults &AR) {
  return LoopPropertiesInfo::get(L, AR.LI, AR.SE, &AR.TTI, &AR.TLI, &AR.AA,
                                 &AR.DT, &AR.AC);
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
