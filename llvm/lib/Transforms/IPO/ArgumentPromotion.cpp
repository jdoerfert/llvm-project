//===- ArgumentPromotion.cpp - Promote by-reference arguments -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass promotes "by reference" arguments to be "by value" arguments.  In
// practice, this means looking for internal functions that have pointer
// arguments.  If it can prove, through the use of alias analysis, that an
// argument is *only* loaded, then it can pass the value into the function
// instead of the address of the value.  This can cause recursive simplification
// of code and lead to the elimination of allocas (especially in C++ template
// code like the STL).
//
// This pass also handles aggregate arguments that are passed into a function,
// scalarizing them if the elements of the aggregate are only loaded.  Note that
// by default it refuses to scalarize aggregates which would require passing in
// more than three operands to the function, because passing thousands of
// operands for a large array or structure is unprofitable! This limit can be
// configured or disabled, however.
//
// Note that this transformation could also be done for arguments that are only
// stored to (returning the value instead), but does not currently.  This case
// would be best handled when and if LLVM begins supporting multiple return
// values from functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/ArgumentPromotion.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>

using namespace llvm;

#define DEBUG_TYPE "argpromotion"

STATISTIC(NumArgumentsPromoted, "Number of pointer arguments promoted");
STATISTIC(NumAggregatesPromoted, "Number of aggregate arguments promoted");
STATISTIC(NumByValArgsPromoted, "Number of byval arguments promoted");
STATISTIC(NumArgumentsDead, "Number of dead pointer args eliminated");
STATISTIC(NumDummiesGuardingInAlloca,
          "Number of dummy args introduced to guard 'inalloca'");
STATISTIC(NumArgsSkippedDueToSretAtPosTwo,
          "Number of args skipped due to 'sret' at position two");

/// Checks if all callers of \p F and \p F agree on the ABI rules for \p Arg.
static bool areFunctionArgsABICompatible(const Function &F,
                                         const TargetTransformInfo &TTI,
                                         Argument &Arg) {
  SmallPtrSet<Argument *, 1> Args;
  Args.insert(&Arg);

  for (const Use &U : F.uses()) {
    CallSite CS(U.getUser());
    const Function *Caller = CS.getCaller();
    assert(&F == CS.getCalledFunction());
    if (!TTI.areFunctionArgsABICompatible(Caller, &F, Args))
      return false;
  }
  return true;
}

/// Helper function to create a pointer of type \p ResTy, based on \p Ptr, and
/// advanced by \p Offset bytes. To aid later analysis the method tries to build
/// getelement pointer instructions that traverse the natural type of \p Ptr if
/// possible. If that fails, the remaining offset is adjusted byte-wise, hence
/// through a cast to i8*.
///
/// TODO: This could probably live somewhere more prominantly if it doesn't
///       already exist.
static Value *constructPointer(Type *ResTy, Value *Ptr, int64_t Offset,
                               IRBuilder<NoFolder> &IRB, const DataLayout &DL) {
  assert(Offset >= 0 && "Negative offset not supported yet!");
  LLVM_DEBUG(dbgs() << "Construct pointer: " << *Ptr << " + " << Offset
                    << "-bytes as " << *ResTy << "\n");

  // The initial type we are trying to traverse to get nice GEPs.
  Type *Ty = Ptr->getType();

  SmallVector<Value *, 4> Indices;
  std::string GEPName = Ptr->getName();
  while (Offset) {
    uint64_t Idx, Rem;

    if (auto *STy = dyn_cast<StructType>(Ty)) {
      const StructLayout *SL = DL.getStructLayout(STy);
      if (int64_t(SL->getSizeInBytes()) < Offset)
        break;
      Idx = SL->getElementContainingOffset(Offset);
      assert(Idx < STy->getNumElements() && "Offset calculation error!");
      Rem = Offset - SL->getElementOffset(Idx);
      Ty = STy->getElementType(Idx);
    } else if (auto *PTy = dyn_cast<PointerType>(Ty)) {
      Ty = PTy->getElementType();
      if (!Ty->isSized())
        break;
      uint64_t ElementSize = DL.getTypeAllocSize(Ty);
      assert(ElementSize && "Expected type with size!");
      Idx = Offset / ElementSize;
      Rem = Offset % ElementSize;
    } else {
      // Non-aggregate type, we cast and make byte-wise progress now.
      break;
    }

    LLVM_DEBUG(errs() << "Ty: " << *Ty << " Offset: " << Offset
                      << " Idx: " << Idx << " Rem: " << Rem << "\n");

    GEPName += "." + std::to_string(Idx);
    Indices.push_back(ConstantInt::get(IRB.getInt32Ty(), Idx));
    Offset = Rem;
  }

  // Create a GEP if we collected indices above.
  if (Indices.size())
    Ptr = IRB.CreateGEP(Ptr, Indices, GEPName);

  // If an offset is left we use byte-wise adjustment.
  if (Offset) {
    Ptr = IRB.CreateBitCast(Ptr, IRB.getInt8PtrTy());
    Ptr = IRB.CreateGEP(Ptr, IRB.getInt32(Offset),
                        GEPName + ".b" + Twine(Offset));
  }

  // Ensure the result has the requested type.
  Ptr = IRB.CreateBitOrPointerCast(Ptr, ResTy, Ptr->getName() + ".cast");

  LLVM_DEBUG(dbgs() << "Constructed pointer: " << *Ptr << "\n");
  return Ptr;
}

/// Helper to generate the loads of byval argument members.
///
/// The loads are all based on \p Base which has type pointer to \p Ty. The new
/// loads miror the existing ones stored in \p ExistingLoads and they are stored
/// in \p NewOperands. Two convert the index into the byval argument to a byte
/// offset and result type the two functions \p Index2Offset and \p Index2Type
/// are used.
template <typename Index2OffsetTy, typename Index2TypeTy>
static void generateLoadsOfByvalMembers(
    Value *Base, Type *Ty, IRBuilder<NoFolder> &IRB,
    SmallVectorImpl<Value *> &NewOperands,
    SmallVectorImpl<ArgumentPromoter::LoadAtOffset> &ExistingLoads,
    Index2OffsetTy &&Index2Offset, Index2TypeTy &&Index2Type) {
  Type *I32Ty = IRB.getInt32Ty();
  Value *Indices[2] = {ConstantInt::get(I32Ty, 0), nullptr};
  // Iterate over existing loads and replicate them. Note that we require them
  // to be in increasing order wrt. the index into the byval argument.
  for (unsigned idx = 0, l = 0, e = ExistingLoads.size(); l != e; ++idx) {
    assert(int64_t(Index2Offset(idx)) <= ExistingLoads[l].Offset &&
           "Expected offset to be aligned with an index offset!");
    // Skip elements we do not need.
    if (int64_t(Index2Offset(idx)) != ExistingLoads[l].Offset)
      continue;

    // Advance to the next load offset.
    ++l;
    Indices[1] = ConstantInt::get(I32Ty, idx);
    auto *Ptr =
        IRB.CreateGEP(Ty, Base, Indices, Base->getName() + "." + Twine(idx));
    // TODO: Tell AA about the new values?
    NewOperands.push_back(
        IRB.CreateLoad(Index2Type(idx), Ptr, Ptr->getName() + ".val"));
  }
}

/// Helper to drop all "tail" markers from calls in \p F.
static void dropTailFromCalls(Function &F) {
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<CallInst>(&I))
      CI->setTailCall(false);
}

void ArgumentPromoter::collectDereferenceableOffsets(
    Function &F, FunctionPromotionInfo &FPI) {
  FPI.ArgInfos.resize(F.arg_size());

  const DataLayout &DL = F.getParent()->getDataLayout();
  BasicBlock &EntryBlock = F.getEntryBlock();

  bool NextIsExecuted = true;
  for (Instruction &I : EntryBlock) {

    if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
      if (LI->isSimple()) {
        Value *V = LI->getPointerOperand();
        APInt Offset(DL.getIndexTypeSizeInBits(V->getType()), 0);
        Value *Base = V->stripAndAccumulateConstantOffsets(
            DL, Offset, /* AllowNonInbounds */ true);
        if (Argument *Arg = dyn_cast<Argument>(Base)) {
          int64_t Offset64 = Offset.getSExtValue();
          FPI.ArgInfos[Arg->getArgNo()].LoadableRange.push_back(
              {Offset64, DL.getTypeAllocSize(LI->getType())});
        }
      }
    }

    if (!isGuaranteedToTransferExecutionToSuccessor(&I))
      break;
  }
}

bool ArgumentPromoter::allowsArgumentPromotion(Function &F) {
  // Don't perform argument promotion for naked functions; otherwise we can end
  // up removing parameters that are seemingly 'not used' as they are referred
  // to in the assembly.
  if (F.hasFnAttribute(Attribute::Naked))
    return false;

  // Make sure that it is local to this module.
  if (!F.hasLocalLinkage())
    return false;

  // Don't promote arguments for variadic functions. Adding, removing, or
  // changing non-pack parameters can change the classification of pack
  // parameters. Frontends encode that classification at the call site in the
  // IR, while in the callee the classification is determined dynamically based
  // on the number of registers consumed so far.
  if (F.isVarArg())
    return false;

  // Make sure that all callers are direct callers. We cannot transform
  // functions that have indirect callers.
  for (Use &U : F.uses()) {
    ImmutableCallSite CS(U.getUser());

    // Must be a direct call.
    if (CS.getInstruction() == nullptr || !CS.isCallee(&U))
      return false;

    // Can't change signature of musttail callee
    if (CS.isMustTailCall())
      return false;
  }

  // Can't change signature of musttail caller
  // FIXME: Support promoting whole chain of musttail functions
  for (BasicBlock &BB : F)
    if (BB.getTerminatingMustTailCall())
      return false;

  // Finally, check that we did not already promoted "this function" with the
  // same prototype before. This is a way to prevent us from unpacking recursive
  // types in recursive functions over and over. We basically stop once the
  // function has a type that an old version of it already had.
  FunctionPromotionInfo &FPI = FunctionPromotionInfoMap[&F];
  if (FPI.SeenTypes.count(F.getFunctionType()))
    return false;

  // All checks passed, promotion of arguments is generally allowed. Prepare for
  // promotion by initializing the FunctionPromotionInfo object with loadable
  // access ranges.
  collectDereferenceableOffsets(F, FPI);
  return true;
}

bool ArgumentPromoter::canPromoteArgument(Argument &Arg) {
  LLVM_DEBUG(dbgs() << "Check if argument can be promoted: " << Arg
                    << " [result]\n");

  if (Arg.hasInAllocaAttr()) {
    LLVM_DEBUG(dbgs() << "- inalloca cannot be promoted [no]\n");
    return false;
  }

  Function &F = *Arg.getParent();
  FunctionPromotionInfo &FPI = FunctionPromotionInfoMap[&F];
  assert(FPI.ArgInfos.size() == F.arg_size() &&
         "Expected allowsArgumentPromotion(F) to be called first!");

  // HACK to ensure we do not move 'sret' annotated arguments into a position
  // other than one or two. If this turns out to be important and we care about
  // it, we should do something smart, e.g., add the expanded arguments of the
  // first argument to the end of the argument list. For now, we simply do not
  // expand the first argument if the second is marked with 'sret'.
  if (Arg.getArgNo() == 0 && F.arg_size() > 1 &&
      F.hasParamAttribute(1, Attribute::StructRet)) {
    ++NumArgsSkippedDueToSretAtPosTwo;
    LLVM_DEBUG(
        dbgs() << "- first argument with a second one marked 'sret' [no]\n");
    return false;
  }

  // Get the promotion information struct for this argument.
  ArgumentPromotionInfo &API = FPI.ArgInfos[Arg.getArgNo()];

  // Quick exit for unused arguments. They are not actively promoted but if
  // promotion happens dropped.
  if (Arg.use_empty()) {
    LLVM_DEBUG(dbgs() << "- unused, dropped during promotion [yes]\n");
    return true;
  }

  const DataLayout &DL = F.getParent()->getDataLayout();

  if (Arg.hasByValAttr()) {
    Type *PointeeTy = Arg.getType()->getPointerElementType();
    if (auto *STy = dyn_cast<StructType>(PointeeTy)) {
      // TODO: Check which elements are actually read, consequently needed.
      const StructLayout *SL = DL.getStructLayout(STy);
      unsigned NumElements = STy->getNumElements();
      for (unsigned u = 0; u < NumElements; u++)
        API.Loads.push_back({nullptr, int64_t(SL->getElementOffset(u))});
    } else if (auto *ATy = dyn_cast<ArrayType>(PointeeTy)) {
      // TODO: Check which elements are actually read, consequently needed.
      unsigned NumElements = ATy->getNumElements();
      unsigned ElementSize = DL.getTypeAllocSize(ATy->getElementType());
      for (unsigned u = 0; u < NumElements; u++)
        API.Loads.push_back({nullptr, int64_t(u * ElementSize)});
    } else {
      API.Loads.push_back({nullptr, 0});
    }
    LLVM_DEBUG(dbgs() << "- byval, promoted through privatization [yes]\n");
    return true;
  }

  // Check how many bytes can be dereferenced savely at the function entry.
  int64_t DerefBytes = 0;
  if (Arg.getType()->isPointerTy()) {
    DerefBytes = Arg.getDereferenceableBytes();
    int64_t MinDerefCSBytes = -1;
    for (Use &U : F.uses()) {
      ImmutableCallSite CS(U.getUser());

      bool CanBeNull;
      int64_t DerefCSBytes =
          CS.getArgOperand(Arg.getArgNo())
              ->getPointerDereferenceableBytes(DL, CanBeNull);
      if (CanBeNull)
        DerefCSBytes = 0;
      MinDerefCSBytes = std::min(MinDerefCSBytes, DerefCSBytes);
    }
    DerefBytes = std::max(DerefBytes, MinDerefCSBytes);
  }

  // If we do not know that the pointer is dereferenceable, give up early.
  if (DerefBytes == 0 && API.LoadableRange.empty()) {
    LLVM_DEBUG(dbgs() << "- no known dereferenceable parts [no]\n");
    return false;
  }

  const TargetTransformInfo &TTI = TTIGetter(F);

  // Check early if promotion would be legal.
  if (!areFunctionArgsABICompatible(F, TTI, Arg)) {
    LLVM_DEBUG(dbgs() << "- potential ABI issues [no]\n");
    return false;
  }

  // Keep track of loads we want to preload and their offsets from the argument.
  // Offsets are currently also used as part of the cost heuristic.
  SmallPtrSet<uintptr_t, 16> LoadedOffsets;
  auto CanAndShouldPreloadOffset = [&](Type *Ty, int64_t Offset) -> bool {
    // It seems negative offsets are not worth the hassle but if they are
    // important we need to make the computation in constructPointer aware.
    if (Offset < 0)
      return false;

    uint64_t Length = DL.getTypeAllocSize(Ty);

    // Check that we are in the known dereferenceable range, at least at the
    // function entry. Note that we do not have to check dereferenceability at
    // the load, e.g., we do not mind if there is an intermediate "free" call,
    // because we will move the loads basically to the position we know they are
    // dereferenceable.
    //
    // The check is done in two parts, if either succeeds the load is known
    // dereferenceable at the entry. Part 1 checks the "known dereferenceable
    // bytes", e.g., as annotated through an attribute. Part 2 checks the "known
    // accessed ranges", e.g., ranges loaded from whenever the function is
    // entered.
    bool Dereferenceable =
        (0 <= Offset && Offset + Length <= DerefBytes) ||
        llvm::any_of(API.LoadableRange, [=](OffsetAndLength &OAL) {
          // The loadable range needs to enclose the accessed range.
          return (OAL.Offset <= Offset &&
                  Offset + Length <= OAL.Offset + OAL.Length);
        });
    if (!Dereferenceable)
      return false;

    // TODO: We should coalesce overlapping accesses but for now we do not.
    //       Consequentlye, we can simply track the base offsets and bail if
    //       there are too many different ones.
    if (MaxElements && LoadedOffsets.insert((uintptr_t)Offset).second)
      if (LoadedOffsets.size() >= MaxElements)
        return false;

    // All checks passed, we can and want to preload the given type at the given
    // offset.
    return true;
  };

  // Vector to keep track of uses associated with the relative offset from the
  // base pointer (=argument).
  SmallVector<std::pair<Use *, int64_t>, 16> UseOffsetVector;
  UseOffsetVector.reserve(Arg.getNumUses());

  // All argument uses have offset 0.
  for (Use &U : Arg.uses())
    UseOffsetVector.push_back({&U, 0});

  // Explore transitive uses through known instructions, accumulate offsets as
  // needed, and determine legality of pre-loads.
  while (!UseOffsetVector.empty()) {
    std::pair<Use *, int64_t> UseOffsetPair = UseOffsetVector.pop_back_val();
    Instruction *UInst = cast<Instruction>(UseOffsetPair.first->getUser());
    int64_t Offset = UseOffsetPair.second;

    // Dead instructions might make problems later on, kill them early.
    if (isInstructionTriviallyDead(UInst)) {
      // There may be remaining metadata uses of the user for things like
      // llvm.dbg.value. Replace them with undef.
      UInst->replaceAllUsesWith(UndefValue::get(UInst->getType()));
      UInst->eraseFromParent();
      continue;
    }

    // Look through bitcasts, no change in offset.
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(UInst)) {
      for (Use &U : BCI->uses())
        UseOffsetVector.push_back({&U, Offset});
      continue;
    }

    // Look through GEPs, but only if we can accumulate the offset.
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(UInst)) {
      APInt OffsetAPInt(DL.getIndexTypeSizeInBits(UInst->getType()), Offset);

      // Accumulate the constant offsets, if impossible, bail.
      if (!GEP->accumulateConstantOffset(DL, OffsetAPInt)) {
        LLVM_DEBUG(dbgs() << "- gep with non-constant offsets at " << Offset
                          << ": " << *GEP << " [no]\n");
        return false;
      }

      // Make sure it fits into "int64_t offset".
      if (OffsetAPInt.getMinSignedBits() > 64) {
        LLVM_DEBUG(dbgs() << "- gep with offset that would require >64 bits at "
                          << Offset << ": " << *GEP << " [no]\n");
        return false;
      }

      // Use the accumulated offset for the users of the GEP.
      Offset = OffsetAPInt.getSExtValue();
      for (Use &U : GEP->uses())
        UseOffsetVector.push_back({&U, Offset});

      continue;
    }

    if (LoadInst *LI = dyn_cast<LoadInst>(UInst)) {
      // Don't hack volatile/atomic loads
      if (!LI->isSimple()) {
        LLVM_DEBUG(dbgs() << "- volatile or atomic load at offset " << Offset
                          << ": " << *LI << " [no]\n");
        return false;
      }

      // Check if we can and should preload the load instruction.
      if (!CanAndShouldPreloadOffset(LI->getType(), Offset)) {
        LLVM_DEBUG(dbgs() << "- load that cannot be preloaded at offset "
                          << Offset << ": " << *LI << " [no]\n");
        return false;
      }

      // Remember the load and its relative offset from the base (=argument).
      API.Loads.push_back({LI, Offset});
      continue;
    }

    if (CallBase *CB = dyn_cast<CallBase>(UInst)) {
      Function *Callee = CB->getCalledFunction();
      if (!Callee || !allowsArgumentPromotion(*Callee)) {
        LLVM_DEBUG(dbgs() << "- use in indirect call: " << *CB << " [no]\n");
        return false;
      }
      if (!allowsArgumentPromotion(*Callee)) {
        LLVM_DEBUG(dbgs() << "- use in non-promotable call: " << *CB
                          << " [no]\n");
        return false;
      }

      // TODO: FIXME: implement arg check
      continue;
    }

    // TODO: Handle more than GEP, bitcast, load, and call.

    // Unhandled user, bail.
    LLVM_DEBUG(dbgs() << "- unhandled user: " << *UInst << " [no]\n");
    return false;
  }

  // Check if we found promotable loads, if not, we are done as there is no
  // meaningful user.
  if (API.Loads.empty()) {
    LLVM_DEBUG(
        dbgs() << "- no meaningful users, dropped during promotion [yes]\n");
    Arg.replaceAllUsesWith(UndefValue::get(Arg.getType()));
    return true;
  }

  // Okay, now we know that the argument is only used by load instructions and
  // it is safe to unconditionally perform all of them. Use alias analysis to
  // check to see which loaded offsets are guaranteed to not be modified from
  // entry of the function to the repsective load instructions.

  // If we have enough information we can skip the potentially costly traversal
  // below. E.g., noalias and readonly pointers cannot be modified in this
  // function as we looked at all the users.
  if (!Arg.hasNoAliasAttr() || !Arg.hasAttribute(Attribute::ReadOnly)) {
    // TODO: See if we can do this smarter, maybe not on a per-load-basis.

    // Because there could be several/many load instructions, remember which
    // blocks we know to be transparent to the load.
    df_iterator_default_set<BasicBlock *, 16> TranspBlocks;

    // Collection of loads for which the loaded memory is potentially modified
    // between the entry of the function and the load instruction. We cannot
    // pre-load these.
    SmallVector<LoadInst *, 8> ModifiedLoads;

    AAResults &AAR = AARGetter(F);
    for (LoadAtOffset &LAO : API.Loads) {
      // Check to see if the load is invalidated from the start of the block to
      // the load itself.
      LoadInst *Load = LAO.L;
      BasicBlock *BB = Load->getParent();

      MemoryLocation Loc = MemoryLocation::get(Load);
      if (AAR.canInstructionRangeModRef(BB->front(), *Load, Loc,
                                        ModRefInfo::Mod)) {
        LLVM_DEBUG(dbgs() << "- potentially overwritten: " << *Load
                          << " in same block [keep]\n");
        ModifiedLoads.push_back(Load);
      }

      // Now check every path from the entry block to the load for transparency.
      // To do this, we perform a depth first search on the inverse CFG from the
      // loading block.
      for (BasicBlock *P : predecessors(BB)) {
        for (BasicBlock *TranspBB : inverse_depth_first_ext(P, TranspBlocks))
          if (AAR.canBasicBlockModify(*TranspBB, Loc)) {
            LLVM_DEBUG(dbgs() << "- potentially overwritten: " << *Load
                              << " in " << TranspBB->getName() << " [keep]\n");
            ModifiedLoads.push_back(Load);
          }
      }
    }

    // Remove loads that maybe modified.
    unsigned Idx = 0;
    auto It = API.Loads.begin(), End = API.Loads.end();
    while (It != End) {
      if (ModifiedLoads[Idx] != It->L) {
        ++It;
      } else {
        API.KeepArgument = true;
        It = API.Loads.erase(It);
        ++Idx;
      }
    }
  }

  // Check if any loads are left we can pre-load.
  if (API.Loads.empty()) {
    LLVM_DEBUG(dbgs() << "- all loads are potentially overwritten [no]\n");
    return false;
  }

  // If the path from the entry of the function to each load is free of
  // instructions that potentially invalidate the load, we can make the
  // transformation!
  return true;
}

Function *
ArgumentPromoter::promoteArguments(SmallPtrSetImpl<Argument *> &ArgsToPromote) {
  if (ArgsToPromote.empty())
    return nullptr;

#ifdef EXPENSIVE_CHECKS
  for (Argument *Arg : ArgsToPromote) {
    assert(ArgumentInfoMap.count(Arg) &&
           "Cannot promote if canPromoteArgument(Arg) was not called first!");
    ArgumentInfoMap.remove(Arg);
    assert(canPromoteArgument(Arg) &&
           "Cannot promote if canPromoteArgument(Arg) returned false!");
  }
#endif

  // HACK to prevent 'inalloca' arguments at the first argument position of the
  // new function. See uses for more information.
  bool FirstIsUndefGuardingInAlloca = false;

  Function &F = *(*ArgsToPromote.begin())->getParent();
  FunctionPromotionInfo &FPI = FunctionPromotionInfoMap[&F];
  const DataLayout &DL = F.getParent()->getDataLayout();

  // Keep track of the parameter attributes for the arguments that we are *not*
  // promoting. For the ones that we do promote, the parameter attributes are
  // lost.
  AttributeList PAL = F.getAttributes();
  SmallVector<AttributeSet, 8> NewArgsAttrVec;

  // Collect the types of the new arguments to create the function prototype.
  SmallVector<Type *, 8> NewArgsTypes;

  for (Argument &Arg : F.args()) {
    // Drop dead arguments.
    if (Arg.getNumUses() == 0) {
      // There may be remaining metadata uses of the argument for things like
      // llvm.dbg.value. Replace them with undef.
      Arg.replaceAllUsesWith(UndefValue::get(Arg.getType()));

      ++NumArgumentsDead;
      continue;
    }

    // Keep non-promoted arguments.
    if (!ArgsToPromote.count(&Arg)) {
      // Check if the first argument of the new function would be an 'inalloca'
      // argument of the old. In that case we introduce a dummy argument before
      // it as a precaution because the calling convention could require the
      // first argument to be passed 'in register'.
      if (Arg.hasInAllocaAttr() && NewArgsTypes.empty()) {
        NewArgsTypes.push_back(Type::getInt8Ty(Arg.getContext()));
        NewArgsAttrVec.push_back(AttributeSet());
        FirstIsUndefGuardingInAlloca = true;
        ++NumDummiesGuardingInAlloca;
      }
      NewArgsTypes.push_back(Arg.getType());
      NewArgsAttrVec.push_back(PAL.getParamAttributes(Arg.getArgNo()));
      continue;
    }

    // Look at the offsets we want to pre-load, if any.
    assert(FPI.ArgInfos.size() > Arg.getArgNo() &&
           "ArgumentPromotionInfo unavailable!");
    ArgumentPromotionInfo &API = FPI.ArgInfos[Arg.getArgNo()];
    assert(!API.Loads.empty() &&
           "Arguments without loads should be unused by now!");

    // Keep track of the promoted argument kinds.
    if (AreStatisticsEnabled()) {
      if (Arg.hasByValAttr())
        ++NumByValArgsPromoted;
      else if (API.Loads.size() == 1 && API.Loads.front().Offset == 0)
        ++NumArgumentsPromoted;
      else
        ++NumAggregatesPromoted;
    }

    // If there is no load specified, this has to be a by-val argument for
    // which we pre-load all specified offsets.
    if (Arg.hasByValAttr()) {
      Type *PointeeTy = Arg.getType()->getPointerElementType();
      if (auto *STy = dyn_cast<StructType>(PointeeTy)) {
        NewArgsTypes.append(STy->element_begin(), STy->element_end());
        NewArgsAttrVec.append(STy->getNumElements(), AttributeSet());
      } else if (auto *ATy = dyn_cast<ArrayType>(PointeeTy)) {
        NewArgsTypes.append(ATy->getNumElements(), ATy->getElementType());
        NewArgsAttrVec.append(ATy->getNumElements(), AttributeSet());
      } else {
        NewArgsTypes.push_back(PointeeTy);
        NewArgsAttrVec.push_back(AttributeSet());
      }
      continue;
    }

    // TODO: Be smarter wrt. coalescing loads/loaded bits.
    for (LoadAtOffset &LAO : API.Loads) {
      assert(LAO.L && isa<LoadInst>(LAO.L) && "Expected a load!");
      NewArgsTypes.push_back(LAO.L->getType());
    }
    NewArgsAttrVec.append(API.Loads.size(), AttributeSet());
  }

  FunctionType *FTy = F.getFunctionType();
  Type *RetTy = FTy->getReturnType();

  // Construct the new function type using the new arguments.
  assert(!FTy->isVarArg() && "Cannot handle varags yet!");
  FunctionType *NFTy =
      FunctionType::get(RetTy, NewArgsTypes, /* IsVarArg */ false);

  // Create the new function body and insert it into the module.
  Function *NF =
      Function::Create(NFTy, F.getLinkage(), F.getAddressSpace(), F.getName());
  NF->copyAttributesFrom(&F);

  // Patch the pointer to LLVM function in debug info descriptor.
  NF->setSubprogram(F.getSubprogram());
  F.setSubprogram(nullptr);

  // Recompute the parameter attributes list based on the new arguments for
  // the function.
  NF->setAttributes(AttributeList::get(F.getContext(), PAL.getFnAttributes(),
                                       PAL.getRetAttributes(), NewArgsAttrVec));
  NewArgsAttrVec.clear();

  LLVM_DEBUG(dbgs() << "ARG PROMOTION:  Promoting to:" << *NF << "\n"
                    << "From: " << F);

  F.getParent()->getFunctionList().insert(F.getIterator(), NF);
  NF->takeName(&F);

  auto NewArgI = NF->arg_begin();
  SmallVector<Value *, 16> NewOperands;
  while (!F.use_empty()) {
    CallSite CS(F.user_back());
    assert(CS.getCalledFunction() == &F);
    Instruction *Call = CS.getInstruction();
    const AttributeList &CallPAL = CS.getAttributes();
    IRBuilder<NoFolder> IRB(Call);

    // If we need a dummy argument, insert it first.
    if (FirstIsUndefGuardingInAlloca) {
      NewOperands.push_back(UndefValue::get(IRB.getInt8Ty()));
      NewArgsAttrVec.push_back({});
    }

    // An iterator for the old arguments.
    auto OldArgI = F.arg_begin();

    for (Value *ArgOp : CS.args()) {
      Argument *Arg = (OldArgI++);
      if (Arg->getNumUses() == 0)
        continue;
      if (!ArgsToPromote.count(Arg)) {
        ++NewArgI;
        NewOperands.push_back(ArgOp);
        NewArgsAttrVec.push_back(CallPAL.getParamAttributes(Arg->getArgNo()));
        continue;
      }

      assert(FPI.ArgInfos.size() > Arg->getArgNo() &&
             "ArgumentPromotionInfo unavailable!");
      ArgumentPromotionInfo &API = FPI.ArgInfos[Arg->getArgNo()];
      assert(!API.Loads.empty() &&
             "Arguments without loads should be unused by now!");

      // If there is no load specified, this has to be a by-val argument for
      // which we pre-load all specified offsets.
      if (Arg->hasByValAttr()) {
        assert(Arg->hasByValAttr() && "Expected a load or by-val argument!");
        Type *PointeeTy = Arg->getType()->getPointerElementType();

        if (auto *STy = dyn_cast<StructType>(PointeeTy)) {
          const StructLayout *SL = DL.getStructLayout(STy);
          generateLoadsOfByvalMembers(
              ArgOp, STy, IRB, NewOperands, API.Loads,
              [=](unsigned idx) { return SL->getElementOffset(idx); },
              [=](unsigned idx) { return STy->getElementType(idx); });
        } else if (auto *ATy = dyn_cast<ArrayType>(PointeeTy)) {
          Type *ElementType = ATy->getElementType();
          unsigned Elementsize = DL.getTypeAllocSize(ElementType);
          generateLoadsOfByvalMembers(
              ArgOp, ATy, IRB, NewOperands, API.Loads,
              [=](unsigned idx) { return Elementsize * idx; },
              [=](unsigned idx) { return ElementType; });
        } else {
          assert(API.Loads.size() == 1);
          NewOperands.push_back(
              IRB.CreateLoad(PointeeTy, ArgOp, ArgOp->getName() + ".val"));
        }
        NewArgsAttrVec.append(API.Loads.size(), AttributeSet());
        continue;
      }

      for (LoadAtOffset &LAO : API.Loads) {
        assert(LAO.L && isa<LoadInst>(LAO.L) && "Expected a load!");

        Type *LoadPtrTy = LAO.L->getPointerOperand()->getType();
        Value *Ptr = constructPointer(LoadPtrTy, ArgOp, LAO.Offset, IRB, DL);
        LoadInst *NewLoad =
            IRB.CreateLoad(LAO.L->getType(), Ptr, Ptr->getName() + ".val");
        NewLoad->setAlignment(LAO.L->getAlignment());

        // Transfer the AA info too.
        AAMDNodes AAInfo;
        LAO.L->getAAMetadata(AAInfo);
        NewLoad->setAAMetadata(AAInfo);

        NewOperands.push_back(NewLoad);
      }
      NewArgsAttrVec.append(API.Loads.size(), AttributeSet());
    }

    SmallVector<OperandBundleDef, 1> OpBundles;
    CS.getOperandBundlesAsDefs(OpBundles);

    CallSite NewCS;
    if (InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
      NewCS = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                                 NewOperands, OpBundles, "", Call);
    } else {
      auto *NewCall = CallInst::Create(NF, NewOperands, OpBundles, "", Call);
      NewCall->setTailCallKind(cast<CallInst>(Call)->getTailCallKind());
      NewCS = NewCall;
    }
    NewCS.setCallingConv(CS.getCallingConv());
    NewCS.setAttributes(
        AttributeList::get(F.getContext(), CallPAL.getFnAttributes(),
                           CallPAL.getRetAttributes(), NewArgsAttrVec));
    NewCS->setDebugLoc(Call->getDebugLoc());
    uint64_t W;
    if (Call->extractProfTotalWeight(W))
      NewCS->setProfWeight(W);
    NewOperands.clear();
    NewArgsAttrVec.clear();

    // Update the callgraph to know that the callsite has been transformed.
    if (CallSiteReplacer)
      (*CallSiteReplacer)(CS, NewCS);

    if (!Call->use_empty()) {
      Call->replaceAllUsesWith(NewCS.getInstruction());
      NewCS->takeName(Call);
    }

    // Remove the old call from the program, reducing the use-count of F.
    Call->eraseFromParent();
  }

  // Since we have now created the new function, splice the body of the old
  // function right into the new function, leaving the old rotting hulk of the
  // function empty.
  NF->getBasicBlockList().splice(NF->begin(), F.getBasicBlockList());

  // Eliminate all uses of old arguments with a local copy of the value.
  Instruction *AllocaIP = &*NF->getEntryBlock().getFirstInsertionPt();
  IRBuilder<NoFolder> IRB(AllocaIP);

  NewArgI = NF->arg_begin();

  // If we introduce a dummy guard, advance the iterator to skip it.
  if (FirstIsUndefGuardingInAlloca)
    ++NewArgI;

  // Flag to remember if we placed an alloca which could invalidate the 'tail'
  // property of calls. For now, we simply drop the tail property but we could
  // investigate if the alloca can, _somehow_, escape into a tail call as
  // argument or not.
  bool DropTailFromCalls = false;

  for (Argument &Arg : F.args()) {
    if (Arg.getNumUses() == 0)
      continue;

    Value *LocalCopy = UndefValue::get(Arg.getType());
    if (!ArgsToPromote.count(&Arg)) {
      LocalCopy = NewArgI++;
    } else {
      assert(FPI.ArgInfos.size() > Arg.getArgNo() &&
             "ArgumentPromotionInfo unavailable!");
      ArgumentPromotionInfo &API = FPI.ArgInfos[Arg.getArgNo()];
      assert(!API.Loads.empty() &&
             "Arguments without loads should be unused by now!");

      if (Arg.hasByValAttr()) {
        // If there is no load specified, this has to be a by-val argument for
        // which we pre-load all specified offsets.
        Type *PointeeTy = Arg.getType()->getPointerElementType();

        AllocaInst *AI = IRB.CreateAlloca(PointeeTy, nullptr, Arg.getName());
        AI->setAlignment(Arg.getParamAlignment());
        DropTailFromCalls = true;
        LocalCopy = AI;
        for (LoadAtOffset &LAO : API.Loads) {
          Value *Ptr = constructPointer(NewArgI->getType()->getPointerTo(),
                                        LocalCopy, LAO.Offset, IRB, DL);
          NewArgI->setName(Ptr->getName() + ".val");
          IRB.CreateStore(NewArgI++, Ptr);
        }
      } else {
        if (API.Loads.size() == 1 && API.Loads.front().Offset == 0)
          NewArgI->setName(Arg.getName() + ".val");
        for (LoadAtOffset &LAO : API.Loads) {
          assert(LAO.L && "Expected a load!");
          if (!NewArgI->hasName())
            NewArgI->takeName(LAO.L);
          LAO.L->replaceAllUsesWith(NewArgI++);
          LAO.L->eraseFromParent();
        }
      }
    }

    LocalCopy->takeName(&Arg);
    Arg.replaceAllUsesWith(LocalCopy);
  }

  // If we inserted an alloca and the function is potentially recursive we drop
  // "tail" markers from calls.
  if (DropTailFromCalls && !F.hasFnAttribute(Attribute::NoRecurse))
    dropTailFromCalls(*NF);

  // Keep track of the function types we have seen for the new function to avoid
  // endless promotion of recursive types.
  FunctionPromotionInfoMap[NF].SeenTypes.insert(F.getFunctionType());
  FunctionPromotionInfoMap.erase(&F);

  return NF;
}

PreservedAnalyses ArgumentPromotionPass::run(LazyCallGraph::SCC &C,
                                             CGSCCAnalysisManager &AM,
                                             LazyCallGraph &CG,
                                             CGSCCUpdateResult &UR) {
  bool Changed = false;

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();
  ArgumentPromoter::AARGetterTy AARGetter = [&](Function &F) -> AAResults & {
    return FAM.getResult<AAManager>(F);
  };

  ArgumentPromoter::TTIGetterTy TTIGetter =
      [&](Function &F) -> const TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };

  // TODO: Eliminate the function and call site replacer once the old PM is gone
  //       and there is only one way of updating the call graph.
  DenseMap<Function *, LazyCallGraph::Node *> NodeMap;
  ArgumentPromoter::FunctionReplacerTy ReplaceFunction = [&](Function &OldF,
                                                             Function &NewF) {
    // Directly substitute the functions in the call graph. Note that this
    // requires the old function to be completely dead and completely
    // replaced by the new function. It does no call graph updates, it merely
    // swaps out the particular function mapped to a particular node in the
    // graph.
    LazyCallGraph::Node *Node = NodeMap.lookup(&OldF);
    assert(Node && "No node found!");
    C.getOuterRefSCC().replaceNodeFunction(*Node, NewF);
    OldF.eraseFromParent();
  };

  ArgumentPromoter ArgPromoter(MaxElements, AARGetter, TTIGetter,
                               ReplaceFunction, nullptr);
  SmallPtrSet<Argument *, 16> ArgsToPromote;

  // Iterate until we stop promoting from this SCC.
  while (true) {

    for (LazyCallGraph::Node &N : C) {
      Function &F = N.getFunction();
      NodeMap[&F] = &N;
      ArgPromoter.analyze(F);
    }

    if (!ArgPromoter.promoteArguments())
      break;

    Changed = true;
  };

  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

namespace {

/// ArgPromotion - The 'by reference' to 'by value' argument promotion pass.
struct ArgPromotion : public CallGraphSCCPass {
  // Pass identification, replacement for typeid
  static char ID;

  explicit ArgPromotion(unsigned MaxElements = 3)
      : CallGraphSCCPass(ID), MaxElements(MaxElements) {
    initializeArgPromotionPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    getAAResultsAnalysisUsage(AU);
    CallGraphSCCPass::getAnalysisUsage(AU);
  }

  bool runOnSCC(CallGraphSCC &SCC) override;

private:
  using llvm::Pass::doInitialization;

  bool doInitialization(CallGraph &CG) override;

  /// The maximum number of elements to expand, or 0 for unlimited.
  unsigned MaxElements;
};

} // end anonymous namespace

char ArgPromotion::ID = 0;

INITIALIZE_PASS_BEGIN(ArgPromotion, "argpromotion",
                      "Promote 'by reference' arguments to scalars", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(ArgPromotion, "argpromotion",
                    "Promote 'by reference' arguments to scalars", false, false)

Pass *llvm::createArgumentPromotionPass(unsigned MaxElements) {
  return new ArgPromotion(MaxElements);
}

bool ArgPromotion::runOnSCC(CallGraphSCC &SCC) {
  if (skipSCC(SCC))
    return false;

  // Get the callgraph information that we need to update to reflect our
  // changes.
  CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  LegacyAARGetter LAARGetter(*this);
  ArgumentPromoter::AARGetterTy AARGetter = [&](Function &F) -> AAResults & {
    return LAARGetter(F);
  };

  ArgumentPromoter::TTIGetterTy TTIGetter =
      [&](Function &F) -> const TargetTransformInfo & {
    return getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  };

  ArgumentPromoter::FunctionReplacerTy ReplaceFunction = [&](Function &OldF,
                                                             Function &NewF) {
    // Update the call graph for the newly promoted function.
    CallGraphNode *OldNode = CG.getOrInsertFunction(&OldF);
    CallGraphNode *NewNode = CG.getOrInsertFunction(&NewF);
    assert(OldNode && NewNode && "TODO");

    NewNode->stealCalledFunctionsFrom(OldNode);
    assert(OldNode->getNumReferences() == 0 && "TODO");
    if (OldNode->getNumReferences() == 0)
      delete CG.removeFunctionFromModule(OldNode);
    else
      OldF.setLinkage(Function::ExternalLinkage);

    // And updat ethe SCC we're iterating as well.
    SCC.ReplaceNode(OldNode, NewNode);
  };

  ArgumentPromoter::CallSiteReplacerTy ReplaceCallSite = [&](CallSite OldCS,
                                                             CallSite NewCS) {
    Function *Caller = OldCS.getInstruction()->getParent()->getParent();
    CallGraphNode *NewCalleeNode =
        CG.getOrInsertFunction(NewCS.getCalledFunction());
    CallGraphNode *CallerNode = CG[Caller];
    CallerNode->replaceCallEdge(*cast<CallBase>(OldCS.getInstruction()),
                                *cast<CallBase>(NewCS.getInstruction()),
                                NewCalleeNode);
  };

  ArgumentPromoter ArgPromoter(MaxElements, AARGetter, TTIGetter,
                               ReplaceFunction, &ReplaceCallSite);

  bool Changed = false;
  // Iterate until we stop promoting from this SCC.
  while (true) {

    // Attempt to promote arguments from all functions in this SCC.
    for (CallGraphNode *OldNode : SCC)
      if (Function *OldF = OldNode->getFunction())
        ArgPromoter.analyze(*OldF);

    if (!ArgPromoter.promoteArguments())
      break;

    Changed = true;
  };

  return Changed;
}

bool ArgPromotion::doInitialization(CallGraph &CG) {
  return CallGraphSCCPass::doInitialization(CG);
}
