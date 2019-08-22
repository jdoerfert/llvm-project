//===- Loads.cpp - Local load analysis ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines simple local analyses for load instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Statepoint.h"

#include <limits.h>

using namespace llvm;

static bool isAligned(const Value *Base, const APInt &Offset, unsigned Align,
                      const DataLayout &DL) {
  APInt BaseAlign(Offset.getBitWidth(),
                  std::max(1U, getPointerAlignment(Base, DL)));
  APInt Alignment(Offset.getBitWidth(), Align);
  assert(Alignment.isPowerOf2() && "must be a power of 2!");
  return BaseAlign.uge(Alignment) && !(Offset & (Alignment-1));
}

<<<<<<< HEAD
/// Test if V is always a pointer to allocated and suitably aligned memory for
/// a simple load or store.
static bool isDereferenceableAndAlignedPointer(
    const Value *V, unsigned Align, const APInt &Size, const DataLayout &DL,
    const Instruction *CtxI, const DominatorTree *DT,
    SmallPtrSetImpl<const Value *> &Visited) {
  // Already visited?  Bail out, we've likely hit unreachable code.
||||||| merged common ancestors
static bool isAligned(const Value *Base, unsigned Align, const DataLayout &DL) {
  Type *Ty = Base->getType();
  assert(Ty->isSized() && "must be sized");
  APInt Offset(DL.getTypeStoreSizeInBits(Ty), 0);
  return isAligned(Base, Offset, Align, DL);
}

/// Test if V is always a pointer to allocated and suitably aligned memory for
/// a simple load or store.
static bool isDereferenceableAndAlignedPointer(
    const Value *V, unsigned Align, const APInt &Size, const DataLayout &DL,
    const Instruction *CtxI, const DominatorTree *DT,
    SmallPtrSetImpl<const Value *> &Visited) {
  // Already visited?  Bail out, we've likely hit unreachable code.
=======
static bool isAligned(const Value *Base, unsigned Align, const DataLayout &DL) {
  Type *Ty = Base->getType();
  assert(Ty->isSized() && "must be sized");
  APInt Offset(DL.getTypeStoreSizeInBits(Ty), 0);
  return isAligned(Base, Offset, Align, DL);
}

static unsigned getPointerAlignment(const Value *V, const DataLayout &DL,
                                    SmallPtrSetImpl<const Value *> &Visited) {
  assert(V->getType()->isPointerTy() && "must be pointer");

  static const unsigned MAX_ALIGN = 1U << 29;
  // The visited set catches recursion for "invalid" SSA instructions and allows
  // recursion on PHI nodes (not yet done).
>>>>>>> [WIP] Expose functions to determine pointer properties (Align & Deref)
  if (!Visited.insert(V).second)
    return MAX_ALIGN;

  const Value *Stripped = V->stripPointerCastsSameRepresentation();
  if (Stripped != V)
    return getPointerAlignment(Stripped, DL, Visited);

  unsigned Align = 0;
  if (auto *GO = dyn_cast<GlobalObject>(V)) {
    if (isa<Function>(GO)) {
      MaybeAlign FunctionPtrAlign = DL.getFunctionPtrAlign();
      unsigned Align = FunctionPtrAlign ? FunctionPtrAlign->value() : 0;
      switch (DL.getFunctionPtrAlignType()) {
      case DataLayout::FunctionPtrAlignType::Independent:
        return Align;
      case DataLayout::FunctionPtrAlignType::MultipleOfFunctionAlign:
        return std::max(Align, GO->getAlignment());
      }
    }
    Align = GO->getAlignment();
    if (Align == 0) {
      if (auto *GVar = dyn_cast<GlobalVariable>(GO)) {
        Type *ObjectType = GVar->getValueType();
        if (ObjectType->isSized()) {
          // If the object is defined in the current Module, we'll be giving
          // it the preferred alignment. Otherwise, we have to assume that it
          // may only have the minimum ABI alignment.
          if (GVar->isStrongDefinitionForLinker())
            Align = DL.getPreferredAlignment(GVar);
          else
            Align = DL.getABITypeAlignment(ObjectType);
        }
      }
    }
  } else if (auto *A = dyn_cast<Argument>(V)) {
    Align = A->getParamAlignment();
  } else if (auto *AI = dyn_cast<AllocaInst>(V)) {
    Align = AI->getAlignment();
    if (Align == 0) {
      Type *AllocatedType = AI->getAllocatedType();
      if (AllocatedType->isSized())
        Align = DL.getPrefTypeAlignment(AllocatedType);
    }
  } else if (auto *RI = dyn_cast<GCRelocateInst>(V)) {
    // For gc.relocate, look through relocations
    Align = getPointerAlignment(RI->getDerivedPtr(), DL, Visited);
  } else if (auto *Call = dyn_cast<CallBase>(V)) {
    Align = Call->getRetAlignment();
    if (Align == 0 && Call->getCalledFunction())
      Align = Call->getCalledFunction()->getAttributes().getRetAlignment();
    if (Align == 0)
      if (auto *RP = getArgumentAliasingToReturnedPointer(Call, true))
        Align = getPointerAlignment(RP, DL, Visited);
  } else if (auto *LI = dyn_cast<LoadInst>(V)) {
    if (MDNode *MD = LI->getMetadata(LLVMContext::MD_align)) {
      ConstantInt *CI = mdconst::extract<ConstantInt>(MD->getOperand(0));
      Align = CI->getLimitedValue();
    }
  } else if (auto *ASC = dyn_cast<AddrSpaceCastInst>(V)) {
    Align = getPointerAlignment(ASC->getPointerOperand(), DL, Visited);
  } else if (auto *GEP = dyn_cast<GEPOperator>(V)) {
    const Value *Base = GEP->getPointerOperand();

    APInt Offset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
    if (GEP->accumulateConstantOffset(DL, Offset) && Offset != 0) {
      auto BaseAlign = getPointerAlignment(Base, DL, Visited);
      if (BaseAlign <= 1) {
        // Propagate a base alignment of 1 and give up on a zero alignment.
        Align = BaseAlign;
      } else {
        // For the purpose of alignment computation we treat negative offsets as
        // if they were positive, so we only discuss positive offsets below.
        // We assume k_x values are existentially quantified.
        //
        // Base has alignment BA, thus:
        //   Base = k_0 * BA
        // GEP equals Base + Offset, thus:
        //   GEP = k_0 * BA + Offset
        // With GCD = gcd(BA, Offset), BA = k_1 * GCD, and Offset = k_2 * GCD we
        // can express GEP as follows:
        //   GEP = k_0 * (k_1 * GCD) + k_2 * GCD
        // The common factor in both terms is GCD, so we can express it as:
        //   GEP = GCD * (k_0 * k_1 + k_2)
        // Which implies that the GEP has GCD alignment.
        APInt GCD = APIntOps::GreatestCommonDivisor(
            Offset.abs(), APInt(Offset.getBitWidth(), BaseAlign));
        Align = GCD.getZExtValue();

      }
    }
    // Do not use type information below to improve the alignment.
    return Align;
  }

<<<<<<< HEAD
  bool CheckForNonNull = false;
  APInt KnownDerefBytes(Size.getBitWidth(),
                        V->getPointerDereferenceableBytes(DL, CheckForNonNull));
  if (KnownDerefBytes.getBoolValue() && KnownDerefBytes.uge(Size))
    if (!CheckForNonNull || isKnownNonZero(V, DL, 0, nullptr, CtxI, DT)) {
      // As we recursed through GEPs to get here, we've incrementally checked
      // that each step advanced by a multiple of the alignment. If our base is
      // properly aligned, then the original offset accessed must also be.
      Type *Ty = V->getType();
      assert(Ty->isSized() && "must be sized");
      APInt Offset(DL.getTypeStoreSizeInBits(Ty), 0);
      return isAligned(V, Offset, Align, DL);
    }
||||||| merged common ancestors
  bool CheckForNonNull = false;
  APInt KnownDerefBytes(Size.getBitWidth(),
                        V->getPointerDereferenceableBytes(DL, CheckForNonNull));
  if (KnownDerefBytes.getBoolValue()) {
    if (KnownDerefBytes.uge(Size))
      if (!CheckForNonNull || isKnownNonZero(V, DL, 0, nullptr, CtxI, DT))
        return isAligned(V, Align, DL);
  }
=======
  if (!Align) {
    Type *Ty = V->getType()->getPointerElementType();
    if (Ty->isSized())
      Align = DL.getABITypeAlignment(Ty);
  }
>>>>>>> [WIP] Expose functions to determine pointer properties (Align & Deref)

  return Align;
}

unsigned llvm::getPointerAlignment(const Value *V, const DataLayout &DL) {
  SmallPtrSet<const Value *, 4> Visited;
  return ::getPointerAlignment(V, DL, Visited);
}

static uint64_t
getPointerDereferenceableBytes(const Value *V, const DataLayout &DL,
                               bool &CanBeNull, bool &IsKnownDeref,
                               SmallPtrSetImpl<const Value *> &Visited) {
  assert(V->getType()->isPointerTy() && "must be pointer");
  CanBeNull = false;
  IsKnownDeref = true;

  // The visited set catches recursion for "invalid" SSA instructions and allows
  // recursion on PHI nodes (not yet done).
  if (!Visited.insert(V).second)
    return std::numeric_limits<uint64_t>::max();

  const Value *Stripped = V->stripPointerCastsSameRepresentation();
  if (Stripped != V)
    return getPointerDereferenceableBytes(Stripped, DL, CanBeNull, IsKnownDeref,
                                          Visited);

  const Function *F = nullptr;
  uint64_t DerefBytes = 0;
  CanBeNull = false;
  if (auto *A = dyn_cast<Argument>(V)) {
    F = A->getParent();
    DerefBytes = A->getDereferenceableBytes();
    if (DerefBytes == 0 && (A->hasByValAttr() || A->hasStructRetAttr())) {
      Type *PT = cast<PointerType>(A->getType())->getElementType();
      if (PT->isSized())
        DerefBytes = DL.getTypeStoreSize(PT);
    }
    if (DerefBytes == 0) {
      DerefBytes = A->getDereferenceableOrNullBytes();
      CanBeNull = true;
    }
  } else if (auto *LI = dyn_cast<LoadInst>(V)) {
    if (MDNode *MD = LI->getMetadata(LLVMContext::MD_dereferenceable)) {
      ConstantInt *CI = mdconst::extract<ConstantInt>(MD->getOperand(0));
      DerefBytes = CI->getLimitedValue();
    }
    if (DerefBytes == 0) {
      if (MDNode *MD =
              LI->getMetadata(LLVMContext::MD_dereferenceable_or_null)) {
        ConstantInt *CI = mdconst::extract<ConstantInt>(MD->getOperand(0));
        DerefBytes = CI->getLimitedValue();
      }
      CanBeNull = true;
    }
  } else if (auto *IP = dyn_cast<IntToPtrInst>(V)) {
    if (MDNode *MD = IP->getMetadata(LLVMContext::MD_dereferenceable)) {
      ConstantInt *CI = mdconst::extract<ConstantInt>(MD->getOperand(0));
      DerefBytes = CI->getLimitedValue();
    }
    if (DerefBytes == 0) {
      if (MDNode *MD =
              IP->getMetadata(LLVMContext::MD_dereferenceable_or_null)) {
        ConstantInt *CI = mdconst::extract<ConstantInt>(MD->getOperand(0));
        DerefBytes = CI->getLimitedValue();
      }
      CanBeNull = true;
    }
  } else if (auto *AI = dyn_cast<AllocaInst>(V)) {
    if (!AI->isArrayAllocation()) {
      DerefBytes = DL.getTypeStoreSize(AI->getAllocatedType());
      CanBeNull = false;
    }
  } else if (auto *GV = dyn_cast<GlobalVariable>(V)) {
    if (GV->getValueType()->isSized() && !GV->hasExternalWeakLinkage()) {
      // TODO: Don't outright reject hasExternalWeakLinkage but set the
      // CanBeNull flag.
      DerefBytes = DL.getTypeStoreSize(GV->getValueType());
      CanBeNull = false;
    }
  } else if (auto *RI = dyn_cast<GCRelocateInst>(V)) {
    // For gc.relocate, look through relocations, must be checkd before CallBase.
    return getPointerDereferenceableBytes(RI->getDerivedPtr(), DL, CanBeNull,
                                          IsKnownDeref, Visited);
  } else if (auto *ASC = dyn_cast<AddrSpaceCastInst>(V)) {
    DerefBytes = getPointerDereferenceableBytes(
        ASC->getPointerOperand(), DL, CanBeNull, IsKnownDeref, Visited);
  } else if (auto *GEP = dyn_cast<GEPOperator>(V)) {
    APInt Offset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
    // Give up on non-constant GEPs.
    if (!GEP->accumulateConstantOffset(DL, Offset))
      return 0;

    // If Base has N dereferenceable bytes and is O (=Offset) away from the
    // GEP, then:
    //  - if 0 is null, we take the base result.
    //  - if O is positive, the GEP has max(0, N - O) dereferenceable bytes
    //                      because it is O bytes advanced from the base.
    //  - if O is negative and inbounds, the GEP has N + abs(O) dereferenceable
    //                                   bytes because inbounds need to stay in
    //                                   the same allocation.
    //  - if O is negative and not inbounds, the GEP has 0 dereferenceable bytes
    //                                       because we do not know if we are
    //                                       still in the allocation or not.
    //
    // The "can be null" is set to false if we have an offset that is not null
    // and an inbounds GEP or dereferenceable bytes. Note that this is later
    // overwritten if null pointers are defined.
    const Value *Base = GEP->getPointerOperand();
    uint64_t BaseDerefBytes = getPointerDereferenceableBytes(
        Base, DL, CanBeNull, IsKnownDeref, Visited);
    if (Offset == 0) {
      DerefBytes = BaseDerefBytes;
    } else if (Offset.getSExtValue() > 0) {
      DerefBytes =
          std::max(int64_t(BaseDerefBytes - Offset.getZExtValue()), int64_t(0));
      CanBeNull = !GEP->isInBounds();
    } else {
      assert(Offset.getSExtValue() < 0 && "Did not expect zero offset!");
      if (GEP->isInBounds())
        DerefBytes = BaseDerefBytes + Offset.abs().getZExtValue();
      else
        DerefBytes = 0;
      CanBeNull = !GEP->isInBounds();
    }
    // Assume dereferenceable implies nonnull here but note that we later revert
    // the decision if it does not based on the pointer address space.
    CanBeNull &= (DerefBytes == 0);
  } else if (const auto *Call = dyn_cast<CallBase>(V)) {
    DerefBytes = Call->getDereferenceableBytes(AttributeList::ReturnIndex);
    if (DerefBytes == 0) {
      DerefBytes =
          Call->getDereferenceableOrNullBytes(AttributeList::ReturnIndex);
      CanBeNull = true;
    }
    if (DerefBytes == 0)
      if (auto *RP = getArgumentAliasingToReturnedPointer(Call, true))
        DerefBytes = getPointerDereferenceableBytes(RP, DL, CanBeNull,
                                                    IsKnownDeref, Visited);
  }

  if (auto *I = dyn_cast<Instruction>(V))
    F = I->getFunction();

  IsKnownDeref = !CanBeNull;
  CanBeNull |= NullPointerIsDefined(F, V->getType()->getPointerAddressSpace());
  return DerefBytes;
}

uint64_t llvm::getPointerDereferenceableBytes(const Value *V,
                                              const DataLayout &DL,
                                              bool &CanBeNull,
                                              bool &IsKnownDeref) {
  SmallPtrSet<const Value *, 4> Visited;
  return ::getPointerDereferenceableBytes(V, DL, CanBeNull, IsKnownDeref,
                                          Visited);
}

bool llvm::isDereferenceableAndAlignedPointer(const Value *V, unsigned Align,
                                              const APInt &Size,
                                              const DataLayout &DL,
                                              const Instruction *CtxI,
                                              const DominatorTree *DT) {
  assert(Align != 0 && "expected explicitly set alignment");
  // Note: At the moment, Size can be zero.  This ends up being interpreted as
  // a query of whether [Base, V] is dereferenceable and V is aligned (since
  // that's what the implementation happened to do).  It's unclear if this is
  // the desired semantic, but at least SelectionDAG does exercise this case.

  bool IsKnownDeref = false, CanBeNull = false;
  APInt KnownDerefBytes(
      Size.getBitWidth(),
      getPointerDereferenceableBytes(V, DL, CanBeNull, IsKnownDeref));
  if (KnownDerefBytes.getBoolValue())
    if (KnownDerefBytes.uge(Size))
      if (IsKnownDeref || isKnownNonZero(V, DL, 0, nullptr, CtxI, DT))
        return ::isAligned(V, Align, DL);

  // If we don't know, assume the worst.
  return false;
}

bool llvm::isDereferenceableAndAlignedPointer(const Value *V, Type *Ty,
                                              unsigned Align,
                                              const DataLayout &DL,
                                              const Instruction *CtxI,
                                              const DominatorTree *DT) {
  // When dereferenceability information is provided by a dereferenceable
  // attribute, we know exactly how many bytes are dereferenceable. If we can
  // determine the exact offset to the attributed variable, we can use that
  // information here.

  // Require ABI alignment for loads without alignment specification
  if (Align == 0)
    Align = DL.getABITypeAlignment(Ty);

  if (!Ty->isSized())
    return false;
<<<<<<< HEAD

  APInt AccessSize(DL.getIndexTypeSizeInBits(V->getType()),
                   DL.getTypeStoreSize(Ty));
  return isDereferenceableAndAlignedPointer(V, Align, AccessSize,
                                            DL, CtxI, DT);
||||||| merged common ancestors

  SmallPtrSet<const Value *, 32> Visited;
  return ::isDereferenceableAndAlignedPointer(
      V, Align,
      APInt(DL.getIndexTypeSizeInBits(V->getType()), DL.getTypeStoreSize(Ty)),
      DL, CtxI, DT, Visited);
=======

  return ::isDereferenceableAndAlignedPointer(
      V, Align,
      APInt(DL.getIndexTypeSizeInBits(V->getType()), DL.getTypeStoreSize(Ty)),
      DL, CtxI, DT);
>>>>>>> [WIP] Expose functions to determine pointer properties (Align & Deref)
}

bool llvm::isDereferenceablePointer(const Value *V, Type *Ty,
                                    const DataLayout &DL,
                                    const Instruction *CtxI,
                                    const DominatorTree *DT) {
  return isDereferenceableAndAlignedPointer(V, Ty, 1, DL, CtxI, DT);
}

/// Test if A and B will obviously have the same value.
///
/// This includes recognizing that %t0 and %t1 will have the same
/// value in code like this:
/// \code
///   %t0 = getelementptr \@a, 0, 3
///   store i32 0, i32* %t0
///   %t1 = getelementptr \@a, 0, 3
///   %t2 = load i32* %t1
/// \endcode
///
static bool AreEquivalentAddressValues(const Value *A, const Value *B) {
  // Test if the values are trivially equivalent.
  if (A == B)
    return true;

  // Test if the values come from identical arithmetic instructions.
  // Use isIdenticalToWhenDefined instead of isIdenticalTo because
  // this function is only used when one address use dominates the
  // other, which means that they'll always either have the same
  // value or one of them will have an undefined value.
  if (isa<BinaryOperator>(A) || isa<CastInst>(A) || isa<PHINode>(A) ||
      isa<GetElementPtrInst>(A))
    if (const Instruction *BI = dyn_cast<Instruction>(B))
      if (cast<Instruction>(A)->isIdenticalToWhenDefined(BI))
        return true;

  // Otherwise they may not be equivalent.
  return false;
}

bool llvm::isDereferenceableAndAlignedInLoop(LoadInst *LI, Loop *L,
                                             ScalarEvolution &SE,
                                             DominatorTree &DT) {
  auto &DL = LI->getModule()->getDataLayout();
  Value *Ptr = LI->getPointerOperand();

  APInt EltSize(DL.getIndexTypeSizeInBits(Ptr->getType()),
                DL.getTypeStoreSize(LI->getType()));
  unsigned Align = LI->getAlignment();
  if (Align == 0)
    Align = DL.getABITypeAlignment(LI->getType());

  Instruction *HeaderFirstNonPHI = L->getHeader()->getFirstNonPHI();

  // If given a uniform (i.e. non-varying) address, see if we can prove the
  // access is safe within the loop w/o needing predication.
  if (L->isLoopInvariant(Ptr))
    return isDereferenceableAndAlignedPointer(Ptr, Align, EltSize, DL,
                                              HeaderFirstNonPHI, &DT);    

  // Otherwise, check to see if we have a repeating access pattern where we can
  // prove that all accesses are well aligned and dereferenceable.
  auto *AddRec = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(Ptr));
  if (!AddRec || AddRec->getLoop() != L || !AddRec->isAffine())
    return false;
  auto* Step = dyn_cast<SCEVConstant>(AddRec->getStepRecurrence(SE));
  if (!Step)
    return false;
  // TODO: generalize to access patterns which have gaps
  if (Step->getAPInt() != EltSize)
    return false;

  // TODO: If the symbolic trip count has a small bound (max count), we might
  // be able to prove safety.
  auto TC = SE.getSmallConstantTripCount(L);
  if (!TC)
    return false;

  const APInt AccessSize = TC * EltSize;

  auto *StartS = dyn_cast<SCEVUnknown>(AddRec->getStart());
  if (!StartS)
    return false;
  assert(SE.isLoopInvariant(StartS, L) && "implied by addrec definition");
  Value *Base = StartS->getValue();

  // For the moment, restrict ourselves to the case where the access size is a
  // multiple of the requested alignment and the base is aligned.
  // TODO: generalize if a case found which warrants
  if (EltSize.urem(Align) != 0)
    return false;
  return isDereferenceableAndAlignedPointer(Base, Align, AccessSize,
                                            DL, HeaderFirstNonPHI, &DT);
}

/// Check if executing a load of this pointer value cannot trap.
///
/// If DT and ScanFrom are specified this method performs context-sensitive
/// analysis and returns true if it is safe to load immediately before ScanFrom.
///
/// If it is not obviously safe to load from the specified pointer, we do
/// a quick local scan of the basic block containing \c ScanFrom, to determine
/// if the address is already accessed.
///
/// This uses the pointee type to determine how many bytes need to be safe to
/// load from the pointer.
bool llvm::isSafeToLoadUnconditionally(Value *V, unsigned Align, APInt &Size,
                                       const DataLayout &DL,
                                       Instruction *ScanFrom,
                                       const DominatorTree *DT) {
  // Zero alignment means that the load has the ABI alignment for the target
  if (Align == 0)
    Align = DL.getABITypeAlignment(V->getType()->getPointerElementType());
  assert(isPowerOf2_32(Align));

  // If DT is not specified we can't make context-sensitive query
  const Instruction* CtxI = DT ? ScanFrom : nullptr;
  if (isDereferenceableAndAlignedPointer(V, Align, Size, DL, CtxI, DT))
    return true;

  if (!ScanFrom)
    return false;

  if (Size.getBitWidth() > 64)
    return false;
  const uint64_t LoadSize = Size.getZExtValue();

  // Otherwise, be a little bit aggressive by scanning the local block where we
  // want to check to see if the pointer is already being loaded or stored
  // from/to.  If so, the previous load or store would have already trapped,
  // so there is no harm doing an extra load (also, CSE will later eliminate
  // the load entirely).
  BasicBlock::iterator BBI = ScanFrom->getIterator(),
                       E = ScanFrom->getParent()->begin();

  // We can at least always strip pointer casts even though we can't use the
  // base here.
  V = V->stripPointerCasts();

  while (BBI != E) {
    --BBI;

    // If we see a free or a call which may write to memory (i.e. which might do
    // a free) the pointer could be marked invalid.
    if (isa<CallInst>(BBI) && BBI->mayWriteToMemory() &&
        !isa<DbgInfoIntrinsic>(BBI))
      return false;

    Value *AccessedPtr;
    unsigned AccessedAlign;
    if (LoadInst *LI = dyn_cast<LoadInst>(BBI)) {
      // Ignore volatile loads. The execution of a volatile load cannot
      // be used to prove an address is backed by regular memory; it can,
      // for example, point to an MMIO register.
      if (LI->isVolatile())
        continue;
      AccessedPtr = LI->getPointerOperand();
      AccessedAlign = LI->getAlignment();
    } else if (StoreInst *SI = dyn_cast<StoreInst>(BBI)) {
      // Ignore volatile stores (see comment for loads).
      if (SI->isVolatile())
        continue;
      AccessedPtr = SI->getPointerOperand();
      AccessedAlign = SI->getAlignment();
    } else
      continue;

    Type *AccessedTy = AccessedPtr->getType()->getPointerElementType();
    if (AccessedAlign == 0)
      AccessedAlign = DL.getABITypeAlignment(AccessedTy);
    if (AccessedAlign < Align)
      continue;

    // Handle trivial cases.
    if (AccessedPtr == V &&
        LoadSize <= DL.getTypeStoreSize(AccessedTy))
      return true;

    if (AreEquivalentAddressValues(AccessedPtr->stripPointerCasts(), V) &&
        LoadSize <= DL.getTypeStoreSize(AccessedTy))
      return true;
  }
  return false;
}

bool llvm::isSafeToLoadUnconditionally(Value *V, Type *Ty, unsigned Align,
                                       const DataLayout &DL,
                                       Instruction *ScanFrom,
                                       const DominatorTree *DT) {
  APInt Size(DL.getIndexTypeSizeInBits(V->getType()), DL.getTypeStoreSize(Ty));
  return isSafeToLoadUnconditionally(V, Align, Size, DL, ScanFrom, DT);
}

  /// DefMaxInstsToScan - the default number of maximum instructions
/// to scan in the block, used by FindAvailableLoadedValue().
/// FindAvailableLoadedValue() was introduced in r60148, to improve jump
/// threading in part by eliminating partially redundant loads.
/// At that point, the value of MaxInstsToScan was already set to '6'
/// without documented explanation.
cl::opt<unsigned>
llvm::DefMaxInstsToScan("available-load-scan-limit", cl::init(6), cl::Hidden,
  cl::desc("Use this to specify the default maximum number of instructions "
           "to scan backward from a given instruction, when searching for "
           "available loaded value"));

Value *llvm::FindAvailableLoadedValue(LoadInst *Load,
                                      BasicBlock *ScanBB,
                                      BasicBlock::iterator &ScanFrom,
                                      unsigned MaxInstsToScan,
                                      AliasAnalysis *AA, bool *IsLoad,
                                      unsigned *NumScanedInst) {
  // Don't CSE load that is volatile or anything stronger than unordered.
  if (!Load->isUnordered())
    return nullptr;

  return FindAvailablePtrLoadStore(
      Load->getPointerOperand(), Load->getType(), Load->isAtomic(), ScanBB,
      ScanFrom, MaxInstsToScan, AA, IsLoad, NumScanedInst);
}

Value *llvm::FindAvailablePtrLoadStore(Value *Ptr, Type *AccessTy,
                                       bool AtLeastAtomic, BasicBlock *ScanBB,
                                       BasicBlock::iterator &ScanFrom,
                                       unsigned MaxInstsToScan,
                                       AliasAnalysis *AA, bool *IsLoadCSE,
                                       unsigned *NumScanedInst) {
  if (MaxInstsToScan == 0)
    MaxInstsToScan = ~0U;

  const DataLayout &DL = ScanBB->getModule()->getDataLayout();

  // Try to get the store size for the type.
  auto AccessSize = LocationSize::precise(DL.getTypeStoreSize(AccessTy));

  Value *StrippedPtr = Ptr->stripPointerCasts();

  while (ScanFrom != ScanBB->begin()) {
    // We must ignore debug info directives when counting (otherwise they
    // would affect codegen).
    Instruction *Inst = &*--ScanFrom;
    if (isa<DbgInfoIntrinsic>(Inst))
      continue;

    // Restore ScanFrom to expected value in case next test succeeds
    ScanFrom++;

    if (NumScanedInst)
      ++(*NumScanedInst);

    // Don't scan huge blocks.
    if (MaxInstsToScan-- == 0)
      return nullptr;

    --ScanFrom;
    // If this is a load of Ptr, the loaded value is available.
    // (This is true even if the load is volatile or atomic, although
    // those cases are unlikely.)
    if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
      if (AreEquivalentAddressValues(
              LI->getPointerOperand()->stripPointerCasts(), StrippedPtr) &&
          CastInst::isBitOrNoopPointerCastable(LI->getType(), AccessTy, DL)) {

        // We can value forward from an atomic to a non-atomic, but not the
        // other way around.
        if (LI->isAtomic() < AtLeastAtomic)
          return nullptr;

        if (IsLoadCSE)
            *IsLoadCSE = true;
        return LI;
      }

    if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
      Value *StorePtr = SI->getPointerOperand()->stripPointerCasts();
      // If this is a store through Ptr, the value is available!
      // (This is true even if the store is volatile or atomic, although
      // those cases are unlikely.)
      if (AreEquivalentAddressValues(StorePtr, StrippedPtr) &&
          CastInst::isBitOrNoopPointerCastable(SI->getValueOperand()->getType(),
                                               AccessTy, DL)) {

        // We can value forward from an atomic to a non-atomic, but not the
        // other way around.
        if (SI->isAtomic() < AtLeastAtomic)
          return nullptr;

        if (IsLoadCSE)
          *IsLoadCSE = false;
        return SI->getOperand(0);
      }

      // If both StrippedPtr and StorePtr reach all the way to an alloca or
      // global and they are different, ignore the store. This is a trivial form
      // of alias analysis that is important for reg2mem'd code.
      if ((isa<AllocaInst>(StrippedPtr) || isa<GlobalVariable>(StrippedPtr)) &&
          (isa<AllocaInst>(StorePtr) || isa<GlobalVariable>(StorePtr)) &&
          StrippedPtr != StorePtr)
        continue;

      // If we have alias analysis and it says the store won't modify the loaded
      // value, ignore the store.
      if (AA && !isModSet(AA->getModRefInfo(SI, StrippedPtr, AccessSize)))
        continue;

      // Otherwise the store that may or may not alias the pointer, bail out.
      ++ScanFrom;
      return nullptr;
    }

    // If this is some other instruction that may clobber Ptr, bail out.
    if (Inst->mayWriteToMemory()) {
      // If alias analysis claims that it really won't modify the load,
      // ignore it.
      if (AA && !isModSet(AA->getModRefInfo(Inst, StrippedPtr, AccessSize)))
        continue;

      // May modify the pointer, bail out.
      ++ScanFrom;
      return nullptr;
    }
  }

  // Got to the start of the block, we didn't find it, but are done for this
  // block.
  return nullptr;
}
