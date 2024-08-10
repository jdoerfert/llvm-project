//===-- OffloadSanitizer.cpp - Offload sanitizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/OffloadSanitizer.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <string>

using namespace llvm;

#define DEBUG_TYPE "offload-sanitizer"

namespace {

class OffloadSanitizerImpl final {
public:
  OffloadSanitizerImpl(Module &M, FunctionAnalysisManager &FAM)
      : M(M), FAM(FAM), Ctx(M.getContext()) {}

  bool instrument();

private:
  /// We support address space 0 to 5 right now.
  static constexpr int NumSupportedAddressSpaces = 6;

  bool isASType(Type &T) {
    return T.isPointerTy() && T.getPointerAddressSpace();
  };
  Type *getWithoutAS(Type &T) { return isASType(T) ? T.getPointerTo() : &T; };

  bool shouldInstrumentFunction(Function *Fn);

  struct AccessInfoTy {
    Instruction *I;
    unsigned PtrOpIdx;
    unsigned AS;
  };

  void removeAS(Function &Fn, SmallVectorImpl<Instruction *> &ASInsts);

  bool instrumentFunction(Function &Fn);
  void instrumentTrapInstructions(SmallVectorImpl<IntrinsicInst *> &TrapCalls);
  void instrumentCallInsts(SmallVectorImpl<CallInst *> &CallInsts);
  void instrumentLifetimeIntrinsics(
      SmallVectorImpl<LifetimeIntrinsic *> &LifetimeInsts);
  void instrumentUnreachableInstructions(
      SmallVectorImpl<UnreachableInst *> &UnreachableInsts);
  void instrumentAccesses(SmallVectorImpl<AccessInfoTy> &Accesses);
  void instrumentAllocaInstructions(SmallVectorImpl<AllocaInst *> &AllocaInsts);

  FunctionCallee getOrCreateFn(FunctionCallee &FC, StringRef Name, Type *RetTy,
                               ArrayRef<Type *> ArgTys) {
    if (!FC) {
      auto *NewAllocationFnTy = FunctionType::get(RetTy, ArgTys, false);
      FC = M.getOrInsertFunction(Name, NewAllocationFnTy);
    }
    return FC;
  }

  /// void __offload_san_trap_info(Int64Ty);
  FunctionCallee TrapInfoFn;
  FunctionCallee getTrapInfoFn() {
    return getOrCreateFn(TrapInfoFn, "__offload_san_trap_info", VoidTy,
                         {/*PC*/ Int64Ty});
  }

  /// void __offload_san_unreachable_info(Int64Ty);
  FunctionCallee UnreachableInfoFn;
  FunctionCallee getUnreachableInfoFn() {
    return getOrCreateFn(UnreachableInfoFn, "__offload_san_unreachable_info",
                         VoidTy, {/*PC*/ Int64Ty});
  }

  /// PtrTy __offload_san_unpack(Int64Ty, PtrTy);
  FunctionCallee UnpackFn;
  FunctionCallee getUnpackFn() {
    return getOrCreateFn(UnpackFn, "__offload_san_unpack", PtrTy,
                         {/*PC*/ Int64Ty, PtrTy});
  }

  /// ptr(AS) __offload_san_check_as<AS>_access(/* PC */Int64Ty,
  /// 					        /* FakePtr */ PtrTy,
  /// 				                /* Size */Int32Ty);
  FunctionCallee CheckAccessFn[NumSupportedAddressSpaces];
  FunctionCallee getCheckAccessFn(unsigned AS) {
    assert(AS < NumSupportedAddressSpaces && "Unexpected address space!");
    return getOrCreateFn(CheckAccessFn[AS],
                         "__offload_san_check_as" + std::to_string(AS) +
                             "_access",
                         ASPtrTy[AS], {/*PC*/ Int64Ty, PtrTy, Int32Ty});
  }

  /// PtrTy __offload_san_register_alloca(/* PC */ Int64Ty,
  /// 						/* RealPtr */ AllocaPtrTy,
  /// 						/* Size */ Int32Ty);
  FunctionCallee AllocaRegisterFn;
  FunctionCallee getAllocaRegisterFn() {
    getOrCreateFn(AllocaRegisterFn, "__offload_san_register_alloca", PtrTy,
                  {/*PC*/ Int64Ty, AllocaPtrTy, Int32Ty});
    //    cast<Function>(AllocaRegisterFn.getCallee())
    //        ->addRetAttr(Attribute::NoAlias);
    return AllocaRegisterFn;
  }

  CallInst *createCall(IRBuilder<> &IRB, FunctionCallee Callee,
                       ArrayRef<Value *> Args = std::nullopt,
                       const Twine &Name = "") {
    RTCalls.push_back(IRB.CreateCall(Callee, Args, Name));
    return RTCalls.back();
  }
  SmallVector<CallInst *> RTCalls;

  Value *getPC(IRBuilder<> &IRB) {
    static int X = 0;
    return ConstantInt::get(Int64Ty, X++);
    return IRB.CreateIntrinsic(Int64Ty, Intrinsic::amdgcn_s_getpc, {}, nullptr,
                               "PC");
  }

  Module &M;
  const DataLayout &DL = M.getDataLayout();
  FunctionAnalysisManager &FAM;
  LLVMContext &Ctx;

  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *IntptrTy = M.getDataLayout().getIntPtrType(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);
  IntegerType *Int8Ty = Type::getInt8Ty(Ctx);
  IntegerType *Int32Ty = Type::getInt32Ty(Ctx);
  IntegerType *Int64Ty = Type::getInt64Ty(Ctx);
  PointerType *AllocaPtrTy = PointerType::get(Ctx, DL.getAllocaAddrSpace());
  PointerType *ASPtrTy[NumSupportedAddressSpaces] = {
      PointerType::get(Ctx, 0), PointerType::get(Ctx, 1),
      PointerType::get(Ctx, 2), PointerType::get(Ctx, 3),
      PointerType::get(Ctx, 4), PointerType::get(Ctx, 5)};
};

} // end anonymous namespace

bool OffloadSanitizerImpl::shouldInstrumentFunction(Function *Fn) {
  if (!Fn || Fn->isDeclaration())
    return false;
  if (Fn->getName().contains("ompx") || Fn->getName().contains("__kmpc") ||
      Fn->getName().starts_with("rpc_"))
    return false;
  return !Fn->hasFnAttribute(Attribute::DisableSanitizerInstrumentation);
}

void OffloadSanitizerImpl::removeAS(Function &Fn,
                                    SmallVectorImpl<Instruction *> &ASInsts) {

  DenseMap<Value *, Value *> VMap;

  std::function<Value *(Value &)> GetAsGeneric = [&](Value &V) -> Value * {
    if (!isASType(*V.getType()))
      return &V;
    auto *&NewV = VMap[&V];
    if (!NewV) {
      auto *IP = &Fn.getEntryBlock().front();
      if (auto *I = dyn_cast<Instruction>(&V))
        IP = I->getNextNode();
      NewV = new AddrSpaceCastInst(&V, getWithoutAS(*V.getType()),
                                   V.getName() + ".noas", IP);
    }
    return NewV;
  };

  SmallVector<PHINode *> PHIs;
  for (auto *I : ASInsts) {
    errs() << "I: " << *I << "\n";
    switch (I->getOpcode()) {
    case Instruction::Load: {
      auto &LI = cast<LoadInst>(*I);
      auto *GenericOp = GetAsGeneric(*LI.getPointerOperand());
      if (LI.getPointerAddressSpace())
        GenericOp = new AddrSpaceCastInst(GenericOp, LI.getPointerOperandType(),
                                          GenericOp->getName() + ".as", &LI);
      LI.setOperand(LI.getPointerOperandIndex(), GenericOp);
      VMap[I] = GetAsGeneric(LI);
      break;
    }
    case Instruction::Store: {
      auto &SI = cast<StoreInst>(*I);
      auto *GenericOp = GetAsGeneric(*SI.getPointerOperand());
      if (SI.getPointerAddressSpace())
        GenericOp = new AddrSpaceCastInst(GenericOp, SI.getPointerOperandType(),
                                          GenericOp->getName() + ".as", &SI);
      SI.setOperand(SI.getPointerOperandIndex(), GenericOp);
      break;
    }
    case Instruction::AtomicRMW: {
      auto &ARMW = cast<AtomicRMWInst>(*I);
      auto *GenericOp = GetAsGeneric(*ARMW.getPointerOperand());
      if (ARMW.getPointerAddressSpace())
        GenericOp = new AddrSpaceCastInst(GenericOp,
                                          ARMW.getPointerOperand()->getType(),
                                          GenericOp->getName() + ".as", &ARMW);
      ARMW.setOperand(ARMW.getPointerOperandIndex(), GenericOp);
      VMap[I] = GetAsGeneric(ARMW);
      break;
    }
    case Instruction::GetElementPtr: {
      auto &GEP = cast<GetElementPtrInst>(*I);
      GEP.mutateType(getWithoutAS(*GEP.getType()));
      GEP.setSourceElementType(getWithoutAS(*GEP.getSourceElementType()));
      GEP.setResultElementType(getWithoutAS(*GEP.getResultElementType()));
      GEP.setOperand(GEP.getPointerOperandIndex(),
                     GetAsGeneric(*GEP.getPointerOperand()));
      break;
    }
    case Instruction::AddrSpaceCast: {
      auto &ASC = cast<AddrSpaceCastInst>(*I);
      VMap[I] = GetAsGeneric(*ASC.getPointerOperand());
      break;
    }
    case Instruction::Select: {
      auto &SI = cast<SelectInst>(*I);
      SI.mutateType(getWithoutAS(*SI.getType()));
      SI.setTrueValue(GetAsGeneric(*SI.getTrueValue()));
      SI.setFalseValue(GetAsGeneric(*SI.getFalseValue()));
      break;
    }
    case Instruction::PHI: {
      auto &PHI = cast<PHINode>(*I);
      PHI.mutateType(getWithoutAS(*PHI.getType()));
      PHIs.push_back(&PHI);
      break;
    }
    case Instruction::ICmp: {
      auto &II = cast<ICmpInst>(*I);
      II.setOperand(0, GetAsGeneric(*II.getOperand(0)));
      II.setOperand(1, GetAsGeneric(*II.getOperand(1)));
      break;
    }
    case Instruction::Call: {
      auto &CI = cast<CallInst>(*I);
      auto *Callee = CI.getCalledFunction();
      if (shouldInstrumentFunction(Callee)) {
        for (unsigned I = 0, E = CI.arg_size(); I < E; ++I)
          CI.setArgOperand(I, GetAsGeneric(*CI.getArgOperand(I)));

        auto *FT = CI.getFunctionType();
        SmallVector<Type *> ArgTypes;
        for (auto *ArgType : FT->params())
          ArgTypes.push_back(getWithoutAS(*ArgType));
        FunctionType *NewFT = FunctionType::get(
            getWithoutAS(*FT->getReturnType()), ArgTypes, FT->isVarArg());
        CI.mutateFunctionType(NewFT);
        CI.mutateType(getWithoutAS(*CI.getType()));
      } else {
        assert(!isASType(*CI.getType()) && "TODO");
        IRBuilder<> IRB(&CI);
        for (unsigned I = 0, E = CI.arg_size(); I < E; ++I) {
          auto *Op = CI.getArgOperand(I);
          if (!(Op->getType()->isPointerTy()))
            continue;
          auto *GenericOp = GetAsGeneric(*Op);
          Value *NewOp = createCall(IRB, getUnpackFn(), {getPC(IRB), GenericOp},
                                    GenericOp->getName() + ".unpack");
          if (auto AS = Op->getType()->getPointerAddressSpace())
            NewOp = IRB.CreateAddrSpaceCast(NewOp, Op->getType());
          CI.setArgOperand(I, NewOp);
        }
      }
      break;
    }
    default:
      I->dump();
      llvm_unreachable("Instruction with AS not handled");
    }
    if (VMap.count(I))
      errs() << "I: " << *I << " --> " << *VMap[I] << "\n";
  }

  for (auto *PHI : PHIs)
    for (unsigned I = 0, E = PHI->getNumIncomingValues(); I < E; ++I)
      PHI->setIncomingValue(I, GetAsGeneric(*PHI->getIncomingValue(I)));
}

void OffloadSanitizerImpl::instrumentCallInsts(
    SmallVectorImpl<CallInst *> &CallInsts) {
  for (auto *CI : CallInsts) {
    if (isa<LifetimeIntrinsic>(CI))
      continue;
    auto *Fn = CI->getCalledFunction();
    if (!Fn)
      continue;
    if (Fn->getName().starts_with("__kmpc_target_init"))
      continue;
    if ((Fn->isDeclaration() || Fn->getName().starts_with("__kmpc") ||
         Fn->getName().starts_with("rpc_")) &&
        !Fn->getName().starts_with("ompx")) {
      IRBuilder<> IRB(CI);
      for (int I = 0, E = CI->arg_size(); I != E; ++I) {
        Value *Op = CI->getArgOperand(I);
        if (!Op->getType()->isPointerTy())
          continue;
        auto *CB = createCall(IRB, getUnpackFn(), {getPC(IRB), Op},
                              Op->getName() + ".unpack");
        CI->setArgOperand(I, CB);
      }
    }
  }
}

void OffloadSanitizerImpl::instrumentLifetimeIntrinsics(
    SmallVectorImpl<LifetimeIntrinsic *> &LifetimeInsts) {
  for (auto *LI : LifetimeInsts)
    LI->eraseFromParent();
}

void OffloadSanitizerImpl::instrumentTrapInstructions(
    SmallVectorImpl<IntrinsicInst *> &TrapCalls) {
  for (auto *II : TrapCalls) {
    IRBuilder<> IRB(II);
    createCall(IRB, getTrapInfoFn(), {getPC(IRB)});
  }
}

void OffloadSanitizerImpl::instrumentUnreachableInstructions(
    SmallVectorImpl<UnreachableInst *> &UnreachableInsts) {
  for (auto *II : UnreachableInsts) {
    // Skip unreachables after traps since we instrument those as well.
    if (&II->getParent()->front() != II)
      if (auto *CI = dyn_cast<CallInst>(II->getPrevNode()))
        if (CI->getIntrinsicID() == Intrinsic::trap)
          continue;
    IRBuilder<> IRB(II);
    createCall(IRB, getUnreachableInfoFn(), {getPC(IRB)});
  }
}

void OffloadSanitizerImpl::instrumentAccesses(
    SmallVectorImpl<AccessInfoTy> &AccessInfos) {
  for (auto &AI : AccessInfos) {
    IRBuilder<> IRB(AI.I);
    auto *FakePtr = AI.I->getOperand(AI.PtrOpIdx);
    auto *Size =
        ConstantInt::get(Int32Ty, DL.getTypeStoreSize(AI.I->getAccessType()));
    errs() << "AI " << AI.I << "\n";
    AI.I->dump();
    FakePtr->dump();
    if (FakePtr->getType()->getPointerAddressSpace()) {
      auto *ASC = cast<AddrSpaceCastInst>(FakePtr);
      FakePtr = ASC->getPointerOperand();
    }
    assert(FakePtr->getType()->getPointerAddressSpace() == 0);
    auto *RealPtr =
        createCall(IRB, getCheckAccessFn(AI.AS), {getPC(IRB), FakePtr, Size});
    AI.I->setOperand(AI.PtrOpIdx, RealPtr);
    assert(RealPtr->getParent());
  }
}

void OffloadSanitizerImpl::instrumentAllocaInstructions(
    SmallVectorImpl<AllocaInst *> &AllocaInsts) {

  auto IsApplicable = [&](AllocaInst &AI, TypeSize &TS) {
    // Check the type and size.
    if (AI.getAllocatedType()->isScalableTy())
      return false;
    auto AllocSize = AI.getAllocationSize(DL);
    assert(AllocSize && "Alloc size not known!");
    if (AllocSize->getKnownMinValue() >= (1UL << 32))
      return false;
    TS = *AllocSize;
    return true;
  };

  for (auto *AI : AllocaInsts) {
    TypeSize TS(0, false);
    if (!IsApplicable(*AI, TS))
      continue;
    AI->dump();

    IRBuilder<> IRB(AI->getNextNode());
    auto *Size = ConstantInt::get(Int32Ty, TS);
    auto *FakePtr =
        createCall(IRB, getAllocaRegisterFn(), {getPC(IRB), AI, Size});
    for (auto *U : AI->users()) {
      auto *UI = cast<Instruction>(U);
      if (UI == FakePtr)
        continue;
      if (isa<MemIntrinsic>(UI))
        continue;
      assert(isa<AddrSpaceCastInst>(UI) &&
             "Expected only address space casts users of allocas");
      assert(UI->getType()->getPointerAddressSpace() == 0 &&
             "Expected only address space casts to AS 0 as users of allocas");
      UI->replaceAllUsesWith(FakePtr);
    }
  }
}

bool OffloadSanitizerImpl::instrumentFunction(Function &Fn) {
  if (!shouldInstrumentFunction(&Fn))
    return false;

  SmallVector<UnreachableInst *> UnreachableInsts;
  SmallVector<IntrinsicInst *> TrapCalls;
  SmallVector<AllocaInst *> AllocaInsts;
  SmallVector<AccessInfoTy> AccessInfos;
  SmallVector<Instruction *> ASInsts;
  SmallVector<LifetimeIntrinsic *> LifetimeInsts;
  SmallVector<CallInst *> CallInsts;
  SmallVector<AddrSpaceCastInst *> ASCInsts;

  ReversePostOrderTraversal<Function *> RPOT(&Fn);
  for (auto &It : RPOT) {
    for (auto &I : *It) {
      if (!I.getType()->isVoidTy())
        I.setName("I");
      switch (I.getOpcode()) {
      case Instruction::Alloca:
        AllocaInsts.push_back(cast<AllocaInst>(&I));
        break;
      case Instruction::Store: {
        auto &SI = cast<StoreInst>(I);
        errs() << &SI << " :" << SI << "\n";
        AccessInfos.push_back(
            {&I, SI.getPointerOperandIndex(), SI.getPointerAddressSpace()});
        if (isASType(*SI.getPointerOperandType()))
          ASInsts.push_back(&I);
        break;
      }
      case Instruction::Load: {
        auto &LI = cast<LoadInst>(I);
        errs() << &LI << " :" << LI << "\n";
        AccessInfos.push_back(
            {&I, LI.getPointerOperandIndex(), LI.getPointerAddressSpace()});
        if (isASType(*LI.getType()) || isASType(*LI.getPointerOperandType()))
          ASInsts.push_back(&I);
        break;
      }
      case Instruction::AtomicRMW: {
        auto &ARMW = cast<AtomicRMWInst>(I);
        if (isASType(*ARMW.getType()) ||
            isASType(*ARMW.getPointerOperand()->getType()))
          ASInsts.push_back(&I);
        break;
      }
      case Instruction::Unreachable:
        UnreachableInsts.push_back(cast<UnreachableInst>(&I));
        break;
      case Instruction::Call: {
        auto &CI = cast<CallInst>(I);
        if (auto *II = dyn_cast<IntrinsicInst>(&CI)) {
          switch (II->getIntrinsicID()) {
          case Intrinsic::trap:
            TrapCalls.push_back(II);
            break;
          case Intrinsic::lifetime_start:
          case Intrinsic::lifetime_end:
            LifetimeInsts.push_back(cast<LifetimeIntrinsic>(II));
            break;
          }
        } else {
          if (isASType(*CI.getType()))
            ASInsts.push_back(&I);
          else if (any_of(CI.args(),
                          [&](Value *Op) { return isASType(*Op->getType()); }))
            ASInsts.push_back(&I);
          CallInsts.push_back(&CI);
        }
        break;
      }
      case Instruction::AddrSpaceCast:
        ASCInsts.push_back(cast<AddrSpaceCastInst>(&I));
        ASInsts.push_back(&I);
        break;
      default:
        if (isASType(*I.getType()))
          ASInsts.push_back(&I);
        else if (any_of(I.operand_values(),
                        [&](Value *Op) { return isASType(*Op->getType()); }))
          ASInsts.push_back(&I);
        break;
      }
    }
  }

  Fn.dump();
  removeAS(Fn, ASInsts);
  Fn.dump();
  //  instrumentCallInsts(CallInsts);
  instrumentLifetimeIntrinsics(LifetimeInsts);
  //  instrumentTrapInstructions(TrapCalls);
  //  instrumentUnreachableInstructions(UnreachableInsts);
  instrumentAccesses(AccessInfos);
  Fn.dump();
  instrumentAllocaInstructions(AllocaInsts);

  for (auto *ASC : ASCInsts) {
    if (ASC->getPointerOperand()->getType() == ASC->getType())
      ASC->replaceAllUsesWith(ASC->getPointerOperand());
    if (ASC->use_empty())
      ASC->eraseFromParent();
  }

  //  for (auto *CI : RTCalls) {
  //    InlineFunctionInfo IFI;
  //    InlineFunction(*CI, IFI);
  //  }
  RTCalls.clear();

  return true;
}

bool OffloadSanitizerImpl::instrument() {
  bool Changed = false;

  for (Function &Fn : M)
    Changed |= instrumentFunction(Fn);

  removeFromUsedLists(M, [&](Constant *C) {
    if (!C->getName().starts_with("__offload_san"))
      return false;
    return Changed = true;
  });

  return Changed;
}

PreservedAnalyses OffloadSanitizerPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  OffloadSanitizerImpl Impl(M, FAM);
  if (!Impl.instrument())
    return PreservedAnalyses::all();
  M.dump();
  assert(!verifyModule(M, &errs()));
  return PreservedAnalyses::none();
}
