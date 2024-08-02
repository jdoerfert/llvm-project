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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
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
  Value *stripASCasts(Value *V) {
    while (auto *ASC = dyn_cast<AddrSpaceCastInst>(V))
      V = ASC->getPointerOperand();
    return V;
  }

  bool shouldInstrumentFunction(Function &Fn);

  struct AccessInfoTy {
    Instruction *I;
    unsigned PtrOpIdx;
    unsigned AS;
  };

  bool instrumentFunction(Function &Fn);
  bool clearAS(Function &Fn, SmallVectorImpl<Instruction *> &ASInsts);
  bool instrumentTrapInstructions(SmallVectorImpl<IntrinsicInst *> &TrapCalls);
  bool instrumentCallInsts(SmallVectorImpl<CallInst *> &CallInsts);
  bool instrumentLifetimeIntrinsics(
      SmallVectorImpl<LifetimeIntrinsic *> &LifetimeInsts);
  bool instrumentUnreachableInstructions(
      SmallVectorImpl<UnreachableInst *> &UnreachableInsts);
  bool instrumentAccesses(SmallVectorImpl<AccessInfoTy> &Accesses);
  bool instrumentAllocaInstructions(SmallVectorImpl<AllocaInst *> &AllocaInsts);

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
    Calls.push_back(IRB.CreateCall(Callee, Args, Name));
    return Calls.back();
  }
  SmallVector<CallInst *> Calls;

  Value *getPC(IRBuilder<> &IRB) {
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

bool OffloadSanitizerImpl::shouldInstrumentFunction(Function &Fn) {
  if (Fn.isDeclaration())
    return false;
  if (Fn.getName().contains("ompx") || Fn.getName().contains("__kmpc") ||
      Fn.getName().starts_with("rpc_"))
    return false;
  return !Fn.hasFnAttribute(Attribute::DisableSanitizerInstrumentation);
}

bool OffloadSanitizerImpl::clearAS(Function &Fn,
                                   SmallVectorImpl<Instruction *> &ASInsts) {
  DenseMap<Value *, Value *> VMap;

  auto HandleRoot = [&](Value *Root) {
    if (!isASType(*Root->getType()))
      return Root;
    Value *&NewRoot = VMap[Root];
    if (!NewRoot) {
      Instruction *IP = dyn_cast<Instruction>(Root);
      if (IP)
        IP = IP->getNextNode();
      else
        IP = &Fn.getEntryBlock().front();
      NewRoot = new AddrSpaceCastInst(Root, getWithoutAS(*Root->getType()),
                                      Root->getName() + ".noas", IP);
    }
    return NewRoot;
  };

  for (auto *I : ASInsts) {
    switch (I->getOpcode()) {
    case Instruction::GetElementPtr: {
      auto &GEP = cast<GetElementPtrInst>(*I);
      GEP.dump();
      GEP.setSourceElementType(getWithoutAS(*GEP.getSourceElementType()));
      GEP.setResultElementType(getWithoutAS(*GEP.getResultElementType()));
      auto *Op = HandleRoot(GEP.getPointerOperand());
      GEP.setOperand(GEP.getPointerOperandIndex(), Op);
      auto *Ty = GEP.getType();
      auto *TyNoAS = getWithoutAS(*Ty);
      auto *ACS = new AddrSpaceCastInst(UndefValue::get(TyNoAS), Ty, "",
                                        GEP.getNextNode());
      GEP.replaceAllUsesWith(ACS);
      GEP.mutateType(TyNoAS);
      ACS->setOperand(ACS->getPointerOperandIndex(), &GEP);
      GEP.dump();
      break;
    }
    case Instruction::AddrSpaceCast: {
      auto &ASC = cast<AddrSpaceCastInst>(*I);
      break;
    }
    default:
      I->dump();
      llvm_unreachable("Instruction with AS not handled");
    }
  }
  return !ASInsts.empty();
}

bool OffloadSanitizerImpl::instrumentCallInsts(
    SmallVectorImpl<CallInst *> &CallInsts) {
  bool Changed = false;
  for (auto *CI : CallInsts) {
    if (isa<LifetimeIntrinsic>(CI))
      return Changed;
    auto *Fn = CI->getCalledFunction();
    if (!Fn)
      continue;
    if (Fn->getName().starts_with("__kmpc_target_init"))
      return Changed;
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
        Changed = true;
      }
    }
  }
  return Changed;
}

bool OffloadSanitizerImpl::instrumentLifetimeIntrinsics(
    SmallVectorImpl<LifetimeIntrinsic *> &LifetimeInsts) {
  for (auto *LI : LifetimeInsts)
    LI->eraseFromParent();
  return !LifetimeInsts.empty();
}

bool OffloadSanitizerImpl::instrumentTrapInstructions(
    SmallVectorImpl<IntrinsicInst *> &TrapCalls) {
  bool Changed = false;
  for (auto *II : TrapCalls) {
    IRBuilder<> IRB(II);
    createCall(IRB, getTrapInfoFn(), {getPC(IRB)});
  }
  return Changed;
}

bool OffloadSanitizerImpl::instrumentUnreachableInstructions(
    SmallVectorImpl<UnreachableInst *> &UnreachableInsts) {
  bool Changed = false;
  for (auto *II : UnreachableInsts) {
    // Skip unreachables after traps since we instrument those as well.
    if (&II->getParent()->front() != II)
      if (auto *CI = dyn_cast<CallInst>(II->getPrevNode()))
        if (CI->getIntrinsicID() == Intrinsic::trap)
          continue;
    IRBuilder<> IRB(II);
    createCall(IRB, getUnreachableInfoFn(), {getPC(IRB)});
  }
  return Changed;
}

bool OffloadSanitizerImpl::instrumentAccesses(
    SmallVectorImpl<AccessInfoTy> &AccessInfos) {
  bool Changed = false;
  for (auto &AI : AccessInfos) {
    IRBuilder<> IRB(AI.I);
    auto *FakePtr = stripASCasts(AI.I->getOperand(AI.PtrOpIdx));
    auto *Size =
        ConstantInt::get(Int32Ty, DL.getTypeStoreSize(AI.I->getAccessType()));
    AI.I->dump();
    FakePtr->dump();
    if (FakePtr->getType()->getPointerAddressSpace())
      FakePtr = IRB.CreateAddrSpaceCast(FakePtr, PtrTy);
    FakePtr->dump();
    auto *RealPtr =
        createCall(IRB, getCheckAccessFn(AI.AS), {getPC(IRB), FakePtr, Size});
    AI.I->setOperand(AI.PtrOpIdx, RealPtr);
  }
  return Changed;
}

bool OffloadSanitizerImpl::instrumentAllocaInstructions(
    SmallVectorImpl<AllocaInst *> &AllocaInsts) {
  bool Changed = false;
  for (auto *AI : AllocaInsts) {
    assert(!AI->getAllocatedType()->isScalableTy());
    auto AllocSize = AI->getAllocationSize(DL);
    assert(AllocSize && "Alloc size not known!");
    assert((AllocSize->getKnownMinValue() < (1UL << 32)) &&
           "Alloc size too large!");

    IRBuilder<> IRB(AI->getNextNode());
    auto *Size = ConstantInt::get(Int32Ty, *AllocSize);
    auto *FakePtr =
        createCall(IRB, getAllocaRegisterFn(), {getPC(IRB), AI, Size});
    auto *NewPtr = IRB.CreateAddrSpaceCast(FakePtr, AI->getType());
    AI->replaceUsesWithIf(NewPtr,
                          [=](Use &U) { return U.getUser() != FakePtr; });
  }
  return Changed;
}

bool OffloadSanitizerImpl::instrumentFunction(Function &Fn) {
  if (!shouldInstrumentFunction(Fn))
    return false;

  SmallVector<UnreachableInst *> UnreachableInsts;
  SmallVector<IntrinsicInst *> TrapCalls;
  SmallVector<AllocaInst *> AllocaInsts;
  SmallVector<AccessInfoTy> AccessInfos;
  SmallVector<Instruction *> ASInsts;
  SmallVector<LifetimeIntrinsic *> LifetimeInsts;
  SmallVector<CallInst *> CallInsts;

  for (auto &Arg : Fn.args())
    if (isASType(*Arg.getType())) {
    }

  bool Changed = false;
  for (auto &I : instructions(Fn)) {
    switch (I.getOpcode()) {
    case Instruction::Alloca:
      AllocaInsts.push_back(cast<AllocaInst>(&I));
      break;
    case Instruction::Store: {
      auto &SI = cast<StoreInst>(I);
      AccessInfos.push_back(
          {&I, SI.getPointerOperandIndex(), SI.getPointerAddressSpace()});
      break;
    }
    case Instruction::Load: {
      auto &LI = cast<LoadInst>(I);
      AccessInfos.push_back(
          {&I, LI.getPointerOperandIndex(), LI.getPointerAddressSpace()});
      if (isASType(*LI.getType()))
        LI.mutateType(getWithoutAS(*LI.getType()));
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
        CallInsts.push_back(&CI);
      }
      //      if (isASType(*CI.getType()) == 4)
      //        break;
      //      assert(!isASType(*CI.getType()) && "TODO");
      break;
    }
    default:
      //      if (any_of(I.operands(),
      //                 [&](auto &Op) { return isASType(*Op.getType()); }))
      //        break;
      if (isASType(*I.getType()))
        ASInsts.push_back(&I);
      break;
    }
  }

  //  Changed |= clearAS(Fn, ASInsts);
  Changed |= instrumentCallInsts(CallInsts);
  Changed |= instrumentLifetimeIntrinsics(LifetimeInsts);
  Changed |= instrumentTrapInstructions(TrapCalls);
  Changed |= instrumentUnreachableInstructions(UnreachableInsts);
  Changed |= instrumentAccesses(AccessInfos);
  Changed |= instrumentAllocaInstructions(AllocaInsts);
  Fn.dump();

  return Changed;
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
  LLVM_DEBUG(M.dump());
  return PreservedAnalyses::none();
}
