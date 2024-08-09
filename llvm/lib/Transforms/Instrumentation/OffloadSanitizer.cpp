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

  bool shouldInstrumentFunction(Function &Fn);

  struct AccessInfoTy {
    Instruction *I;
    unsigned PtrOpIdx;
    unsigned AS;
  };

  void removeASFromUses(SmallVectorImpl<const Use *> &Uses,
                        DenseMap<Value *, Value *> &VMap);

  bool instrumentFunction(Function &Fn);
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
  DenseMap<Value *, Value *> VMap;

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

void OffloadSanitizerImpl::removeASFromUses(SmallVectorImpl<const Use *> &Uses,
                                            DenseMap<Value *, Value *> &VMap) {
  auto Clone = [&](Instruction &I) {
    Instruction *NewI = I.clone();
    NewI->setName(I.getName() + ".noas");
    NewI->insertAfter(&I);
    NewI->mutateType(getWithoutAS(*I.getType()));
    VMap[&I] = NewI;
    return NewI;
  };

  auto GetCasted = [&](Value &Op, Instruction &I) {
    Instruction *IP = nullptr;
    if (auto *PHIOp = dyn_cast<PHINode>(&Op))
      IP = PHIOp->getParent()->getFirstNonPHI();
    else if (auto *OpI = dyn_cast<Instruction>(&Op))
      IP = OpI->getNextNode();
    else
      IP = &I.getFunction()->getEntryBlock().front();
    auto *ACS = new AddrSpaceCastInst(&Op, getWithoutAS(*Op.getType()),
                                      Op.getName() + ".noas", IP);
    VMap[&Op] = ACS;
    return ACS;
  };

  for (auto *U : Uses) {
    auto *I = cast<Instruction>(U->getUser());
    errs() << "I: " << *I << "\n";
    if (VMap.count(I))
      continue;
    switch (I->getOpcode()) {
    case Instruction::Load: {
      auto &LI = cast<LoadInst>(*I);
      LI.setOperand(LI.getPointerOperandIndex(), VMap[LI.getPointerOperand()]);
      VMap[I] = I;
      break;
    }
    case Instruction::Store: {
      auto &SI = cast<StoreInst>(*I);
      SI.setOperand(SI.getPointerOperandIndex(), VMap[SI.getPointerOperand()]);
      VMap[I] = I;
      break;
    }
    case Instruction::GetElementPtr: {
      auto &NewGEP = cast<GetElementPtrInst>(*Clone(*I));
      NewGEP.setSourceElementType(getWithoutAS(*NewGEP.getSourceElementType()));
      NewGEP.setResultElementType(getWithoutAS(*NewGEP.getResultElementType()));
      NewGEP.setOperand(NewGEP.getPointerOperandIndex(),
                        VMap[NewGEP.getPointerOperand()]);
      break;
    }
    case Instruction::AddrSpaceCast: {
      auto &ASC = cast<AddrSpaceCastInst>(*I);
      VMap[I] = VMap[ASC.getPointerOperand()];
      break;
    }
    case Instruction::Select: {
      auto &NewSI = cast<SelectInst>(*Clone(*I));
      if (auto *NewTV = VMap.lookup(NewSI.getTrueValue()))
        NewSI.setTrueValue(NewTV);
      else
        NewSI.setTrueValue(GetCasted(*NewSI.getTrueValue(), NewSI));
      if (auto *NewFV = VMap.lookup(NewSI.getFalseValue()))
        NewSI.setFalseValue(NewFV);
      else
        NewSI.setFalseValue(GetCasted(*NewSI.getFalseValue(), NewSI));
      break;
    }
    case Instruction::PHI: {
      auto &NewPHI = cast<PHINode>(*Clone(*I));
      for (unsigned I = 0, E = NewPHI.getNumIncomingValues(); I < E; ++I) {
        if (auto *NewV = VMap.lookup(NewPHI.getIncomingValue(I)))
          NewPHI.setIncomingValue(I, NewV);
        else
          NewPHI.setIncomingValue(
              I, GetCasted(*NewPHI.getIncomingValue(I), NewPHI));
      }
      break;
    }
    case Instruction::Call: {
      break;
    }
    default:
      I->dump();
      llvm_unreachable("Instruction with AS not handled");
    }
    if (VMap.count(I))
      errs() << "I: " << *I << " --> " << *VMap[I] << "\n";
  }
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
    auto *FakePtr = AI.I->getOperand(AI.PtrOpIdx);
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

  SmallVector<const Use *> Worklist, Visited;
  auto IsApplicable = [&](AllocaInst &AI, TypeSize &TS) {
    // Check the type and size.
    if (AI.getAllocatedType()->isScalableTy())
      return false;
    auto AllocSize = AI.getAllocationSize(DL);
    assert(AllocSize && "Alloc size not known!");
    if (AllocSize->getKnownMinValue() >= (1UL << 32))
      return false;
    TS = *AllocSize;

    // Now we check the users.
    Visited.clear();
    Worklist.clear();
    for (auto &U : AI.uses())
      Worklist.push_back(&U);
    errs() << "AI: " << AI << "\n";
    while (!Worklist.empty()) {
      auto *U = Worklist.pop_back_val();
      auto *I = cast<Instruction>(U->getUser());
      errs() << "WI: " << *I << "\n";
      Visited.push_back(U);
      switch (I->getOpcode()) {
      case Instruction::Load:
        break;
      case Instruction::GetElementPtr:
      case Instruction::AddrSpaceCast:
      case Instruction::Select:
      case Instruction::PHI:
        for (auto &U : I->uses())
          Worklist.push_back(&U);
        break;
      case Instruction::Call: {
        break;
      }
      case Instruction::Store:
        if (cast<StoreInst>(I)->getValueOperand() == U->get())
          return false;
        break;
      default:
        return false;
      }
    }
    return true;
  };

  bool Changed = false;
  for (auto *AI : AllocaInsts) {
    TypeSize TS(0, false);
    if (!IsApplicable(*AI, TS))
      continue;
    Changed = true;

    IRBuilder<> IRB(AI->getNextNode());
    auto *Size = ConstantInt::get(Int32Ty, TS);
    auto *FakePtr =
        createCall(IRB, getAllocaRegisterFn(), {getPC(IRB), AI, Size});
    VMap[AI] = FakePtr;
    removeASFromUses(Visited, VMap);
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
