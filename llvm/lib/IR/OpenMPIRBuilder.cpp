//===- OpenMPIRBuilder.cpp - Builder for LLVM-IR for OpenMP directives ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the OpenMPIRBuilder class, which is used as a
/// convenient way to create LLVM instructions for OpenMP directives.
///
//===----------------------------------------------------------------------===//

#include "llvm/IR/OpenMPIRBuilder.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/IRBuilder.h"

#define DEBUG_TYPE "openmp-ir-builder"

using namespace llvm;
using namespace omp;

Function *OpenMPIRBuilder::getOrCreateRuntimeFunction(RuntimeFunction FnID) {
  Function *Fn = nullptr;

  // Try to find the declation in the module first.
  switch (FnID) {
#define OMP_RTL(Enum, Str, IsVarArg, ReturnType, ...)                          \
  case Enum:                                                                   \
    Fn = M.getFunction(Str);                                                   \
    break;
#include "llvm/IR/OpenMPKinds.def"
  }

  if (!Fn) {

    // Create a new declaration if we need one.
    switch (FnID) {
#define OMP_RTL(Enum, Str, IsVarArg, ReturnType, ...)                          \
  case Enum:                                                                   \
    Fn = Function::Create(FunctionType::get(ReturnType,                        \
                                            ArrayRef<Type *>{__VA_ARGS__},     \
                                            IsVarArg),                         \
                          GlobalValue::ExternalLinkage, Str, M);               \
    break;
#include "llvm/IR/OpenMPKinds.def"
    }

    assert(Fn && "Failed to create OpenMP runtime function");

    LLVMContext &Ctx = Fn->getContext();
    // Add attributes to the new declaration.
    switch (FnID) {
#define OMP_RTL_ATTRS(Enum, FnAttrSet, RetAttrSet, ArgAttrSets)                \
  case Enum:                                                                   \
    Fn->setAttributes(                                                         \
        AttributeList::get(Ctx, FnAttrSet, RetAttrSet, ArgAttrSets));          \
    break;
#include "llvm/IR/OpenMPKinds.def"
    default:
      // Attributes are optional.
      break;
    }
  }

  return Fn;
}

void OpenMPIRBuilder::initialize() {
  LLVMContext &Ctx = M.getContext();

  // Create all simple and struct types exposed by the runtime and remember the
  // llvm::PointerTypes of them for easy access later.
  Type *T;
#define OMP_TYPE(VarName, InitValue) this->VarName = InitValue;
#define OMP_STRUCT_TYPE(VarName, StructName, ...)                              \
  T = M.getTypeByName(StructName);                                             \
  if (!T)                                                                      \
    T = StructType::create(Ctx, {__VA_ARGS__}, StructName);                    \
  this->VarName = PointerType::getUnqual(T);
#include "llvm/IR/OpenMPKinds.def"
}

Value *OpenMPIRBuilder::getOrCreateIdent(Constant *SrcLocStr,
                                         IdentFlag LocFlags) {
  // Enable "C-mode".
  LocFlags |= OMP_IDENT_FLAG_KMPC;

  GlobalVariable *&DefaultIdent = IdentMap[{SrcLocStr, uint64_t(LocFlags)}];
  if (!DefaultIdent) {
    Constant *I32Null = ConstantInt::getNullValue(Int32);
    Constant *IdentData[] = {I32Null,
                             ConstantInt::get(Int32, uint64_t(LocFlags)),
                             I32Null, I32Null, SrcLocStr};
    Constant *Initializer = ConstantStruct::get(
        cast<StructType>(IdentPtr->getPointerElementType()), IdentData);

    // Look for existing encoding of the location + flags, not needed but
    // minimizes the difference to the existing solution while we transition.
    for (GlobalVariable &GV : M.getGlobalList())
      if (GV.getType() == IdentPtr && GV.hasInitializer())
        if (GV.getInitializer() == Initializer)
          return DefaultIdent = &GV;

    DefaultIdent = new GlobalVariable(M, IdentPtr->getPointerElementType(),
                                      /* isConstant = */ false,
                                      GlobalValue::PrivateLinkage, Initializer);
    DefaultIdent->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    DefaultIdent->setAlignment(Align(8));
  }
  return DefaultIdent;
}

Constant *OpenMPIRBuilder::getOrCreateSrcLocStr(StringRef LocStr) {
  Constant *&SrcLocStr = SrcLocStrMap[LocStr];
  if (!SrcLocStr) {
    Constant *Initializer =
        ConstantDataArray::getString(M.getContext(), LocStr);

    // Look for existing encoding of the location, not needed but minimizes the
    // difference to the existing solution while we transition.
    for (GlobalVariable &GV : M.getGlobalList())
      if (GV.isConstant() && GV.hasInitializer() &&
          GV.getInitializer() == Initializer)
        return SrcLocStr = ConstantExpr::getPointerCast(&GV, Int8Ptr);

    SrcLocStr = Builder.CreateGlobalStringPtr(LocStr);
  }
  return SrcLocStr;
}

Constant *OpenMPIRBuilder::getOrCreateDefaultSrcLocStr() {
  return getOrCreateSrcLocStr(";unknown;unknown;0;0;;");
}

Constant *
OpenMPIRBuilder::getOrCreateSrcLocStr(const LocationDescription &Loc) {
  // TODO: Support actual source locations.
  return getOrCreateDefaultSrcLocStr();
}

Value *OpenMPIRBuilder::getOrCreateThreadID(Value *Ident) {
  // TODO: It makes only so much sense to actually cache the global_thread_num
  //       calls in the front-end as we can do a better job later on. Once
  //       the middle-end combines global_thread_num calls (user calls and
  //       generated ones!) we can rethink having a caching scheme here.
  Function *Fn = Builder.GetInsertBlock()->getParent();
  Value *&TID = ThreadIDMap[Fn];
  if (!TID) {
    // Search the entry block, not needed once all thread id calls go through
    // here and are cached in the OpenMPIRBuilder.
    for (Instruction &I : Fn->getEntryBlock())
      if (CallInst *CI = dyn_cast<CallInst>(&I))
        if (CI->getCalledFunction() &&
            CI->getCalledFunction()->getName() == "__kmpc_global_thread_num")
          return TID = CI;

    Function *FnDecl =
        getOrCreateRuntimeFunction(OMPRTL___kmpc_global_thread_num);
    Instruction *Call =
        Builder.CreateCall(FnDecl, Ident, "omp_global_thread_num");
    if (Instruction *IdentI = dyn_cast<Instruction>(Ident))
      Call->moveAfter(IdentI);
    else
      Call->moveBefore(&*Fn->getEntryBlock().getFirstInsertionPt());
    TID = Call;
  }
  return TID;
}

void OpenMPIRBuilder::CreateBarrier(const LocationDescription &Loc,
                                    Directive DK, bool ForceSimpleCall,
                                    bool CheckCancelFlag) {
  // TODO: Do we really expect these create calls to happen at an invalid
  //       location and if so is ignoring them the right thing to do? This
  //       mimics Clang's behavior for now.
  if (!Loc.IP.getBlock())
    return;
  Builder.restoreIP(Loc.IP);
  emitBarrierImpl(Loc, DK, ForceSimpleCall, CheckCancelFlag);
}

void OpenMPIRBuilder::emitBarrierImpl(const LocationDescription &Loc,
                                      Directive Kind, bool ForceSimpleCall,
                                      bool CheckCancelFlag) {
  // Build call __kmpc_cancel_barrier(loc, thread_id) or
  //            __kmpc_barrier(loc, thread_id);

  IdentFlag BarrierLocFlags;
  switch (Kind) {
  case OMPD_for:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL_FOR;
    break;
  case OMPD_sections:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL_SECTIONS;
    break;
  case OMPD_single:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL_SINGLE;
    break;
  case OMPD_barrier:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_EXPL;
    break;
  default:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL;
    break;
  }

  // Set new insertion point for the internal builder.
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Args[] = {getOrCreateIdent(SrcLocStr, BarrierLocFlags),
                   getOrCreateThreadID(getOrCreateIdent(SrcLocStr))};
  bool UseCancelBarrier = !ForceSimpleCall && CancelationBlock;
  Value *Result = Builder.CreateCall(
      getOrCreateRuntimeFunction(UseCancelBarrier ? OMPRTL___kmpc_cancel_barrier
                                                  : OMPRTL___kmpc_barrier),
      Args);

  if (UseCancelBarrier && CheckCancelFlag) {
    Value *Cmp = Builder.CreateIsNotNull(Result);
    // TODO Reimplement part of llvm::SplitBlockAndInsertIfThen in a helper and
    // use it here.
    (void)Cmp;
  }
}
