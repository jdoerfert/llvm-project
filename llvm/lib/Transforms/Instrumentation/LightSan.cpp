//===-- LightSan.cpp - Sanitization instrumentation pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/LightSan.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Transforms/Instrumentation/Instrumentor.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <cstdint>
#include <functional>
#include <optional>

using namespace llvm;
using namespace llvm::instrumentor;

#define DEBUG_TYPE "lightsan"

static constexpr char LightSanRuntimePrefix[] = "__lightsan_";

namespace {

struct LightSanImpl;

struct LightSanInstrumentationConfig : public InstrumentationConfig {

  LightSanInstrumentationConfig(LightSanImpl &LSI);
  virtual ~LightSanInstrumentationConfig() {}

  void populate(InstrumentorIRBuilderTy &IRB) override;

  LightSanImpl &LSI;
};

struct LightSanImpl {
  LightSanImpl(Module &M, ModuleAnalysisManager &MAM)
      : M(M), MAM(MAM), IConf(*this) {}

  bool instrument();

  bool shouldInstrumentCall(CallInst &CI);
  bool shouldInstrumentLoad(LoadInst &LI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentStore(StoreInst &SI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentAlloca(AllocaInst &AI, InstrumentorIRBuilderTy &IIRB);

private:
  Module &M;
  ModuleAnalysisManager &MAM;
  LightSanInstrumentationConfig IConf;
  const DataLayout &DL = M.getDataLayout();
};

bool LightSanImpl::shouldInstrumentLoad(LoadInst &LI,
                                              InstrumentorIRBuilderTy &IIRB) {
  if (auto *AI = dyn_cast<AllocaInst>(LI.getPointerOperand()))
    return shouldInstrumentAlloca(*AI, IIRB);
  return true;
}

bool LightSanImpl::shouldInstrumentStore(StoreInst &SI,
                                               InstrumentorIRBuilderTy &IIRB) {
  if (auto *AI = dyn_cast<AllocaInst>(SI.getPointerOperand()))
    return shouldInstrumentAlloca(*AI, IIRB);
  return true;
}

bool LightSanImpl::shouldInstrumentAlloca(AllocaInst &AI,
                                                InstrumentorIRBuilderTy &IIRB) {
  // TODO: look trough transitive users.
  auto IsUseOK = [&](Use &U) -> bool {
    if (auto *SI = dyn_cast<StoreInst>(U.getUser())) {
      if (SI->getPointerOperandIndex() == U.getOperandNo() &&
          AI.getAllocationSize(DL) >=
              DL.getTypeStoreSize(SI->getValueOperand()->getType()))
        return false;
    }
    if (auto *LI = dyn_cast<LoadInst>(U.getUser())) {
      if (LI->getPointerOperandIndex() == U.getOperandNo() &&
          AI.getAllocationSize(DL) >= DL.getTypeStoreSize(LI->getType()))
        return false;
    }
    return true;
  };
  return all_of(AI.uses(), IsUseOK);
}

bool LightSanImpl::shouldInstrumentCall(CallInst &CI) {
  if (CI.getCaller()->getName().starts_with(IConf.getRTName()) &&
      !CI.getCaller()->hasFnAttribute("instrument"))
    return false;
  return true;
}

bool LightSanImpl::instrument() {
  bool Changed = false;

  InstrumentorPass IP(&IConf);

  auto PA = IP.run(M, MAM);
  if (!PA.areAllPreserved())
    Changed = true;

  return Changed;
}

LightSanInstrumentationConfig::LightSanInstrumentationConfig(LightSanImpl &Impl)
    : InstrumentationConfig(), LSI(Impl) {
  ReadConfig = false;
  RuntimePrefix->setString(LightSanRuntimePrefix);
  RuntimeStubsFile->setString("");
}

void LightSanInstrumentationConfig::populate(InstrumentorIRBuilderTy &IIRB) {
  UnreachableIO::populate(*this, IIRB.Ctx);
  BasePointerIO::populate(*this, IIRB.Ctx);
  ModuleIO::populate(*this, IIRB.Ctx);
  GlobalIO::populate(*this, IIRB.Ctx);

  auto *AIC = InstrumentationConfig::allocate<AllocaIO>(/*IsPRE=*/false);
  AIC->CB = [&](Value &V) {
    return LSI.shouldInstrumentAlloca(cast<AllocaInst>(V), IIRB);
  };
  AIC->init(*this, IIRB.Ctx, /*ReplaceAddr=*/true, /*ReplaceSize=*/false);

  LoadIO::ConfigTy LICConfig;
  LICConfig.PassPointerAS = false;
  LICConfig.PassLoopValueRangeInfo = false;
  LICConfig.PassValue = false;
  LICConfig.ReplaceValue = false;
  LICConfig.PassAlignment = false;
  LICConfig.PassValueTypeId = false;
  LICConfig.PassAtomicityOrdering = false;
  LICConfig.PassSyncScopeId = false;
  LICConfig.PassIsVolatile = false;
  auto *LIC = InstrumentationConfig::allocate<LoadIO>(/*IsPRE=*/true);
  LIC->HoistKind = HOIST_MAXIMALLY;
  LIC->CB = [&](Value &V) {
    return LSI.shouldInstrumentLoad(cast<LoadInst>(V), IIRB);
  };
  LIC->init(*this, IIRB, &LICConfig);

  StoreIO::ConfigTy SICConfig;
  SICConfig.PassPointerAS = false;
  SICConfig.PassLoopValueRangeInfo = false;
  SICConfig.PassStoredValue = false;
  SICConfig.PassAlignment = false;
  SICConfig.PassValueTypeId = false;
  SICConfig.PassAtomicityOrdering = false;
  SICConfig.PassSyncScopeId = false;
  SICConfig.PassIsVolatile = false;
  auto *SIC = InstrumentationConfig::allocate<StoreIO>(/*IsPRE=*/true);
  SIC->HoistKind = HOIST_MAXIMALLY;
  SIC->CB = [&](Value &V) {
    return LSI.shouldInstrumentStore(cast<StoreInst>(V), IIRB);
  };
  SIC->init(*this, IIRB, &SICConfig);

  CallIO::ConfigTy CICConfig;
  CICConfig.PassCallee = false;
  CICConfig.PassCalleeName = false;
  CICConfig.PassReturnedValue = false;
  CICConfig.PassReturnedValueSize = false;
  CICConfig.ArgFilter = [&](Use &Op) {
    auto *CI = cast<CallInst>(Op.getUser());
    auto &TLI = IIRB.TLIGetter(*CI->getFunction());
    auto ACI = getAllocationCallInfo(CI, &TLI);
    return Op->getType()->isPointerTy() || ACI;
  };
  auto *CIC = InstrumentationConfig::allocate<CallIO>(/*IsPRE=*/true);
  CIC->CB = [&](Value &V) {
    return LSI.shouldInstrumentCall(cast<CallInst>(V));
  };
  CIC->init(*this, IIRB.Ctx, &CICConfig);
}

} // namespace

PreservedAnalyses
LightSanPass::run(Module &M, AnalysisManager<Module> &MAM) {
  LightSanImpl Impl(M, MAM);

  bool Changed = Impl.instrument();
  if (!Changed)
    return PreservedAnalyses::all();

  if (verifyModule(M))
    M.dump();
  assert(!verifyModule(M, &errs()));

  return PreservedAnalyses::none();
}
