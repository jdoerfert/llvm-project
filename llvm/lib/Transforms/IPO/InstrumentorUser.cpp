//===-- InstrumentorUser.cpp - Example pass that uses Instrumentor --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/InstrumentorUser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GEPNoWrapFlags.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/IPO/Instrumentor.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <cassert>
#include <cstdint>
#include <functional>

using namespace llvm;
using namespace llvm::instrumentor;

#define DEBUG_TYPE "instrumentor_user"

static constexpr char InstrumentorUserRuntimePrefix[] = "__instrumentor_user_";

namespace {

struct InstrumentorUserImpl;

struct InstrumentorUserConfig : public InstrumentationConfig {
  InstrumentorUserConfig(InstrumentorUserImpl &IUI, Module &M);
  virtual ~InstrumentorUserConfig() {}

  void populate(InstrumentorIRBuilderTy &IRB) override;

  InstrumentorUserImpl &IUI;
};

struct InstrumentorUserImpl {
  InstrumentorUserImpl(Module &M, ModuleAnalysisManager &MAM)
    : M(M), MAM(MAM),
      FAM(MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager()),
      IConf(*this, M), IIRB(M, FAM) {}

  bool instrument();

  bool shouldInstrumentCall(CallInst &CI) { return true; }
  bool shouldInstrumentFunction(Function &Fn) { return true; }
  bool shouldInstrumentLoad(LoadInst &LI) { return true; }
  bool shouldInstrumentStore(StoreInst &SI) { return true; }
  bool shouldInstrumentAlloca(AllocaInst &AI) { return true; }

private:
  Module &M;
  ModuleAnalysisManager &MAM;
  FunctionAnalysisManager &FAM;
  InstrumentorUserConfig IConf;
  InstrumentorIRBuilderTy IIRB;
  const DataLayout &DL = M.getDataLayout();
};

struct ExtendedAllocaIO : public AllocaIO {
  ExtendedAllocaIO(bool IsPRE) : AllocaIO(IsPRE) {}
  virtual ~ExtendedAllocaIO() {};

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    AllocaIO::ConfigTy AICConfig(/*Enable=*/false);
    AICConfig.set(AllocaIO::PassAddress);
    AICConfig.set(AllocaIO::ReplaceAddress);
    AICConfig.set(AllocaIO::PassSize);
    AllocaIO::init(IConf, IIRB.Ctx, &AICConfig);

    IRTArgs.push_back(IRTArg(IIRB.PtrTy, "function_name",
                             "The function name.", IRTArg::STRING,
                             getFunctionName));
  }

  static Value *getFunctionName(Value &V, Type &Ty,
                                InstrumentationConfig &IConf,
                                InstrumentorIRBuilderTy &IIRB) {
    auto &AI = cast<AllocaInst>(V);
    Function *Fn = AI.getFunction();
    return IConf.getGlobalString(Fn->getName(), IIRB);
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *EAIO = IConf.allocate<ExtendedAllocaIO>(/*IsPRE*/ false);
    auto &LSIConf = static_cast<InstrumentorUserConfig &>(IConf);
    EAIO->CB = [&](Value &V) {
      return LSIConf.IUI.shouldInstrumentAlloca(cast<AllocaInst>(V));
    };
    EAIO->init(IConf, IIRB);
  }
};

bool InstrumentorUserImpl::instrument() {
  bool Changed = false;

  InstrumentorPass IP(&IConf, &IIRB);
  auto PA = IP.run(M, MAM);
  if (!PA.areAllPreserved())
    Changed = true;

  return Changed;
}

InstrumentorUserConfig::InstrumentorUserConfig(InstrumentorUserImpl &Impl,
                                                           Module &M)
    : InstrumentationConfig(), IUI(Impl) {
  ReadConfig = false;
  RuntimePrefix->setString(InstrumentorUserRuntimePrefix);
  DemangleFunctionNames->setBool(true);
  RuntimeStubsFile->setString("instrumentor_user_rt_stub.c");
  RuntimeBitcode->setString("");
  InlineRuntimeEagerly->setBool(false);
}

void InstrumentorUserConfig::populate(InstrumentorIRBuilderTy &IIRB) {
  ExtendedAllocaIO::populate(*this, IIRB);

  CallIO::ConfigTy CICConfig(/*Enable=*/false);
  CICConfig.set(CallIO::PassCallee);
  CICConfig.set(CallIO::PassNumParameters);
  CICConfig.set(CallIO::PassParameters);
  CICConfig.ArgFilter = [&](Use &Op) {
    return Op->getType()->isPointerTy();
  };
  auto *PreCIC = InstrumentationConfig::allocate<CallIO>(/*IsPRE=*/true);
  PreCIC->CB = [&](Value &V) {
    return IUI.shouldInstrumentCall(cast<CallInst>(V));
  };
  PreCIC->init(*this, IIRB.Ctx, &CICConfig);

  LoadIO::ConfigTy LICConfig(/*Enable=*/false);
  LICConfig.set(LoadIO::PassPointer);
  LICConfig.set(LoadIO::ReplacePointer);
  LICConfig.set(LoadIO::PassValueSize);
  auto *LIC = InstrumentationConfig::allocate<LoadIO>(/*IsPRE=*/true);
  LIC->HoistKind = DO_NOT_HOIST;
  LIC->CB = [&](Value &V) {
    return IUI.shouldInstrumentLoad(cast<LoadInst>(V));
  };
  LIC->init(*this, IIRB, &LICConfig);

  StoreIO::ConfigTy SICConfig(/*Enable=*/false);
  SICConfig.set(StoreIO::PassPointer);
  SICConfig.set(StoreIO::ReplacePointer);
  SICConfig.set(StoreIO::PassStoredValueSize);
  auto *SIC = InstrumentationConfig::allocate<StoreIO>(/*IsPRE=*/true);
  SIC->HoistKind = DO_NOT_HOIST;
  SIC->CB = [&](Value &V) {
    return IUI.shouldInstrumentStore(cast<StoreInst>(V));
  };
  SIC->init(*this, IIRB, &SICConfig);

  FunctionIO::ConfigTy FICConfig(/*Enable=*/false);
  FICConfig.set(FunctionIO::PassName);
  FICConfig.set(FunctionIO::PassAddress);
  FICConfig.set(FunctionIO::PassNumArguments);
  FICConfig.set(FunctionIO::PassArguments);
  FICConfig.set(FunctionIO::ReplaceArguments);
  auto *FIC = InstrumentationConfig::allocate<FunctionIO>(/*IsPRE=*/true);
  FIC->CB = [&](Value &V) {
    return IUI.shouldInstrumentFunction(cast<Function>(V));
  };
  FIC->init(*this, IIRB.Ctx, &FICConfig);
}

} // namespace

PreservedAnalyses InstrumentorUserPass::run(Module &M, AnalysisManager<Module> &MAM) {
  InstrumentorUserImpl Impl(M, MAM);

  bool Changed = Impl.instrument();
  if (!Changed)
    return PreservedAnalyses::all();

  if (verifyModule(M))
    M.dump();

  assert(!verifyModule(M, &errs()));

  return PreservedAnalyses::none();
}
