//===-- Instrumentor.cpp - Highly configurable instrumentation pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/InputGen.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
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

using namespace llvm;
using namespace llvm::instrumentor;

#define DEBUG_TYPE "input-gen"

static cl::opt<IGIMode> ClInstrumentationMode(
    "input-gen-mode", cl::desc("input-gen instrumentation mode"), cl::Hidden,
    cl::init(IGIMode::Disabled),
    cl::values(clEnumValN(IGIMode::Disabled, "disable", ""),
               clEnumValN(IGIMode::Record, "record", ""),
               clEnumValN(IGIMode::Generate, "generate", ""),
               clEnumValN(IGIMode::ReplayGenerated, "replay_generated", ""),
               clEnumValN(IGIMode::ReplayRecorded, "replay_recorded", "")));

static cl::list<std::string>
    AllowedExternalFuncs("input-gen-allow-external-funcs",
                         cl::desc("Specify allowed external function(s)"),
                         cl::Hidden);

#ifndef NDEBUG
static cl::opt<std::string>
    ClGenerateStubs("input-gen-generate-stubs",
                    cl::desc("Filename to generate the stubs for the input-gen "
                             "runtime in. Leave blank to disable."),
                    cl::Hidden);
#else
static constexpr char ClGenerateStubs[] = "";
#endif

static constexpr char InputGenRuntimePrefix[] = "__ig_";
static constexpr char InputGenRenamePrefix[] = "__renamed_ig_";

namespace {

struct InputGenMemoryImpl;

struct BranchConditionInfo {
  struct ParameterInfo {
    enum { INST, ARG, LOAD, MEMCMP } Kind;
    Value *const V;
    Value *const Ptr1 = nullptr;
    Value *const Ptr2 = nullptr;
    const uint32_t TypeId = 0;
    const uint32_t Size = 0;
    using ArgumentMapTy = DenseMap<Value *, uint32_t>;
    ParameterInfo(Argument &A)
        : Kind(ARG), V(&A), TypeId(A.getType()->getTypeID()) {}
    ParameterInfo(Instruction &I)
        : Kind(INST), V(&I), TypeId(I.getType()->getTypeID()) {}
    ParameterInfo(LoadInst &LI, CallInst &CI, const DataLayout &DL)
        : Kind(LOAD), V(&LI), Ptr1(CI.getArgOperand(0)),
          TypeId(LI.getType()->getTypeID()),
          Size(DL.getTypeStoreSize(LI.getType())) {}
    ParameterInfo(CallInst &CI, const DataLayout &DL)
        : Kind(MEMCMP), V(CI.getArgOperand(2)), Ptr1(CI.getArgOperand(0)),
          Ptr2(CI.getArgOperand(1)), TypeId(Type::TypeID::TokenTyID),
          Size(DL.getTypeStoreSize(CI.getType())) {}
  };
  uint32_t No;
  SmallVector<ParameterInfo> ParameterInfos;
  Function *Fn;
};

struct InputGenInstrumentationConfig : public InstrumentationConfig {

  InputGenInstrumentationConfig(InputGenMemoryImpl &IGMI);
  virtual ~InputGenInstrumentationConfig() {}

  void populate(LLVMContext &Ctx) override;

  DenseMap<Value *, BranchConditionInfo *> BCIMap;
  BranchConditionInfo &createBCI(Value &V) {
    auto *BCI = new BranchConditionInfo;
    BCIMap[&V] = BCI;
    return *BCI;
  }
  BranchConditionInfo &getBCI(Value &V) { return *BCIMap[&V]; }

  InputGenMemoryImpl &IGMI;

  using DTGetterTy = std::function<DominatorTree &(Function &F)>;
  DTGetterTy DTGetter;
  using PDTGetterTy = std::function<PostDominatorTree &(Function &F)>;
  PDTGetterTy PDTGetter;

  DenseMap<BranchInst *, uint32_t> BranchMap;
};

struct InputGenInstrumentationConfig;

struct InputGenMemoryImpl {
  InputGenMemoryImpl(Module &M, ModuleAnalysisManager &MAM, IGIMode Mode)
      : M(M), MAM(MAM), Mode(Mode),
        FAM(MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager()),
        IConf(*this) {}

  bool instrument();

  bool createPathTable();
  void createPathTable(Function &Fn);

  bool shouldInstrumentCall(CallInst &CI);
  bool shouldInstrumentLoad(LoadInst &LI);
  bool shouldInstrumentStore(StoreInst &SI);
  bool shouldInstrumentAlloca(AllocaInst &AI);
  bool shouldInstrumentBranch(BranchInst &BI);

  FunctionAnalysisManager &getFAM() { return FAM; };

private:
  Module &M;
  ModuleAnalysisManager &MAM;
  const IGIMode Mode;
  FunctionAnalysisManager &FAM;
  InputGenInstrumentationConfig IConf;
  const DataLayout &DL = M.getDataLayout();
};

struct InputGenEntriesImpl {
  InputGenEntriesImpl(Module &M, ModuleAnalysisManager &MAM, IGIMode Mode)
      : M(M), MAM(MAM), Mode(Mode),
        FAM(MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager()) {
  }

  bool instrument();

private:
  bool createEntryPoint();
  bool processFunctions();
  bool processOtherFunctions();

  FunctionAnalysisManager &getFAM() { return FAM; };

  Module &M;
  ModuleAnalysisManager &MAM;
  const IGIMode Mode;
  FunctionAnalysisManager &FAM;
  const DataLayout &DL = M.getDataLayout();
  SmallVector<Function *> UserFunctions;
  SmallVector<Function *> OtherFunctions;
};

struct BranchConditionIO : public InstructionIO<Instruction::Br> {
  BranchConditionIO() : InstructionIO<Instruction::Br>(/*IsPRE*/ true) {}
  virtual ~BranchConditionIO() {};

  Instruction *analyzeBranch(BranchInst &BI,
                             InputGenInstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);

  StringRef getName() const override { return "branch_condition_info"; }

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    IRTArgs.push_back(IRTArg(
        IntegerType::getInt32Ty(Ctx), "branch_condition_no",
        "The unique number of the branch condition.", IRTArg::NONE,
        [&](Value &V, Type &Ty, InstrumentationConfig &IConf,
            InstrumentorIRBuilderTy &IIRB) {
          auto &IGIConf = static_cast<InputGenInstrumentationConfig &>(IConf);
          return ConstantInt::get(&Ty, IGIConf.getBCI(V).No);
        }));
    IRTArgs.push_back(IRTArg(
        PointerType::getUnqual(Ctx), "branch_condition_fn",
        "The function computing the branch condition.", IRTArg::NONE,
        [&](Value &V, Type &Ty, InstrumentationConfig &IConf,
            InstrumentorIRBuilderTy &IIRB) {
          auto &IGIConf = static_cast<InputGenInstrumentationConfig &>(IConf);
          return IGIConf.getBCI(V).Fn;
        }));
    IRTArgs.push_back(IRTArg(
        IntegerType::getInt32Ty(Ctx), "num_branch_condition_arguments",
        "Number of arguments of the branch condition function.", IRTArg::NONE,
        [&](Value &V, Type &Ty, InstrumentationConfig &IConf,
            InstrumentorIRBuilderTy &IIRB) {
          auto &IGIConf = static_cast<InputGenInstrumentationConfig &>(IConf);
          return ConstantInt::get(&Ty, IGIConf.getBCI(V).ParameterInfos.size());
        }));
    IRTArgs.push_back(
        IRTArg(PointerType::getUnqual(Ctx), "arguments",
               "Description of the arguments.", IRTArg::NONE,
               [&](Value &V, Type &Ty, InstrumentationConfig &IConf,
                   InstrumentorIRBuilderTy &IIRB) {
                 return getArguments(V, Ty, IConf, IIRB);
               }));
    IConf.addChoice(*this);
  }

  static uint32_t BranchConditionNo;

  Value *getArguments(Value &V, Type &Ty, InstrumentationConfig &IConf,
                      InstrumentorIRBuilderTy &IIRB);

  Value *instrument(Value *&V, InstrumentationConfig &IConf,
                    InstrumentorIRBuilderTy &IIRB) override {
    if (CB && !CB(*V))
      return nullptr;
    auto *BI = cast<BranchInst>(V);
    auto &IGIConf = static_cast<InputGenInstrumentationConfig &>(IConf);
    auto *IP = analyzeBranch(*BI, IGIConf, IIRB);
    if (!IP)
      return nullptr;
    IRBuilderBase::InsertPointGuard IPG(IIRB.IRB);
    IIRB.IRB.SetInsertPoint(IP);
    return InstructionIO::instrument(V, IConf, IIRB);
  }

  //  Type *getRetTy(LLVMContext &Ctx) const override {
  //    return Type::getInt1Ty(Ctx);
  //  }
};

uint32_t BranchConditionIO::BranchConditionNo = 0;

Instruction *
BranchConditionIO::analyzeBranch(BranchInst &BI,
                                 InputGenInstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
  assert(BI.isConditional() && "Expected a conditional branch!");
  auto &BCI = IConf.createBCI(BI);
  BCI.No = BranchConditionIO::BranchConditionNo++;

  const auto &DL = BI.getDataLayout();

  DenseMap<Value *, uint32_t> UseCountMap;
  DenseMap<Value *, uint32_t> ArgumentMap;
  SmallVector<Value *> Worklist;
  auto AddValue = [&](Instruction *I, uint32_t IncUses) {
    Worklist.push_back(I);
    UseCountMap[I] += IncUses ? 1 : -1;
    //    if (IncUses && UseCountMap[I] == I->getNumUses())
    //      IIRB.eraseLater(I);
  };
  AddValue(cast<Instruction>(BI.getCondition()), /*IncUses=*/true);

  bool HasLoad = false;
  while (!Worklist.empty()) {
    auto *V = Worklist.pop_back_val();
    if (auto *A = dyn_cast<Argument>(V)) {
      if (!ArgumentMap.contains(A)) {
        ArgumentMap[A] = ArgumentMap.size();
        BCI.ParameterInfos.emplace_back(*A);
      }
      continue;
    }
    bool InstIsOK = false;
    if (auto *LI = dyn_cast<LoadInst>(V)) {
      auto *CI = dyn_cast<CallInst>(LI->getPointerOperand());
      if (CI && CI->getCalledFunction() &&
          CI->getCalledFunction()->getName() ==
              IConf.getRTName("pre_", "load")) {
        BCI.ParameterInfos.emplace_back(*LI, *CI, DL);
        HasLoad = true;
        InstIsOK = true;
      }
    }
    if (auto *CI = dyn_cast<CallInst>(V)) {
      // TODO: use target library info here
      if (CI->getCalledFunction() &&
          CI->getCalledFunction()->getName() == IConf.getRTName("", "memcmp")) {
        BCI.ParameterInfos.emplace_back(*CI, DL);
        HasLoad = true;
        InstIsOK = true;
      }
    }
    if (auto *I = dyn_cast<Instruction>(V)) {
      if (!InstIsOK && (I->mayHaveSideEffects() || isa<PHINode>(I) ||
                        I->mayReadFromMemory())) {
        if (!ArgumentMap.contains(I)) {
          ArgumentMap[I] = ArgumentMap.size();
          BCI.ParameterInfos.emplace_back(*I);
        }
        continue;
      }
      for (auto *Op : I->operand_values()) {
        if (auto *OpI = dyn_cast<Instruction>(Op))
          AddValue(OpI, /*IncUses=*/true);
        if (auto *OpA = dyn_cast<Argument>(Op))
          Worklist.push_back(OpA);
      }
      continue;
    }
    assert(isa<Constant>(V));
  }
  if (!HasLoad)
    return nullptr;

  auto &Ctx = BI.getContext();
  Instruction *IP = nullptr;
  auto &DT = IConf.DTGetter(*BI.getFunction());

  std::function<void(Instruction *)> HoistInsts = [&](Instruction *I) {
    if (I->mayHaveSideEffects() || I->mayReadFromMemory())
      return;
    SmallVector<Instruction *> OpInsts;
    for (auto *Op : I->operand_values())
      if (auto *OpI = dyn_cast<Instruction>(Op)) {
        HoistInsts(OpI);
        OpInsts.push_back(OpI);
      }
    Instruction *IP = nullptr;
    for (auto *OpI : OpInsts) {
      if (!IP || DT.dominates(IP, OpI))
        IP = OpI;
    }
    if (!IP)
      IP = &*I->getFunction()->getEntryBlock().getFirstNonPHIOrDbgOrAlloca();
    I->moveAfter(IP);
  };

  auto AdjustIP = [&](Instruction *I) {
    if (!IP || DT.dominates(IP, I)) {
      if (isa<PHINode>(I)) {
        IP = &*I->getParent()->getFirstNonPHIOrDbgOrLifetime();
      } else {
        HoistInsts(I);
        IP = I->getNextNode();
      }
      return;
    }
    assert(DT.dominates(I, IP));
  };

  SmallVector<Type *> ParameterTypes;
  for (auto &PI : BCI.ParameterInfos) {
    switch (PI.Kind) {
    case BranchConditionInfo::ParameterInfo::ARG:
      ParameterTypes.push_back(PI.V->getType());
      break;
    case BranchConditionInfo::ParameterInfo::MEMCMP: {
      auto *SizeI = dyn_cast<Instruction>(PI.V);
      if (SizeI)
        AdjustIP(SizeI);
      auto *Ptr1I = dyn_cast<Instruction>(PI.Ptr1);
      if (Ptr1I)
        AdjustIP(Ptr1I);
      auto *Ptr2I = dyn_cast<Instruction>(PI.Ptr2);
      if (Ptr2I)
        AdjustIP(Ptr2I);
      break;
    }
    case BranchConditionInfo::ParameterInfo::LOAD: {
      auto *PtrI = dyn_cast<Instruction>(PI.Ptr1);
      if (PtrI)
        AdjustIP(PtrI);
      break;
    }
    case BranchConditionInfo::ParameterInfo::INST:
      ParameterTypes.push_back(PI.V->getType());
      AdjustIP(cast<Instruction>(PI.V));
      break;
    }
  }
  if (!IP)
    IP = &*BI.getFunction()->getEntryBlock().getFirstNonPHIOrDbgOrAlloca();

  auto *RetTy = Type::getInt8Ty(Ctx);
  Function *BCIFn = Function::Create(
      FunctionType::get(RetTy, {PointerType::getUnqual(Ctx)}, false),
      GlobalValue::InternalLinkage, IConf.getRTName("", "branch_cond_fn"),
      BI.getModule());

  auto *EntryBB = BasicBlock::Create(Ctx, "entry", BCIFn);
  auto *ComputeBB = BasicBlock::Create(Ctx, "compute", BCIFn);

  StructType *STy =
      StructType::get(IIRB.Ctx, ParameterTypes, /*isPacked=*/true);
  ValueToValueMapTy VM;

  IRBuilder<> IRB(EntryBB);
  AddValue(cast<Instruction>(BI.getCondition()), /*IncUses=*/false);
  while (!Worklist.empty()) {
    auto *V = Worklist.pop_back_val();
    if (isa<Constant>(V))
      continue;
    auto AMIt = ArgumentMap.find(V);
    if (AMIt != ArgumentMap.end()) {
      auto *Ptr = IRB.CreateStructGEP(STy, BCIFn->getArg(0), AMIt->second);
      VM[V] = IRB.CreateLoad(V->getType(), Ptr);
      continue;
    }
    auto &Uses = UseCountMap[V];
    if (Uses > 0) {
      assert(Worklist.size());
      Worklist.push_back(V);
      continue;
    }

    auto *I = cast<Instruction>(V);
    auto *CloneI = I->clone();
    CloneI->insertInto(ComputeBB, ComputeBB->begin());
    if (auto *CI = dyn_cast<CallInst>(CloneI))
      if (auto *Callee = CI->getCalledFunction())
        if (Callee->getName().starts_with(IConf.getRTName()))
          CI->setCalledFunction(
              CI->getModule()->getOrInsertFunction((Callee->getName() + "2").str(), Callee->getFunctionType()));
    // Callee->getName().drop_front(IConf.getRTName().size())));

    VM[I] = CloneI;
    for (auto *Op : I->operand_values()) {
      if (auto *OpI = dyn_cast<Instruction>(Op))
        AddValue(OpI, /*IncUses=*/false);
      if (auto *OpA = dyn_cast<Argument>(Op))
        Worklist.push_back(OpA);
    }
  }
  RemapFunction(*BCIFn, VM, RF_IgnoreMissingLocals);

  IRB.CreateBr(ComputeBB);
  ReturnInst::Create(Ctx,
                     new ZExtInst(VM[BI.getCondition()], RetTy, "", ComputeBB),
                     ComputeBB);
  BCI.Fn = BCIFn;
  return IP;
}

Value *BranchConditionIO::getArguments(Value &V, Type &Ty,
                                       InstrumentationConfig &IConf,
                                       InstrumentorIRBuilderTy &IIRB) {
  auto &BI = cast<BranchInst>(V);
  auto &IGIConf = static_cast<InputGenInstrumentationConfig &>(IConf);
  auto &BCI = IGIConf.getBCI(V);
  if (BCI.ParameterInfos.empty())
    return Constant::getNullValue(&Ty);

  SmallVector<Type *> ParameterTypes;
  SmallVector<Value *> ParameterValues;
  for (auto &PI : BCI.ParameterInfos) {
    ParameterTypes.push_back(IIRB.IRB.getInt32Ty());
    ParameterValues.push_back(IIRB.IRB.getInt32(PI.Kind));
    ParameterTypes.push_back(IIRB.IRB.getInt32Ty());
    ParameterValues.push_back(IIRB.IRB.getInt32(PI.TypeId));
    switch (PI.Kind) {
    case BranchConditionInfo::ParameterInfo::INST:
    case BranchConditionInfo::ParameterInfo::ARG:
      ParameterTypes.push_back(IIRB.IRB.getInt32Ty());
      ParameterValues.push_back(
          IIRB.IRB.getInt32(IIRB.DL.getTypeAllocSize(PI.V->getType())));
      ParameterTypes.push_back(PI.V->getType());
      ParameterValues.push_back(PI.V);
      break;
    case BranchConditionInfo::ParameterInfo::LOAD:
      ParameterTypes.push_back(IIRB.IRB.getInt32Ty());
      ParameterValues.push_back(IIRB.IRB.getInt32(PI.Size));
      ParameterTypes.push_back(PI.Ptr1->getType());
      ParameterValues.push_back(PI.Ptr1);
      break;
    case BranchConditionInfo::ParameterInfo::MEMCMP:
      ParameterTypes.push_back(IIRB.IRB.getInt32Ty());
      ParameterValues.push_back(
          IIRB.IRB.getInt32(IIRB.DL.getTypeAllocSize(IIRB.IRB.getInt32Ty())));
      ParameterTypes.push_back(PI.V->getType());
      ParameterValues.push_back(PI.V);
      ParameterTypes.push_back(PI.Ptr1->getType());
      ParameterValues.push_back(PI.Ptr1);
      ParameterTypes.push_back(PI.Ptr2->getType());
      ParameterValues.push_back(PI.Ptr2);
      break;
    }
  }

  StructType *STy =
      StructType::get(IIRB.Ctx, ParameterTypes, /*isPacked=*/true);
  auto *AI = IIRB.getAlloca(BI.getFunction(), STy);
  for (auto [Idx, V] : enumerate(ParameterValues)) {
    auto *Ptr = IIRB.IRB.CreateStructGEP(STy, AI, Idx);
    IIRB.IRB.CreateStore(V, Ptr);
  }
  IIRB.returnAllocas({AI});
  return AI;
}

bool InputGenMemoryImpl::shouldInstrumentBranch(BranchInst &BI) {
  return BI.isConditional() && isa<Instruction>(BI.getCondition());
}

bool InputGenMemoryImpl::shouldInstrumentLoad(LoadInst &LI) {
  const Value *UnderlyingPtr =
      getUnderlyingObjectAggressive(LI.getPointerOperand());
  if (auto *AI = dyn_cast<AllocaInst>(UnderlyingPtr)) {
    if (AI->getAllocationSize(DL) >= DL.getTypeStoreSize(LI.getType()))
      return false;
  }
  return true;
}

bool InputGenMemoryImpl::shouldInstrumentStore(StoreInst &SI) {
  const Value *UnderlyingPtr =
      getUnderlyingObjectAggressive(SI.getPointerOperand());
  if (auto *AI = dyn_cast<AllocaInst>(UnderlyingPtr)) {
    if (AI->getAllocationSize(DL) >=
        DL.getTypeStoreSize(SI.getValueOperand()->getType()))
      return false;
  }
  return true;
}

bool InputGenMemoryImpl::shouldInstrumentAlloca(AllocaInst &AI) {
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

bool InputGenMemoryImpl::shouldInstrumentCall(CallInst &CI) {
  if (CI.getCaller()->getName().starts_with(IConf.getRTName()))
    return false;
  auto *Callee = CI.getCalledFunction();
  if (!Callee)
    return true;
  // Rewrite some known functions instead of instrumenting them.
  if (Callee->getName() == "memcmp") {
    CI.setCalledFunction(M.getOrInsertFunction(IConf.getRTName("", "memcmp"),
                                               Callee->getFunctionType()));
    return false;
  }
  return true;
}

void InputGenMemoryImpl::createPathTable(Function &Fn) {

  // DenseMap<BasicBlock *, SmallVector<StringRef, 4>> PathMap;

  // ReversePostOrderTraversal<Function *> RPOT(&Fn);
  // for (auto *BB : RPOT) {
  //   auto *BI = dyn_cast<BranchInst>(BB->getTerminator());
  //   if (!BI || !BI->isConditional() || !IConf.BranchMap.contains(BI))
  //     continue;
  //   uint32_t Idx = IConf.BranchMap[BI];

  //   auto AddSuccessor = [&](BranchInst &BI, uint32_t SuccIdx) {
  //     auto *SuccBB = BI.getSuccessor(SuccIdx);
  //     auto &Succ = PathMap[SuccBB];
  //   };

  //   AddSuccessor(*BI, 0);
  //   AddSuccessor(*BI, 1);
  // }
}

bool InputGenMemoryImpl::createPathTable() {
  bool Changed = false;

  // auto *GV = M.getGlobalVariable(std::string(InputGenRuntimePrefix) +
  //                                "entry_point_names");
  // errs() << GV << "\n";
  // if (GV)
  //   errs() << *GV << "\n";
  //  auto *NameArray = cast<ConstantArray>(
  //          ->getInitializer());
  //  NameArray->dump();
  //  for (auto *NameGV : NameArray->operand_values()) {
  //    NameGV->dump();
  //    auto Name =
  //        cast<ConstantDataArray>(cast<GlobalVariable>(NameGV)->getInitializer())
  //            ->getRawDataValues();
  //    createPathTable(*M.getFunction(Name));
  //  }

  return Changed;
}

bool InputGenMemoryImpl::instrument() {
  if (Mode != IGIMode::Generate || Mode == IGIMode::Record)
    return false;

  bool Changed = false;

  InstrumentorPass IP(&IConf);

  auto PA = IP.run(M, MAM);
  if (!PA.areAllPreserved())
    Changed = true;

  Changed |= createPathTable();

  return Changed;
}

bool InputGenEntriesImpl::instrument() {
  if (Mode == IGIMode::Generate || Mode == IGIMode::ReplayGenerated ||
      Mode == IGIMode::ReplayRecorded) {
    bool Changed = false;

    for (auto &Fn : M.functions()) {
      if (Fn.hasFnAttribute(Attribute::InputGenEntry)) {
        UserFunctions.push_back(&Fn);
      } else if (!Fn.isDeclaration()) {
        OtherFunctions.push_back(&Fn);
      }
    }

    Changed |= createEntryPoint();
    Changed |= processFunctions();

    return Changed;
  } else {
    return false;
  }
}

static bool isPersonalityFunction(Function &F) {
  return !F.use_empty() && all_of(F.uses(), [&](Use &U) {
    if (auto *UserF = dyn_cast<Function>(U.getUser()))
      if (UserF->getPersonalityFn() == &F)
        return true;
    return false;
  });
}

static bool shouldPreserveName(Function *F) {
  StringRef Name = F->getName();
  bool UserAllowedExternal = llvm::any_of(
      AllowedExternalFuncs, [&](std::string N) { return N == Name; });
  bool IsCxaThrow = Name == "__cxa_throw";
  return isPersonalityFunction(*F) || UserAllowedExternal || IsCxaThrow ||
         F->isIntrinsic();
}

// TODO instrumentor needs to instrument memory functions such as malloc calloc
// free memcpy etc. If we rename them it can't do that? We probably want to do
// the renaming in the memory pass and not the entries pass.
static void processFunctionDeclarationForGeneration(Function *F) {
  assert(F->isDeclaration());
  // We need to rename external functions that we will be stubbing so as not to
  // clash with an existing definition somewhere.
  if (!shouldPreserveName(F))
    F->setName(InputGenRenamePrefix + F->getName());
}

static void processFunctionDefinitionForGenerate(Function *F) {
  assert(!F->isDeclaration());
  // We want to aggressively inline to strengthen the InputGenMemory
  // instrumentation analysis.
  F->addFnAttr(Attribute::AlwaysInline);
  // TODO also look at the callsites for noinline
  F->removeFnAttr(Attribute::NoInline);

  // We do not want any definitions to clash with any other modules we may link
  // in.
  F->setLinkage(GlobalValue::PrivateLinkage);
  F->setVisibility(GlobalValue::DefaultVisibility);
}

static void processFunctionDefinitionForReplayGenerated(Function *F) {
  // We do not want any definitions to clash with any other modules we may link
  // in.
  F->setLinkage(GlobalValue::PrivateLinkage);
  F->setVisibility(GlobalValue::DefaultVisibility);
}

bool InputGenEntriesImpl::processFunctions() {
  for (Function *F : UserFunctions) {
    if (Mode == IGIMode::Generate)
      processFunctionDefinitionForGenerate(F);
    if (Mode == IGIMode::ReplayGenerated)
      processFunctionDefinitionForReplayGenerated(F);
  }
  for (Function *F : OtherFunctions) {
    if (!F->isDeclaration()) {
      if (Mode == IGIMode::Generate) {
        processFunctionDefinitionForGenerate(F);
      } else if (Mode == IGIMode::ReplayGenerated) {
        processFunctionDefinitionForReplayGenerated(F);
      }
    }
  }

  // Since the OtherFunctions may be unused, we need to make sure they do not
  // get optimized away before we have a chance to consider them for indirect
  // call candidates later.
  if (Mode == IGIMode::Generate || Mode == IGIMode::ReplayGenerated)
    appendToCompilerUsed(M,
                         llvm::map_to_vector(OtherFunctions, [](Function *F) {
                           return cast<GlobalValue>(F);
                         }));

  return true;
}

bool InputGenEntriesImpl::createEntryPoint() {
  auto &Ctx = M.getContext();
  auto *I32Ty = IntegerType::getInt32Ty(Ctx);
  auto *PtrTy = PointerType::getUnqual(Ctx);

  uint32_t NumEntryPoints = UserFunctions.size();
  new GlobalVariable(M, I32Ty, true, GlobalValue::ExternalLinkage,
                     ConstantInt::get(I32Ty, NumEntryPoints),
                     std::string(InputGenRuntimePrefix) + "num_entry_points");

  Function *IGEntry = Function::Create(
      FunctionType::get(Type::getVoidTy(Ctx), {I32Ty, PtrTy}, false),
      GlobalValue::ExternalLinkage,
      std::string(InputGenRuntimePrefix) + "entry", M);
  // Tell Instrumentor not to ignore this function
  IGEntry->addFnAttr("instrument");

  auto *EntryChoice = IGEntry->getArg(0);
  auto *InitialObj = IGEntry->getArg(1);

  auto *EntryBB = BasicBlock::Create(Ctx, "entry", IGEntry);
  auto *ReturnBB = BasicBlock::Create(Ctx, "return", IGEntry);
  auto *SI = SwitchInst::Create(EntryChoice, ReturnBB, NumEntryPoints, EntryBB);
  ReturnInst::Create(Ctx, ReturnBB);

  SmallVector<Constant *> Names;
  IRBuilder<> IRB(SI);
  for (uint32_t I = 0; I < NumEntryPoints; ++I) {
    Value *ObjPtr = InitialObj;
    auto *DispatchBB = BasicBlock::Create(Ctx, "dispatch", IGEntry);
    Function *EntryPoint = UserFunctions[I];

    SmallVector<Value *> Parameters;
    for (auto &Arg : EntryPoint->args()) {
      auto *LI = new LoadInst(Arg.getType(), ObjPtr, Arg.getName(), DispatchBB);
      Parameters.push_back(LI);
      ObjPtr = GetElementPtrInst::Create(
          PtrTy, ObjPtr,
          {ConstantInt::get(I32Ty, DL.getTypeStoreSize(Arg.getType()))}, "",
          DispatchBB);
    }

    auto *CI = CallInst::Create(EntryPoint->getFunctionType(), EntryPoint,
                                Parameters, "", DispatchBB);
    CI->addFnAttr(Attribute::NoInline);
    if (!CI->getType()->isVoidTy())
      new StoreInst(CI, ObjPtr, DispatchBB);
    else if (auto *I = dyn_cast<Instruction>(ObjPtr))
      I->eraseFromParent();
    SI->addCase(ConstantInt::get(I32Ty, I), DispatchBB);

    BranchInst::Create(ReturnBB, DispatchBB);

    Names.push_back(IRB.CreateGlobalString(EntryPoint->getName()));
  }
  ArrayType *NameArrayTy = ArrayType::get(PtrTy, NumEntryPoints);
  Constant *NameArray = ConstantArray::get(NameArrayTy, Names);

  new GlobalVariable(M, NameArrayTy, true, GlobalValue::ExternalLinkage,
                     NameArray,
                     std::string(InputGenRuntimePrefix) + "entry_point_names");

  return true;
}

InputGenInstrumentationConfig::InputGenInstrumentationConfig(
    InputGenMemoryImpl &IGI)
    : InstrumentationConfig(), IGMI(IGI),
      DTGetter([&](Function &F) -> DominatorTree & {
        return IGI.getFAM().getResult<DominatorTreeAnalysis>(F);
      }),
      PDTGetter([&](Function &F) -> PostDominatorTree & {
        return IGI.getFAM().getResult<PostDominatorTreeAnalysis>(F);
      }) {
  ReadConfig = false;
  RuntimePrefix->setString(InputGenRuntimePrefix);
  RuntimeStubsFile->setString(ClGenerateStubs);
}

void InputGenInstrumentationConfig::populate(LLVMContext &Ctx) {
  UnreachableIO::populate(*this, Ctx);
  BasePointerIO::populate(*this, Ctx);

  auto *BIC = new (ChoiceAllocator.Allocate()) BranchConditionIO;
  BIC->CB = [&](Value &V) {
    return IGMI.shouldInstrumentBranch(cast<BranchInst>(V));
  };
  BIC->init(*this, Ctx);

  auto *AIC = new (ChoiceAllocator.Allocate()) AllocaIO(/*IsPRE=*/false);
  AIC->CB = [&](Value &V) {
    return IGMI.shouldInstrumentAlloca(cast<AllocaInst>(V));
  };
  AIC->init(*this, Ctx, /*ReplaceAddr=*/true, /*ReplaceSize=*/false,
            /*PassAlignment*/ true);

  auto *LIC = new (ChoiceAllocator.Allocate()) LoadIO(/*IsPRE=*/true);
  LIC->CB = [&](Value &V) {
    return IGMI.shouldInstrumentLoad(cast<LoadInst>(V));
  };
  LIC->init(*this, Ctx, /*PassPointer=*/true, /*ReplacePointer=*/true,
            /*PassPointerAS=*/false, /*PassBasePointerInfo=*/true,
            /*PassValue=*/false, /*ReplaceValue*/ false,
            /*PassValueSize=*/true, /*PassAlignment=*/true,
            /*PassValueTypeId=*/true, /*PassAtomicityOrdering=*/false,
            /*PassSyncScopeId=*/false, /*PassIsVolatile=*/false);

  auto *SIC = new (ChoiceAllocator.Allocate()) StoreIO(/*IsPRE=*/true);
  SIC->CB = [&](Value &V) {
    return IGMI.shouldInstrumentStore(cast<StoreInst>(V));
  };
  SIC->init(*this, Ctx, /*PassPointer=*/true, /*ReplacePointer=*/true,
            /*PassPointerAS=*/false, /*PassBasePointerInfo=*/true,
            /*PassStoredValue=*/true, /*PassStoredValueSize*/ true,
            /*PassAlignment=*/true,
            /*PassValueTypeId=*/true, /*PassAtomicityOrdering=*/false,
            /*PassSyncScopeId=*/false, /*PassIsVolatile=*/false);

  auto *CIC = new (ChoiceAllocator.Allocate()) CallIO(/*IsPRE=*/true);
  CIC->CB = [&](Value &V) {
    return IGMI.shouldInstrumentCall(cast<CallInst>(V));
  };
  CIC->init(*this, Ctx, /*PassCallee=*/true, /*PassCalleeName=*/true,
            /*PassIntrinsicId=*/true, /*PassAllocationInfo=*/true,
            /*PassReturnedValue=*/true, /*PassReturnedValueSize=*/true,
            /*PassNumParameters=*/true, /*PassParameters=*/true,
            /*PassIsDefinition=*/false);
}

} // namespace

PreservedAnalyses
InputGenInstrumentEntriesPass::run(Module &M, AnalysisManager<Module> &MAM) {
  IGIMode Mode = ClInstrumentationMode;
  InputGenEntriesImpl Impl(M, MAM, Mode);

  bool Changed = Impl.instrument();
  if (!Changed)
    return PreservedAnalyses::all();

  if (verifyModule(M))
    M.dump();
  assert(!verifyModule(M, &errs()));

  return PreservedAnalyses::none();
}

PreservedAnalyses
InputGenInstrumentMemoryPass::run(Module &M, AnalysisManager<Module> &MAM) {
  IGIMode Mode = ClInstrumentationMode;
  InputGenMemoryImpl Impl(M, MAM, Mode);

  bool Changed = Impl.instrument();
  if (!Changed)
    return PreservedAnalyses::all();

  if (verifyModule(M))
    M.dump();
  assert(!verifyModule(M, &errs()));

  return PreservedAnalyses::none();
}
