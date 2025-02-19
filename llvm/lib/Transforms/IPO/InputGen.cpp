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
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
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

static cl::list<std::string>
    EntryFunctionNames("input-gen-entry-function",
                       cl::desc("Tag the provided functions as entries."),
                       cl::Hidden);

static cl::opt<bool>
    EntryAllFunctions("input-gen-entry-all-functions",
                      cl::desc("Tag all function definitions as entries."),
                      cl::init(false), cl::Hidden);

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
    enum KindTy { INST, ARG, LOAD, MEMCMP, STRCMP } Kind;
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
    ParameterInfo(KindTy K, Value *V, Value *Ptr1, Value *Ptr2)
        : Kind(K), V(V), Ptr1(Ptr1), Ptr2(Ptr2),
          TypeId(Type::TypeID::TokenTyID), Size(0) {}
  };
  uint32_t No;
  SmallVector<ParameterInfo> ParameterInfos;
  Function *Fn;
};

struct InputGenInstrumentationConfig : public InstrumentationConfig {

  InputGenInstrumentationConfig(InputGenMemoryImpl &IGMI);
  virtual ~InputGenInstrumentationConfig() {}

  void populate(InstrumentorIRBuilderTy &IRB) override;

  DenseMap<Value *, BranchConditionInfo *> BCIMap;
  BranchConditionInfo &createBCI(Value &V) {
    auto *BCI = new BranchConditionInfo;
    BCIMap[&V] = BCI;
    return *BCI;
  }
  BranchConditionInfo &getBCI(Value &V) { return *BCIMap[&V]; }

  InputGenMemoryImpl &IGMI;

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
  bool shouldInstrumentLoad(LoadInst &LI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentStore(StoreInst &SI, InstrumentorIRBuilderTy &IIRB);
  bool shouldInstrumentAlloca(AllocaInst &AI, InstrumentorIRBuilderTy &IIRB);
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

  // The below three vectors contain all Functions in the module.

  /// The entry point functions.
  SmallVector<Function *> EntryFunctions;
  /// Other function definitions in the module.
  SmallVector<Function *> OtherFunctions;
  /// The function declarations.
  SmallVector<Function *> DeclaredFunctions;
};

struct BranchConditionIO : public InstructionIO<Instruction::Br> {
  BranchConditionIO() : InstructionIO<Instruction::Br>(/*IsPRE*/ true) {}
  virtual ~BranchConditionIO() {};

  std::optional<BasicBlock::iterator>
  analyzeBranch(BranchInst &BI, InputGenInstrumentationConfig &IConf,
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
                    InstrumentorIRBuilderTy &IIRB,
                    InstrumentationCaches &ICaches) override {
    if (CB && !CB(*V))
      return nullptr;
    auto *BI = cast<BranchInst>(V);
    auto &IGIConf = static_cast<InputGenInstrumentationConfig &>(IConf);
    auto IP = analyzeBranch(*BI, IGIConf, IIRB);
    if (!IP)
      return nullptr;
    IRBuilderBase::InsertPointGuard IPG(IIRB.IRB);
    IIRB.IRB.SetInsertPoint(*IP);
    return InstructionIO::instrument(V, IConf, IIRB, ICaches);
  }

  //  Type *getRetTy(LLVMContext &Ctx) const override {
  //    return Type::getInt1Ty(Ctx);
  //  }
};

uint32_t BranchConditionIO::BranchConditionNo = 0;

std::optional<BasicBlock::iterator>
BranchConditionIO::analyzeBranch(BranchInst &BI,
                                 InputGenInstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
  assert(BI.isConditional() && "Expected a conditional branch!");
  auto &BCI = IConf.createBCI(BI);

  const auto &DL = BI.getDataLayout();

  ValueToValueMapTy PtrRemap;
  DenseMap<Value *, uint32_t> UseCountMap;
  DenseMap<Value *, uint32_t> ArgumentMap;
  SmallVector<Value *> Worklist;
  auto AddValue = [&](Value *V, uint32_t IncUses) {
    uint32_t &Uses = UseCountMap[V];
    if (IncUses) {
      if (Uses++)
        return;
    } else if (--Uses)
      return;
    Worklist.push_back(V);
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
        AddValue(CI->getArgOperand(0), /*IncUses=*/true);
        PtrRemap[CI] = CI->getArgOperand(0);
        continue;
      }
    }
    if (auto *CI = dyn_cast<CallInst>(V)) {
      // TODO: use target library info here
      assert(!(CI->getCalledFunction() && CI->getCalledFunction()->getName() ==
                                              IConf.getRTName("pre_", "load")));
      if (CI->getCalledFunction() &&
          CI->getCalledFunction()->getName() == IConf.getRTName("", "memcmp")) {
        BCI.ParameterInfos.emplace_back(*CI, DL);
        HasLoad = true;
        InstIsOK = true;
      }
      if (CI->getCalledFunction() &&
          CI->getCalledFunction()->getName() == IConf.getRTName("", "strcmp")) {
        BCI.ParameterInfos.emplace_back(
            BranchConditionInfo::ParameterInfo::STRCMP, nullptr,
            CI->getArgOperand(0), CI->getArgOperand(1));
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
          AddValue(OpA, /*IncUses=*/true);
      }
      continue;
    }
    assert(isa<Constant>(V));
  }
  if (!HasLoad)
    return std::nullopt;

  BasicBlock::iterator IP =
      BI.getFunction()->getEntryBlock().getFirstNonPHIOrDbgOrAlloca();

  SmallVector<Type *> ParameterTypes;
  for (auto &PI : BCI.ParameterInfos) {
    switch (PI.Kind) {
    case BranchConditionInfo::ParameterInfo::ARG:
      ParameterTypes.push_back(PI.V->getType());
      break;
    case BranchConditionInfo::ParameterInfo::MEMCMP:
      if (auto *SizeI = dyn_cast<Instruction>(PI.V))
        IP = IIRB.hoistInstructionsAndAdjustIP(*SizeI, HoistKind, IP).IP;
      LLVM_FALLTHROUGH;
    case BranchConditionInfo::ParameterInfo::STRCMP:
      if (auto *Ptr1I = dyn_cast<Instruction>(PI.Ptr1))
        IP = IIRB.hoistInstructionsAndAdjustIP(*Ptr1I, HoistKind, IP).IP;
      if (auto *Ptr2I = dyn_cast<Instruction>(PI.Ptr2))
        IP = IIRB.hoistInstructionsAndAdjustIP(*Ptr2I, HoistKind, IP).IP;
      break;
    case BranchConditionInfo::ParameterInfo::LOAD:
      if (auto *PtrI = dyn_cast<Instruction>(PI.Ptr1))
        IP = IIRB.hoistInstructionsAndAdjustIP(*PtrI, HoistKind, IP).IP;
      break;
    case BranchConditionInfo::ParameterInfo::INST:
      ParameterTypes.push_back(PI.V->getType());
      IP = IIRB.hoistInstructionsAndAdjustIP(*cast<Instruction>(PI.V),
                                             HoistKind, IP)
               .IP;
      break;
    }
  }

  auto &Ctx = BI.getContext();
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
  Type *PtrTy = IRB.getPtrTy();
  FunctionCallee DecodeFn = BCIFn->getParent()->getOrInsertFunction(
      IConf.getRTName("", "decode"), FunctionType::get(PtrTy, {PtrTy}, false));

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
    assert(!isa<PHINode>(V));
    assert(UseCountMap[V] == 0);

    auto *I = cast<Instruction>(V);
    auto *CloneI = I->clone();
    CloneI->insertInto(ComputeBB, ComputeBB->begin());
    if (auto *CI = dyn_cast<CallInst>(CloneI))
      if (auto *Callee = CI->getCalledFunction())
        if (Callee->getName().starts_with(IConf.getRTName()))
          CI->setCalledFunction(CI->getModule()->getOrInsertFunction(
              (Callee->getName() + "2").str(), Callee->getFunctionType()));
    // Callee->getName().drop_front(IConf.getRTName().size())));

    VM[V] = CloneI;
    for (auto *Op : I->operand_values()) {
      if (const auto &NewOp = PtrRemap.lookup(Op)) {
        auto *CI = CallInst::Create(DecodeFn, {NewOp}, "", ComputeBB->begin());
        VM[Op] = CI;
        Op = NewOp;
      }
      if (auto *OpI = dyn_cast<Instruction>(Op)) {
        AddValue(OpI, /*IncUses=*/false);
      }
      if (auto *OpA = dyn_cast<Argument>(Op))
        AddValue(OpA, /*IncUses=*/false);
    }
  }
  RemapFunction(*BCIFn, VM, RF_IgnoreMissingLocals);

  IRB.CreateBr(ComputeBB);
  ReturnInst::Create(Ctx,
                     new ZExtInst(VM[BI.getCondition()], RetTy, "", ComputeBB),
                     ComputeBB);
  BCI.Fn = BCIFn;
  BCI.No = BranchConditionIO::BranchConditionNo++;
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

  auto GetTypeOrEquivInt = [&](Type *Ty) -> Type * {
    if (Ty->isPointerTy())
      return IIRB.IRB.getIntNTy(IIRB.DL.getPointerSizeInBits());
    return Ty;
  };

  SmallVector<Type *> ParameterTypes;
  SmallVector<Constant *> ConstantValues;
  SmallVector<std::pair<Value *, uint32_t>> ParameterValues;

  auto PushValue = [&](Value *V) {
    ParameterTypes.push_back(GetTypeOrEquivInt(V->getType()));
    ParameterValues.push_back({V, ConstantValues.size()});
    ConstantValues.push_back(Constant::getNullValue(ParameterTypes.back()));
  };

  for (auto &PI : BCI.ParameterInfos) {
    ParameterTypes.push_back(IIRB.IRB.getInt32Ty());
    ConstantValues.push_back(IIRB.IRB.getInt32(PI.Kind));
    ParameterTypes.push_back(IIRB.IRB.getInt32Ty());
    ConstantValues.push_back(IIRB.IRB.getInt32(PI.TypeId));
    switch (PI.Kind) {
    case BranchConditionInfo::ParameterInfo::INST:
    case BranchConditionInfo::ParameterInfo::ARG:
      ParameterTypes.push_back(IIRB.IRB.getInt32Ty());
      ConstantValues.push_back(
          IIRB.IRB.getInt32(IIRB.DL.getTypeAllocSize(PI.V->getType())));
      PushValue(PI.V);
      break;
    case BranchConditionInfo::ParameterInfo::LOAD:
      ParameterTypes.push_back(IIRB.IRB.getInt32Ty());
      ConstantValues.push_back(IIRB.IRB.getInt32(PI.Size));
      PushValue(PI.Ptr1);
      break;
    case BranchConditionInfo::ParameterInfo::MEMCMP:
      ParameterTypes.push_back(IIRB.IRB.getInt32Ty());
      ConstantValues.push_back(
          IIRB.IRB.getInt32(IIRB.DL.getTypeAllocSize(IIRB.IRB.getInt32Ty())));
      PushValue(PI.V);
      PushValue(PI.Ptr1);
      PushValue(PI.Ptr2);
      break;
    case BranchConditionInfo::ParameterInfo::STRCMP:
      ParameterTypes.push_back(IIRB.IRB.getInt32Ty());
      ConstantValues.push_back(
          IIRB.IRB.getInt32(IIRB.DL.getTypeAllocSize(IIRB.IRB.getInt32Ty())));
      PushValue(PI.Ptr1);
      PushValue(PI.Ptr2);
      break;
    }
  }

  StructType *STy =
      StructType::get(IIRB.Ctx, ParameterTypes, /*isPacked=*/true);
  Constant *Initializer = ConstantStruct::get(STy, ConstantValues);
  GlobalVariable *&GV = IConf.ConstantGlobalsCache[Initializer];
  if (!GV)
    GV = new GlobalVariable(*BI.getModule(), STy, false,
                            GlobalValue::InternalLinkage, Initializer,
                            IConf.getRTName("", "bci_pack"));

  auto *AI = IIRB.getAlloca(BI.getFunction(), STy);
  IIRB.IRB.CreateMemCpy(AI, AI->getAlign(), GV, MaybeAlign(GV->getAlignment()),
                        IIRB.DL.getTypeAllocSize(STy));
  for (auto [V, Idx] : ParameterValues) {
    auto *Ptr = IIRB.IRB.CreateStructGEP(STy, AI, Idx);
    IIRB.IRB.CreateStore(V, Ptr);
  }
  IIRB.returnAllocas({AI});
  return AI;
}

bool InputGenMemoryImpl::shouldInstrumentBranch(BranchInst &BI) {
  return BI.isConditional() && isa<Instruction>(BI.getCondition());
}

bool InputGenMemoryImpl::shouldInstrumentLoad(LoadInst &LI,
                                              InstrumentorIRBuilderTy &IIRB) {
  if (auto *AI = dyn_cast<AllocaInst>(LI.getPointerOperand()))
    return shouldInstrumentAlloca(*AI, IIRB);
  return true;
}

bool InputGenMemoryImpl::shouldInstrumentStore(StoreInst &SI,
                                               InstrumentorIRBuilderTy &IIRB) {
  if (auto *AI = dyn_cast<AllocaInst>(SI.getPointerOperand()))
    return shouldInstrumentAlloca(*AI, IIRB);
  return true;
}

bool InputGenMemoryImpl::shouldInstrumentAlloca(AllocaInst &AI,
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

bool InputGenMemoryImpl::shouldInstrumentCall(CallInst &CI) {
  if (CI.getCaller()->getName().starts_with(IConf.getRTName()) &&
      !CI.getCaller()->hasFnAttribute("instrument"))
    return false;
  auto *Callee = CI.getCalledFunction();
  if (!Callee)
    return (CI.mayHaveSideEffects() || CI.mayReadFromMemory()) && CI.arg_size();
  if (!Callee->isDeclaration())
    return false;
  if (!CI.getType()->isPointerTy() && none_of(CI.args(), [](Value *Arg) {
        return Arg->getType()->isPointerTy();
      }))
    return false;
  if (auto *II = dyn_cast<IntrinsicInst>(&CI)) {
    if (II->isAssumeLikeIntrinsic())
      return false;
  }
  if (Callee->getName().starts_with(IConf.getRTName()))
    return false;
  // Rewrite some known functions instead of instrumenting them.
  if (StringSwitch<bool>(Callee->getName())
          .Case("memcmp", true)
          .Case("strcmp", true)
          //          .Case("__sprintf_chk", true)
          .Default(false)) {
    CI.setCalledFunction(M.getOrInsertFunction(
        IConf.getRTName("", Callee->getName()), Callee->getFunctionType()));
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

  SmallVector<Function *> OldFunctions;
  removeFromUsedLists(M, [&](Constant *C) {
    auto *Fn = dyn_cast<Function>(C);
    if (!Fn || Fn->getNumUses() > 1)
      return false;
    OldFunctions.push_back(Fn);
    return true;
  });
  for (auto *Fn : OldFunctions)
    Fn->eraseFromParent();

  // TODO: HACK for qsort, we need to actually check the functions we rename
  // here and qsort explicitly.
  for (auto &Fn : M) {
    if (Fn.isDeclaration() || !Fn.hasLocalLinkage())
      continue;
    bool HasNonQSortUses = false;
    for (auto &U : Fn.uses()) {
      if (auto *CU = dyn_cast<Constant>(U.getUser()))
        if (CU->getNumUses() == 1)
          if (auto *GV = dyn_cast<GlobalVariable>(CU->user_back()))
            if (GV->getName() == "llvm.compiler.used" ||
                GV->getName() == "llvm.used")
              continue;
      auto *CI = dyn_cast<CallInst>(U.getUser());
      if (!CI || &CI->getCalledOperandUse() == &U || !CI->getCalledFunction() ||
          CI->getCalledFunction()->getName() != "qsort") {
        HasNonQSortUses = true;
        break;
      }
    }
    if (!HasNonQSortUses)
      Fn.setName(IConf.getRTName() + Fn.getName());
  }

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
      if (Fn.hasFnAttribute(Attribute::InputGenEntry) && !Fn.isDeclaration()) {
        EntryFunctions.push_back(&Fn);
      } else if (Fn.isDeclaration()) {
        DeclaredFunctions.push_back(&Fn);
      } else {
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
  for (Function *F : EntryFunctions) {
    if (Mode == IGIMode::Generate)
      processFunctionDefinitionForGenerate(F);
    if (Mode == IGIMode::ReplayGenerated)
      processFunctionDefinitionForReplayGenerated(F);
  }
  for (Function *F : OtherFunctions) {
    if (Mode == IGIMode::Generate) {
      processFunctionDefinitionForGenerate(F);
    } else if (Mode == IGIMode::ReplayGenerated) {
      processFunctionDefinitionForReplayGenerated(F);
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

  uint32_t NumEntryPoints = EntryFunctions.size();
  new GlobalVariable(M, I32Ty, true, GlobalValue::ExternalLinkage,
                     ConstantInt::get(I32Ty, NumEntryPoints),
                     std::string(InputGenRuntimePrefix) + "num_entry_points");

  Function *IGEntry = Function::Create(
      FunctionType::get(Type::getVoidTy(Ctx), {I32Ty, PtrTy}, false),
      GlobalValue::ExternalLinkage,
      std::string(InputGenRuntimePrefix) + "entry", M);

  auto *EntryChoice = IGEntry->getArg(0);
  auto *EntryObj = IGEntry->getArg(1);

  auto *EntryBB = BasicBlock::Create(Ctx, "entry", IGEntry);
  auto *ReturnBB = BasicBlock::Create(Ctx, "return", IGEntry);
  auto *SI = SwitchInst::Create(EntryChoice, ReturnBB, NumEntryPoints, EntryBB);
  ReturnInst::Create(Ctx, ReturnBB);

  SmallVector<Constant *> Names;
  IRBuilder<> IRB(SI);
  for (uint32_t I = 0; I < NumEntryPoints; ++I) {
    Function *EntryPoint = EntryFunctions[I];
    Names.push_back(IRB.CreateGlobalString(EntryPoint->getName()));
    EntryPoint->setName(std::string(InputGenRuntimePrefix) +
                        EntryPoint->getName());

    Function *EntryPointWrapper = Function::Create(
        FunctionType::get(Type::getVoidTy(Ctx), {PtrTy}, false),
        GlobalValue::InternalLinkage, EntryPoint->getName() + ".wrapper", M);
    // Tell Instrumentor not to ignore these functions.
    EntryPoint->addFnAttr("instrument");
    EntryPointWrapper->addFnAttr("instrument");
    EntryPointWrapper->addFnAttr(Attribute::NoInline);

    auto *WrapperEntryBB = BasicBlock::Create(Ctx, "entry", EntryPointWrapper);

    SmallVector<Value *> Parameters;
    Value *WrapperObjPtr = EntryPointWrapper->getArg(0);
    for (auto &Arg : EntryPoint->args()) {
      auto *LI = new LoadInst(Arg.getType(), WrapperObjPtr, Arg.getName(),
                              WrapperEntryBB);
      Parameters.push_back(LI);
      WrapperObjPtr = GetElementPtrInst::Create(
          PtrTy, WrapperObjPtr,
          {ConstantInt::get(I32Ty, DL.getTypeStoreSize(Arg.getType()))}, "",
          WrapperEntryBB);
    }

    auto *CI = CallInst::Create(EntryPoint->getFunctionType(), EntryPoint,
                                Parameters, "", WrapperEntryBB);
    if (!CI->getType()->isVoidTy())
      new StoreInst(CI, WrapperObjPtr, WrapperEntryBB);
    ReturnInst::Create(Ctx, WrapperEntryBB);

    EntryPoint->addFnAttr(Attribute::AlwaysInline);
    EntryPoint->removeFnAttr(Attribute::NoInline);

    auto *DispatchBB = BasicBlock::Create(Ctx, "dispatch", IGEntry);
    CallInst::Create(EntryPointWrapper->getFunctionType(), EntryPointWrapper,
                     {EntryObj}, "", DispatchBB);
    SI->addCase(ConstantInt::get(I32Ty, I), DispatchBB);

    BranchInst::Create(ReturnBB, DispatchBB);
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
      PDTGetter([&](Function &F) -> PostDominatorTree & {
        return IGI.getFAM().getResult<PostDominatorTreeAnalysis>(F);
      }) {
  ReadConfig = false;
  RuntimePrefix->setString(InputGenRuntimePrefix);
  RuntimeStubsFile->setString(ClGenerateStubs);
}

void InputGenInstrumentationConfig::populate(InstrumentorIRBuilderTy &IIRB) {
  UnreachableIO::populate(*this, IIRB.Ctx);
  BasePointerIO::populate(*this, IIRB.Ctx);
  LoopValueRangeIO::populate(*this, IIRB);

  auto *BIC = InstrumentationConfig::allocate<BranchConditionIO>();
  BIC->HoistKind = HOIST_MAXIMALLY;
  BIC->CB = [&](Value &V) {
    return IGMI.shouldInstrumentBranch(cast<BranchInst>(V));
  };
  BIC->init(*this, IIRB.Ctx);

  auto *AIC = InstrumentationConfig::allocate<AllocaIO>(/*IsPRE=*/false);
  AIC->CB = [&](Value &V) {
    return IGMI.shouldInstrumentAlloca(cast<AllocaInst>(V), IIRB);
  };
  AIC->init(*this, IIRB.Ctx, /*ReplaceAddr=*/true, /*ReplaceSize=*/false,
            /*PassAlignment*/ true);

  LoadIO::ConfigTy LICConfig;
  LICConfig.PassPointerAS = false;
  LICConfig.PassValue = false;
  LICConfig.ReplaceValue = false;
  LICConfig.PassAtomicityOrdering = false;
  LICConfig.PassSyncScopeId = false;
  LICConfig.PassIsVolatile = false;
  auto *LIC = InstrumentationConfig::allocate<LoadIO>(/*IsPRE=*/true);
  LIC->HoistKind = HOIST_MAXIMALLY;
  LIC->CB = [&](Value &V) {
    return IGMI.shouldInstrumentLoad(cast<LoadInst>(V), IIRB);
  };
  LIC->init(*this, IIRB, &LICConfig);

  StoreIO::ConfigTy SICConfig;
  SICConfig.PassPointerAS = false;
  SICConfig.PassAtomicityOrdering = false;
  SICConfig.PassSyncScopeId = false;
  SICConfig.PassIsVolatile = false;
  SICConfig.PassStoredValue = false;
  auto *SIC = InstrumentationConfig::allocate<StoreIO>(/*IsPRE=*/true);
  SIC->HoistKind = HOIST_MAXIMALLY;
  SIC->CB = [&](Value &V) {
    return IGMI.shouldInstrumentStore(cast<StoreInst>(V), IIRB);
  };
  SIC->init(*this, IIRB, &SICConfig);

  CallIO::ConfigTy CICConfig;
  CICConfig.ArgFilter = [&](Use &Op) {
    auto *CI = cast<CallInst>(Op.getUser());
    auto &TLI = IIRB.TLIGetter(*CI->getFunction());
    auto ACI = getAllocationCallInfo(CI, &TLI);
    return Op->getType()->isPointerTy() || ACI;
  };
  for (bool IsPRE : {true, false}) {
    auto *CIC = InstrumentationConfig::allocate<CallIO>(IsPRE);
    CIC->CB = [&](Value &V) {
      return IGMI.shouldInstrumentCall(cast<CallInst>(V));
    };
    CIC->init(*this, IIRB.Ctx, &CICConfig);
  }
}

} // namespace

static bool tagEntries(Module &M) {
  bool Changed = false;
  if (EntryAllFunctions) {
    for (auto &F : M) {
      if (!F.isDeclaration()) {
        F.addFnAttr(Attribute::InputGenEntry);
        Changed = true;
      }
    }
  } else {
    for (std::string &Name : EntryFunctionNames) {
      Function *F = M.getFunction(Name);
      if (!F->isDeclaration()) {
        F->addFnAttr(Attribute::InputGenEntry);
        Changed = true;
      }
    }
  }
  return Changed;
}

PreservedAnalyses
InputGenInstrumentEntriesPass::run(Module &M, AnalysisManager<Module> &MAM) {
  IGIMode Mode = ClInstrumentationMode;
  InputGenEntriesImpl Impl(M, MAM, Mode);

  bool Changed = false;

  Changed |= tagEntries(M);
  Changed |= Impl.instrument();

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
