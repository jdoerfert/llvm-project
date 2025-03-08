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

#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Instrumentation/Instrumentor.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <functional>

using namespace llvm;
using namespace llvm::instrumentor;

#define DEBUG_TYPE "lightsan"

static constexpr char LightSanRuntimePrefix[] = "__objsan_";

namespace {

struct LightSanImpl;

struct LightSanInstrumentationConfig : public InstrumentationConfig {

  LightSanInstrumentationConfig(LightSanImpl &LSI);
  virtual ~LightSanInstrumentationConfig() {}

  void populate(InstrumentorIRBuilderTy &IRB) override;

  struct ExtendedBasePointerInfo {
    Value *ObjectSize = nullptr;
    Value *EncodingNo = nullptr;
    Value *Offset = nullptr;
  };

  Value *getUnderlyingObject(Value *Ptr) {
    auto *NewVPtr = const_cast<Value *>(getUnderlyingObjectAggressive(Ptr));
    while (NewVPtr != Ptr) {
      Ptr = NewVPtr;
      NewVPtr = const_cast<Value *>(getUnderlyingObjectAggressive(Ptr));
    }
    return Ptr;
  }

  DenseMap<std::pair<Value *, Function *>, ExtendedBasePointerInfo>
      BasePointerSizeOffsetMap;
  Value *getBasePointerObjectSize(Value &V, InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    Value *Ptr = getUnderlyingObject(&V);
    return BasePointerSizeOffsetMap[{Ptr, Fn}].ObjectSize;
  }
  Value *getBasePointerEncodingNo(Value &V, InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    Value *Ptr = getUnderlyingObject(&V);
    return BasePointerSizeOffsetMap[{Ptr, Fn}].EncodingNo;
  }
  Value *getBasePointerOffset(Value &V, InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    Value *Ptr = getUnderlyingObject(&V);
    return BasePointerSizeOffsetMap[{Ptr, Fn}].Offset;
  }

  LightSanImpl &LSI;
};

struct LightSanImpl {
  LightSanImpl(Module &M, ModuleAnalysisManager &MAM)
      : M(M), MAM(MAM), IConf(*this) {}

  bool instrument();

  bool shouldInstrumentFunction(Function &Fn);
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
  return true;
}

bool LightSanImpl::shouldInstrumentStore(StoreInst &SI,
                                         InstrumentorIRBuilderTy &IIRB) {
  return true;
}

bool LightSanImpl::shouldInstrumentAlloca(AllocaInst &AI,
                                          InstrumentorIRBuilderTy &IIRB) {
  return true;
}

bool LightSanImpl::shouldInstrumentFunction(Function &Fn) {
  return Fn.getName() == "main";
}

bool LightSanImpl::shouldInstrumentCall(CallInst &CI) {
  Function *CalledFn = CI.getCalledFunction();
  if (!CalledFn && CalledFn->isDeclaration())
    return true;
  return false;
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

struct ExtendedAllocaIO : public AllocaIO {
  ExtendedAllocaIO(bool IsPRE) : AllocaIO(IsPRE) {}
  virtual ~ExtendedAllocaIO() {};

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    AllocaIO::ConfigTy AICConfig(/*Enable=*/false);
    AICConfig.set(AllocaIO::PassAddress);
    AICConfig.set(AllocaIO::ReplaceAddress);
    AICConfig.set(AllocaIO::PassSize);
    AllocaIO::init(IConf, IIRB.Ctx, &AICConfig);

    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "requires_temporal_check",
                             "Flag to indicate that the alloca might be "
                             "accessed after it was freed.",
                             IRTArg::NONE, getRequiresTemporalCheck));
  }

  static Value *getRequiresTemporalCheck(Value &V, Type &Ty,
                                         InstrumentationConfig &IConf,
                                         InstrumentorIRBuilderTy &IIRB) {
    // TODO: Be smarter about this, e.g., AAPointerInfo.
    bool MayEscape = !all_of(V.uses(), [](const Use &U) {
      return isa<LoadInst>(U.getUser()) ||
             (isa<StoreInst>(U.getUser()) &&
              U.getOperandNo() != StoreInst::getPointerOperandIndex());
    });
    return ConstantInt::get(&Ty, MayEscape);
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *EAIO = IConf.allocate<ExtendedAllocaIO>(/*IsPRE*/ false);
    EAIO->init(IConf, IIRB);
  }
};

struct ExtendedBasePointerIO : public BasePointerIO {
  ExtendedBasePointerIO() : BasePointerIO() {}
  virtual ~ExtendedBasePointerIO() {};

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    BasePointerIO::ConfigTy BaseConfig(/*Enable=*/false);
    BaseConfig.set(BasePointerIO::PassPointer, true);
    BasePointerIO::init(IConf, Ctx, &BaseConfig);

    IRTArgs.push_back(
        IRTArg(PointerType::getUnqual(Ctx), "object_size_ptr",
               "Return the size of the object in question as uint64_t.",
               IRTArg::REPLACABLE_CUSTOM, getObjectSizePtr, setObjectSize));
    IRTArgs.push_back(IRTArg(
        PointerType::getUnqual(Ctx), "encoding_no_ptr",
        "Return the encoding number of the object in question as uint8_t.",
        IRTArg::REPLACABLE_CUSTOM, getEncodingNoPtr, setEncodingNo));
    IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "offset_ptr",
                             "Return the offset of the pointer as int64_t.",
                             IRTArg::REPLACABLE_CUSTOM, getOffsetPtr,
                             setOffset));
  }

  static Value *getObjectSizePtr(Value &V, Type &Ty,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    return IIRB.getAlloca(Fn, IIRB.Int64Ty);
  }
  static Value *setObjectSize(Value &V, Value &NewV,
                              InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    auto *ObjectSize = IIRB.IRB.CreateLoad(IIRB.Int64Ty, &NewV);
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto *BasePtr = cast<CallInst>(V).getArgOperand(0);
    LSIConf.BasePointerSizeOffsetMap[{BasePtr, Fn}].ObjectSize = ObjectSize;
    return &V;
  }
  static Value *getEncodingNoPtr(Value &V, Type &Ty,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    return IIRB.getAlloca(Fn, IIRB.Int8Ty);
  }
  static Value *setEncodingNo(Value &V, Value &NewV,
                              InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    auto *EncodingNo = IIRB.IRB.CreateLoad(IIRB.Int8Ty, &NewV);
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto *BasePtr = cast<CallInst>(V).getArgOperand(0);
    LSIConf.BasePointerSizeOffsetMap[{BasePtr, Fn}].EncodingNo = EncodingNo;
    return &V;
  }
  static Value *getOffsetPtr(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    return IIRB.getAlloca(Fn, IIRB.Int64Ty);
  }
  static Value *setOffset(Value &V, Value &NewV, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    auto *Offset = IIRB.IRB.CreateLoad(IIRB.Int64Ty, &NewV);
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    auto *BasePtr = cast<CallInst>(V).getArgOperand(0);
    LSIConf.BasePointerSizeOffsetMap[{BasePtr, Fn}].Offset = Offset;
    return &V;
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *EVPIO = IConf.allocate<ExtendedBasePointerIO>();
    EVPIO->init(IConf, IIRB.Ctx);
  }
};

struct ExtendedLoopValueRangeIO : public LoopValueRangeIO {
  ExtendedLoopValueRangeIO() : LoopValueRangeIO() {}
  virtual ~ExtendedLoopValueRangeIO() {};

  StringRef getName() const override { return "loop_value_range"; }

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    LoopValueRangeIO::init(IConf, IIRB);

    IRTArgs.push_back(IRTArg(IIRB.PtrTy, "base_pointer_info",
                             "The runtime provided base pointer info.",
                             IRTArg::NONE, getBasePointerInfo));
    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "object_size",
                             "The size of the underlying object.", IRTArg::NONE,
                             getObjectSize));
    IRTArgs.push_back(
        IRTArg(IIRB.Int64Ty, "object_offset",
               "The offset of the underlying base pointer in the object.",
               IRTArg::NONE, getBaseOffset));
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "encoding_no",
                             "The encoding number used for the pointer.",
                             IRTArg::NONE, getEncodingNo));
  }

  static Value *getBasePointerInfo(Value &V, Type &Ty,
                                   InstrumentationConfig &IConf,
                                   InstrumentorIRBuilderTy &IIRB) {
    return IConf.getBasePointerInfo(V, IIRB);
  }
  static Value *getObjectSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerObjectSize(V, IIRB);
  }
  static Value *getBaseOffset(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerOffset(V, IIRB);
  }
  static Value *getEncodingNo(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerEncodingNo(V, IIRB);
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *LVRIO = IConf.allocate<ExtendedLoopValueRangeIO>();
    LVRIO->init(IConf, IIRB);
  }
};

struct ExtendedLoadIO : public LoadIO {
  ExtendedLoadIO(bool IsPRE) : LoadIO(IsPRE) {}
  virtual ~ExtendedLoadIO() {};

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    LoadIO::ConfigTy LICConfig(/*Enable=*/false);
    LICConfig.set(LoadIO::PassPointer);
    LICConfig.set(LoadIO::ReplacePointer);
    LICConfig.set(LoadIO::PassBasePointerInfo);
    LICConfig.set(LoadIO::PassValueSize);
    LICConfig.set(LoadIO::PassLoopValueRangeInfo);
    LoadIO::init(IConf, IIRB, &LICConfig);

    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "object_size",
                             "The size of the underlying object.", IRTArg::NONE,
                             getObjectSize));
    IRTArgs.push_back(
        IRTArg(IIRB.Int64Ty, "object_offset",
               "The offset of the underlying base pointer in the object.",
               IRTArg::NONE, getBaseOffset));
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "encoding_no",
                             "The encoding number used for the pointer.",
                             IRTArg::NONE, getEncodingNo));
  }

  static Value *getObjectSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerObjectSize(
        *cast<LoadInst>(V).getPointerOperand(), IIRB);
  }
  static Value *getBaseOffset(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerOffset(*cast<LoadInst>(V).getPointerOperand(),
                                        IIRB);
  }
  static Value *getEncodingNo(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerEncodingNo(
        *cast<LoadInst>(V).getPointerOperand(), IIRB);
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *ESIO = IConf.allocate<ExtendedLoadIO>(/*IsPRE*/ true);
    ESIO->init(IConf, IIRB);
  }
};

struct ExtendedStoreIO : public StoreIO {
  ExtendedStoreIO(bool IsPRE) : StoreIO(IsPRE) {}
  virtual ~ExtendedStoreIO() {};

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    StoreIO::ConfigTy SICConfig(/*Enable=*/false);
    SICConfig.set(StoreIO::PassPointer);
    SICConfig.set(StoreIO::ReplacePointer);
    SICConfig.set(StoreIO::PassBasePointerInfo);
    SICConfig.set(StoreIO::PassStoredValueSize);
    SICConfig.set(StoreIO::PassLoopValueRangeInfo);
    StoreIO::init(IConf, IIRB, &SICConfig);

    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "object_size",
                             "The size of the underlying object.", IRTArg::NONE,
                             getObjectSize));
    IRTArgs.push_back(
        IRTArg(IIRB.Int64Ty, "object_offset",
               "The offset of the underlying base pointer in the object.",
               IRTArg::NONE, getBaseOffset));
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "encoding_no",
                             "The encoding number used for the pointer.",
                             IRTArg::NONE, getEncodingNo));
  }

  static Value *getObjectSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerObjectSize(
        *cast<StoreInst>(V).getPointerOperand(), IIRB);
  }
  static Value *getBaseOffset(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerOffset(*cast<StoreInst>(V).getPointerOperand(),
                                        IIRB);
  }
  static Value *getEncodingNo(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
    auto &LSIConf = static_cast<LightSanInstrumentationConfig &>(IConf);
    return LSIConf.getBasePointerEncodingNo(
        *cast<StoreInst>(V).getPointerOperand(), IIRB);
  }

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *ESIO = IConf.allocate<ExtendedStoreIO>(/*IsPRE*/ true);
    ESIO->init(IConf, IIRB);
  }
};

void LightSanInstrumentationConfig::populate(InstrumentorIRBuilderTy &IIRB) {
  UnreachableIO::populate(*this, IIRB.Ctx);
  ExtendedBasePointerIO::populate(*this, IIRB);
  ExtendedStoreIO::populate(*this, IIRB);
  ExtendedLoadIO::populate(*this, IIRB);
  ExtendedLoopValueRangeIO::populate(*this, IIRB);
  //  ModuleIO::populate(*this, IIRB.Ctx);
  //  GlobalIO::populate(*this, IIRB.Ctx);

  AllocaIO::ConfigTy AICConfig(/*Enable=*/false);
  AICConfig.set(AllocaIO::PassAddress);
  AICConfig.set(AllocaIO::ReplaceAddress);
  AICConfig.set(AllocaIO::PassSize);
  auto *AIC = InstrumentationConfig::allocate<AllocaIO>(/*IsPRE=*/false);
  AIC->CB = [&](Value &V) {
    return LSI.shouldInstrumentAlloca(cast<AllocaInst>(V), IIRB);
  };
  AIC->init(*this, IIRB.Ctx, &AICConfig);

  CallIO::ConfigTy CICConfig(/*Enable=*/false);
  CICConfig.set(CallIO::PassIntrinsicId);
  CICConfig.set(CallIO::PassAllocationInfo);
  CICConfig.set(CallIO::PassNumParameters);
  CICConfig.set(CallIO::PassParameters);
  CICConfig.set(CallIO::PassIsDefinition);
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

  //  FunctionIO::ConfigTy FICConfig(/*Enable=*/false);
  //  FICConfig.set(FunctionIO::PassName);
  //  FICConfig.set(FunctionIO::PassNumArguments);
  //  FICConfig.set(FunctionIO::PassArguments);
  //  FICConfig.set(FunctionIO::ReplaceArguments);
  //  auto *FIC = InstrumentationConfig::allocate<FunctionIO>();
  //  FIC->CB = [&](Value &V) {
  //    return LSI.shouldInstrumentFunction(cast<Function>(V));
  //  };
  //  FIC->init(*this, IIRB.Ctx, &FICConfig);
}

} // namespace

PreservedAnalyses LightSanPass::run(Module &M, AnalysisManager<Module> &MAM) {
  LightSanImpl Impl(M, MAM);

  bool Changed = Impl.instrument();
  if (!Changed)
    return PreservedAnalyses::all();

  if (verifyModule(M))
    M.dump();
  assert(!verifyModule(M, &errs()));

  return PreservedAnalyses::none();
}
