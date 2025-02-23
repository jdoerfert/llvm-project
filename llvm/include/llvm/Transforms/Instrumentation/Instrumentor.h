//===- Transforms/Instrumentation/Instrumentor.h --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A highly configurable instrumentation pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_INSTRUMENTOR_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_INSTRUMENTOR_H

#include "llvm/ADT/EnumeratedArray.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Transforms/Utils/Instrumentation.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>

namespace llvm {
namespace instrumentor {
enum HoistKindTy {
  DO_NOT_HOIST = 0,
  HOIST_IN_BLOCK,
  HOIST_OUT_OF_LOOPS,
  HOIST_MAXIMALLY,
};

struct InstrumentationConfig;
struct InstrumentationOpportunity;

struct InstrumentorIRBuilderTy {
  using TLIGetterTy = std::function<TargetLibraryInfo &(Function &F)>;
  using DTGetterTy = std::function<DominatorTree &(Function &F)>;
  using SEGetterTy = std::function<ScalarEvolution &(Function &F)>;
  using LIGetterTy = std::function<LoopInfo &(Function &F)>;

  InstrumentorIRBuilderTy(Module &M, TLIGetterTy &&TLIGetter,
                          DTGetterTy &&DTGetter, SEGetterTy &&SEGetter,
                          LIGetterTy &&LIGetter)
      : M(M), Ctx(M.getContext()), TLIGetter(TLIGetter), DTGetter(DTGetter),
        SEGetter(SEGetter), LIGetter(LIGetter),
        IRB(Ctx, ConstantFolder(),
            IRBuilderCallbackInserter(
                [&](Instruction *I) { NewInsts[I] = Epoche; })) {}
  ~InstrumentorIRBuilderTy() {
    for (auto *I : ToBeErased) {
      if (!I->getType()->isVoidTy())
        I->replaceAllUsesWith(PoisonValue::get(I->getType()));
      I->eraseFromParent();
    }
  }

  /// Get a temporary alloca to communicate (large) values with the runtime.
  AllocaInst *getAlloca(Function *Fn, Type *Ty) {
    const DataLayout &DL = Fn->getDataLayout();
    auto &AllocaList = AllocaMap[{Fn, DL.getTypeAllocSize(Ty)}];
    if (AllocaList.empty())
      return new AllocaInst(Ty, DL.getAllocaAddrSpace(), "",
                            Fn->getEntryBlock().begin());
    return AllocaList.pop_back_val();
  }

  /// Return the temporary allocas.
  void returnAllocas(SmallVector<AllocaInst *> &&TmpAllocas) {
    if (TmpAllocas.empty())
      return;

    const DataLayout &DL = TmpAllocas.front()->getDataLayout();
    for (AllocaInst *AI : TmpAllocas) {
      auto &AllocaList = AllocaMap[{
          AI->getFunction(), DL.getTypeAllocSize(AI->getAllocatedType())}];
      AllocaList.push_back(AI);
    }
  }

  DenseMap<Instruction *, HoistKindTy> HoistedInsts;
  DenseSet<Instruction *> HoistableInsts;

  BasicBlock::iterator getBestHoistPoint(BasicBlock::iterator,
                                         HoistKindTy HoistKind);

  DenseMap<Instruction *, std::pair<Value *, Value *>> MinMaxMap;
  struct HoistResult {
    HoistResult(BasicBlock::iterator IP, Value *MinVal, Value *MaxVal)
        : IP(IP), MinVal(MinVal), MaxVal(MaxVal) {}

    BasicBlock::iterator IP;
    Value *MinVal;
    Value *MaxVal;
  };
  HoistResult hoistInstructionsAndAdjustIP(Instruction &I,
                                           HoistKindTy HoistKind,
                                           BasicBlock::iterator IP);

  bool isKnownDereferenceableAccess(Instruction &I, Value &Ptr,
                                    uint32_t AccessSize);

  DenseMap<Value *, std::pair<Value *, Value *>> LoopRangeValueMap;

  std::pair<BasicBlock::iterator, bool>
  computeLoopRangeValues(Value &V, uint32_t AdditionalSize);

  Value *getInitialLoopValue(Value &V) {
    return std::get<0>(LoopRangeValueMap[&V]);
  }
  Value *getFinalLoopValue(Value &V) {
    return std::get<1>(LoopRangeValueMap[&V]);
  }

  /// Commonly used values for IR inspection and creation.
  ///{

  Module &M;

  /// The underying LLVM context.
  LLVMContext &Ctx;

  const DataLayout &DL = M.getDataLayout();

  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *IntptrTy = M.getDataLayout().getIntPtrType(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);
  IntegerType *Int8Ty = Type::getInt8Ty(Ctx);
  IntegerType *Int32Ty = Type::getInt32Ty(Ctx);
  IntegerType *Int64Ty = Type::getInt64Ty(Ctx);
  Constant *NullPtrVal = Constant::getNullValue(PtrTy);
  ///}

  /// Mapping to remember temporary allocas for reuse.
  DenseMap<std::pair<Function *, unsigned>, SmallVector<AllocaInst *>>
      AllocaMap;

  void eraseLater(Instruction *I) { ToBeErased.insert(I); }
  SmallPtrSet<Instruction *, 32> ToBeErased;

  TLIGetterTy TLIGetter;
  DTGetterTy DTGetter;
  SEGetterTy SEGetter;
  LIGetterTy LIGetter;

  IRBuilder<ConstantFolder, IRBuilderCallbackInserter> IRB;
  /// Each instrumentation, i.a., of an instruction, is happening in a dedicated
  /// epoche. The epoche allows to determine if instrumentation instructions
  /// were already around, due to prior instrumentations, or have been
  /// introduced to support the current instrumentation, i.a., compute
  /// information about the current instruction.
  unsigned Epoche = 0;

  /// A mapping from instrumentation instructions to the epoche they have been
  /// created.
  DenseMap<Instruction *, unsigned> NewInsts;
};

using GetterCallbackTy = std::function<Value *(
    Value &, Type &, InstrumentationConfig &, InstrumentorIRBuilderTy &)>;
using SetterCallbackTy = std::function<Value *(
    Value &, Value &, InstrumentationConfig &, InstrumentorIRBuilderTy &)>;

/// An optional callback that takes the global object that is about to be
/// instrumented and can return false if it should be skipped.
using GlobalCallbackTy = std::function<bool(GlobalObject &)>;

struct IRTArg {
  enum IRArgFlagTy {
    NONE = 0,
    STRING = 1 << 0,
    REPLACABLE = 1 << 1,
    REPLACABLE_CUSTOM = 1 << 2,
    POTENTIALLY_INDIRECT = 1 << 3,
    INDIRECT_HAS_SIZE = 1 << 4,
    MIN_MAX_HOISTABLE = 1 << 5,

    LAST,
  };

  IRTArg(Type *Ty, StringRef Name, StringRef Description, unsigned Flags,
         GetterCallbackTy GetterCB, SetterCallbackTy SetterCB = nullptr,
         bool Enabled = true)
      : Enabled(Enabled), Ty(Ty), Name(Name), Description(Description),
        Flags(Flags), GetterCB(std::move(GetterCB)),
        SetterCB(std::move(SetterCB)) {}

  bool Enabled;
  Type *Ty;
  StringRef Name;
  StringRef Description;
  unsigned Flags;
  GetterCallbackTy GetterCB;
  SetterCallbackTy SetterCB;
};

struct InstrumentationCaches {
  DenseMap<std::tuple<unsigned, StringRef, StringRef>, Value *> DirectArgCache;
  DenseMap<std::tuple<unsigned, StringRef, StringRef>, Value *>
      IndirectArgCache;
};

struct IRTCallDescription {
  IRTCallDescription(InstrumentationOpportunity &IConf, Type *RetTy = nullptr);

  std::pair<std::string, std::string>
  createCBodies(InstrumentationConfig &IConf, const DataLayout &DL);

  std::pair<std::string, std::string>
  createCSignature(InstrumentationConfig &IConf, const DataLayout &DL);

  FunctionType *createLLVMSignature(InstrumentationConfig &IConf,
                                    LLVMContext &Ctx, const DataLayout &DL,
                                    bool ForceIndirection);
  CallInst *createLLVMCall(Value *&V, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB, const DataLayout &DL,
                           InstrumentationCaches &ICaches);

  bool isReplacable(IRTArg &IRTA) {
    return (IRTA.Flags & (IRTArg::REPLACABLE | IRTArg::REPLACABLE_CUSTOM));
  }

  bool isPotentiallyIndirect(IRTArg &IRTA) {
    return ((IRTA.Flags & IRTArg::POTENTIALLY_INDIRECT) ||
            ((IRTA.Flags & IRTArg::REPLACABLE) && NumReplaceableArgs > 1));
  }

  bool RequiresIndirection = false;
  bool MightRequireIndirection = false;
  unsigned NumReplaceableArgs = 0;
  InstrumentationOpportunity &IO;
  Type *RetTy = nullptr;
};

struct InstrumentationLocation {

  enum KindTy {
    MODULE_PRE,
    MODULE_POST,
    GLOBAL_PRE,
    GLOBAL_POST,
    FUNCTION_PRE,
    BASIC_BLOCK_PRE,
    BASIC_BLOCK_POST,
    INSTRUCTION_PRE,
    INSTRUCTION_POST,
    SPECIAL_VALUE,
    Last = SPECIAL_VALUE,
  };

  InstrumentationLocation(KindTy Kind) : Kind(Kind) {
    assert(Kind != INSTRUCTION_PRE && Kind != INSTRUCTION_POST &&
           "Opcode required!");
  }

  InstrumentationLocation(unsigned Opcode, bool IsPRE)
      : Kind(IsPRE ? INSTRUCTION_PRE : INSTRUCTION_POST), Opcode(Opcode) {}

  KindTy getKind() const { return Kind; }

  static StringRef getKindStr(KindTy Kind) {
    switch (Kind) {
    case MODULE_PRE:
      return "module_pre";
    case MODULE_POST:
      return "module_post";
    case GLOBAL_PRE:
      return "global_pre";
    case GLOBAL_POST:
      return "global_post";
    case FUNCTION_PRE:
      return "function_pre";
    case BASIC_BLOCK_PRE:
      return "basic_block_pre";
    case BASIC_BLOCK_POST:
      return "basic_block_post";
    case INSTRUCTION_PRE:
      return "instruction_pre";
    case INSTRUCTION_POST:
      return "instruction_post";
    case SPECIAL_VALUE:
      return "special_value";
    }
    llvm_unreachable("Invalid kind!");
  }
  static KindTy getKindFromStr(StringRef S) {
    return StringSwitch<KindTy>(S)
        .Case("module_pre", MODULE_PRE)
        .Case("module_post", MODULE_POST)
        .Case("global_pre", GLOBAL_PRE)
        .Case("global_post", GLOBAL_POST)
        .Case("function_pre", FUNCTION_PRE)
        .Case("basic_block_pre", BASIC_BLOCK_PRE)
        .Case("basic_block_post", BASIC_BLOCK_POST)
        .Case("instruction_pre", INSTRUCTION_PRE)
        .Case("instruction_post", INSTRUCTION_POST)
        .Case("special_value", SPECIAL_VALUE)
        .Default(Last);
  }

  static bool isPRE(KindTy Kind) {
    switch (Kind) {
    case MODULE_PRE:
    case GLOBAL_PRE:
    case FUNCTION_PRE:
    case BASIC_BLOCK_PRE:
    case INSTRUCTION_PRE:
      return true;
    case MODULE_POST:
    case GLOBAL_POST:
    case BASIC_BLOCK_POST:
    case INSTRUCTION_POST:
    case SPECIAL_VALUE:
      return false;
    }
    llvm_unreachable("Invalid kind!");
  }
  bool isPRE() const { return isPRE(Kind); }

  unsigned getOpcode() const {
    assert((Kind == INSTRUCTION_PRE || Kind == INSTRUCTION_POST) &&
           "Expected instruction!");
    return Opcode;
  }

private:
  const KindTy Kind;
  const unsigned Opcode = -1;
};

struct BaseConfigurationOpportunity {
  enum KindTy {
    STRING,
    BOOLEAN,
  };

  static BaseConfigurationOpportunity *getBoolOption(InstrumentationConfig &IC,
                                                     StringRef Name,
                                                     StringRef Description,
                                                     bool B);
  static BaseConfigurationOpportunity *
  getStringOption(InstrumentationConfig &IC, StringRef Name,
                  StringRef Description, StringRef Value);
  union ValueTy {
    bool B;
    int64_t I;
    StringRef S;
  };

  void setBool(bool B) {
    assert(Kind == BOOLEAN && "Not a boolean!");
    V.B = B;
  }
  bool getBool() const {
    assert(Kind == BOOLEAN && "Not a boolean!");
    return V.B;
  }
  void setString(StringRef S) {
    assert(Kind == STRING && "Not a string!");
    V.S = S;
  }
  StringRef getString() const {
    assert(Kind == STRING && "Not a string!");
    return V.S;
  }

  StringRef Name;
  StringRef Description;
  KindTy Kind;
  ValueTy V = {0};
};

struct InstrumentationConfig {

  virtual ~InstrumentationConfig() {}

  InstrumentationConfig() : SS(StringAllocator) {
    RuntimePrefix = BaseConfigurationOpportunity::getStringOption(
        *this, "runtime_prefix", "The runtime API prefix.", "__instrumentor_");
    RuntimeStubsFile = BaseConfigurationOpportunity::getStringOption(
        *this, "runtime_stubs_file",
        "The file into which runtime stubs should be written.", "test.c");
    DemangleFunctionNames = BaseConfigurationOpportunity::getBoolOption(
        *this, "demangle_function_names",
        "Demangle functions names passed to the runtime.", true);
  }

  bool ReadConfig = true;

  virtual void populate(InstrumentorIRBuilderTy &IIRB);
  StringRef getRTName() const { return RuntimePrefix->getString(); }

  std::string getRTName(StringRef Prefix, StringRef Name,
                        StringRef Suffix1 = "", StringRef Suffix2 = "") {
    return (getRTName() + Prefix + Name + Suffix1 + Suffix2).str();
  }

  void addBaseChoice(BaseConfigurationOpportunity *BCO) {
    BaseConfigurationOpportunities.push_back(BCO);
  }
  SmallVector<BaseConfigurationOpportunity *> BaseConfigurationOpportunities;

  BaseConfigurationOpportunity *RuntimePrefix;
  BaseConfigurationOpportunity *RuntimeStubsFile;
  BaseConfigurationOpportunity *DemangleFunctionNames;

  EnumeratedArray<StringMap<InstrumentationOpportunity *>,
                  InstrumentationLocation::KindTy>
      IChoices;
  void addChoice(InstrumentationOpportunity &IO);

  template <typename Ty, typename... ArgsTy>
  static Ty *allocate(ArgsTy &&...Args) {
    static SpecificBumpPtrAllocator<Ty> Allocator;
    Ty *Obj = Allocator.Allocate();
    new (Obj) Ty(std::forward<ArgsTy>(Args)...);
    return Obj;
  }

  BumpPtrAllocator StringAllocator;
  StringSaver SS;

  DenseMap<std::pair<Value *, Function *>, Value *> BasePointerInfoMap;
  Value *getBasePointerInfo(Value &V, InstrumentorIRBuilderTy &IIRB);

  DenseMap<std::pair<const SCEV *, Function *>, Value *> LoopValueRangeMap;
  Value *getLoopValueRange(Value &V, InstrumentorIRBuilderTy &IIRB,
                           uint32_t AdditionalSize);

  /// Mapping to remember global strings passed to the runtime.
  DenseMap<StringRef, Constant *> GlobalStringsMap;

  DenseMap<Constant *, GlobalVariable *> ConstantGlobalsCache;

  Constant *getGlobalString(StringRef S, InstrumentorIRBuilderTy &IIRB) {
    Constant *&V = GlobalStringsMap[SS.save(S)];
    if (!V) {
      auto &M = *IIRB.IRB.GetInsertBlock()->getModule();
      V = IIRB.IRB.CreateGlobalString(
          S, getRTName() + ".str",
          M.getDataLayout().getDefaultGlobalsAddressSpace(), &M);
      if (V->getType() != IIRB.IRB.getPtrTy())
        V = ConstantExpr::getAddrSpaceCast(V, IIRB.IRB.getPtrTy());
    }
    return V;
  }
};

struct InstrumentationOpportunity {
  InstrumentationOpportunity(const InstrumentationLocation IP) : IP(IP) {}
  virtual ~InstrumentationOpportunity() {}

  struct InstrumentationLocation IP;

  SmallVector<IRTArg> IRTArgs;
  bool Enabled = true;

  /// Helpers to cast values, pass them to the runtime, and replace them. To be
  /// used as part of the getter/setter of a InstrumentationOpportunity.
  ///{
  static Value *forceCast(Value &V, Type &Ty, InstrumentorIRBuilderTy &IIRB);
  static Value *getValue(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB) {
    return forceCast(V, Ty, IIRB);
  }

  static Value *replaceValue(Value &V, Value &NewV,
                             InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  ///}

  virtual Value *instrument(Value *&V, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB,
                            InstrumentationCaches &ICaches) {
    if (CB && !CB(*V))
      return nullptr;

    const DataLayout &DL = IIRB.IRB.GetInsertBlock()->getDataLayout();
    IRTCallDescription IRTCallDesc(*this, getRetTy(V->getContext()));
    return IRTCallDesc.createLLVMCall(V, IConf, IIRB, DL, ICaches);
  }

  virtual Type *getRetTy(LLVMContext &Ctx) const { return nullptr; }
  virtual StringRef getName() const = 0;

  unsigned getOpcode() const { return IP.getOpcode(); }
  InstrumentationLocation::KindTy getLocationKind() const {
    return IP.getKind();
  }

  /// An optional callback that takes the value that is about to be
  /// instrumented and can return false if it should be skipped.
  using CallbackTy = std::function<bool(Value &)>;

  CallbackTy CB = nullptr;

  HoistKindTy HoistKind = DO_NOT_HOIST;
};

template <unsigned Opcode>
struct InstructionIO : public InstrumentationOpportunity {
  InstructionIO(bool IsPRE)
      : InstrumentationOpportunity(InstrumentationLocation(Opcode, IsPRE)) {}
  virtual ~InstructionIO() {}

  unsigned getOpcode() const { return Opcode; }

  StringRef getName() const override {
    return Instruction::getOpcodeName(Opcode);
  }
};

struct AllocaIO : public InstructionIO<Instruction::Alloca> {
  AllocaIO(bool IsPRE) : InstructionIO(IsPRE) {}
  virtual ~AllocaIO() {};

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            bool ReplaceAddr = true, bool ReplaceSize = true,
            bool PassAlignment = true) {
    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (!IsPRE && ReplaceAddr)
      IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "address",
                               "The allocated memory address.",
                               ReplaceAddr ? IRTArg::REPLACABLE : IRTArg::NONE,
                               InstrumentationOpportunity::getValue,
                               InstrumentationOpportunity::replaceValue));
    IRTArgs.push_back(
        IRTArg(IntegerType::getInt64Ty(Ctx), "size", "The allocation size.",
               (ReplaceSize && IsPRE) ? IRTArg::REPLACABLE : IRTArg::NONE,
               getSize, setSize));
    IRTArgs.push_back(IRTArg(IntegerType::getInt64Ty(Ctx), "alignment",
                             "The allocation alignment.", IRTArg::NONE,
                             getAlignment));

    IConf.addChoice(*this);
  }

  static Value *getSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                        InstrumentorIRBuilderTy &IIRB);
  static Value *setSize(Value &V, Value &NewV, InstrumentationConfig &IConf,
                        InstrumentorIRBuilderTy &IIRB);
  static Value *getAlignment(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    for (auto IsPRE : {true, false}) {
      auto *AIC = IConf.allocate<AllocaIO>(IsPRE);
      AIC->init(IConf, Ctx);
    }
  }
};

struct StoreIO : public InstructionIO<Instruction::Store> {
  StoreIO(bool IsPRE) : InstructionIO(IsPRE) {}
  virtual ~StoreIO() {};

  struct ConfigTy {
    bool PassPointer = true;
    bool ReplacePointer = true;
    bool PassPointerAS = true;
    bool PassBasePointerInfo = true;
    bool PassLoopValueRangeInfo = true;
    bool PassStoredValue = true;
    bool PassStoredValueSize = true;
    bool PassAlignment = true;
    bool PassValueTypeId = true;
    bool PassAtomicityOrdering = true;
    bool PassSyncScopeId = true;
    bool PassIsVolatile = true;
  } Config;

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;
    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (Config.PassPointer)
      IRTArgs.push_back(
          IRTArg(IIRB.PtrTy, "pointer", "The accessed pointer.",
                 ((IsPRE && Config.ReplacePointer) ? IRTArg::REPLACABLE
                                                   : IRTArg::NONE),
                 getPointer, setPointer));
    if (Config.PassPointerAS)
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "pointer_as",
                               "The address space of the accessed pointer.",
                               IRTArg::NONE, getPointerAS));
    if (Config.PassBasePointerInfo)
      IRTArgs.push_back(IRTArg(IIRB.PtrTy, "base_pointer_info",
                               "The runtime provided base pointer info.",
                               IRTArg::NONE, getBasePointerInfo));
    if (Config.PassLoopValueRangeInfo)
      IRTArgs.push_back(IRTArg(IIRB.PtrTy, "loop_value_range_info",
                               "The runtime provided loop value range info.",
                               IRTArg::NONE, getLoopValueRangeInfo));
    if (Config.PassStoredValue)
      IRTArgs.push_back(
          IRTArg(IIRB.Int64Ty, "value", "The stored value.",
                 IRTArg::POTENTIALLY_INDIRECT |
                     (Config.PassStoredValueSize ? IRTArg::INDIRECT_HAS_SIZE
                                                 : IRTArg::NONE),
                 getValue));
    if (Config.PassStoredValueSize)
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "value_size",
                               "The size of the stored value.", IRTArg::NONE,
                               getValueSize));
    if (Config.PassAlignment)
      IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "alignment",
                               "The known access alignment.", IRTArg::NONE,
                               getAlignment));
    if (Config.PassValueTypeId)
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "value_type_id",
                               "The type id of the stored value.", IRTArg::NONE,
                               getValueTypeId));
    if (Config.PassAtomicityOrdering)
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "atomicity_ordering",
                               "The atomicity ordering of the store.",
                               IRTArg::NONE, getAtomicityOrdering));
    if (Config.PassSyncScopeId)
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "sync_scope_id",
                               "The sync scope id of the store.", IRTArg::NONE,
                               getSyncScopeId));
    if (Config.PassIsVolatile)
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "is_volatile",
                               "Flag indicating a volatile store.",
                               IRTArg::NONE, isVolatile));

    IConf.addChoice(*this);
  }

  static Value *getPointer(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);
  static Value *setPointer(Value &V, Value &NewV, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);
  static Value *getPointerAS(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  static Value *getBasePointerInfo(Value &V, Type &Ty,
                                   InstrumentationConfig &IConf,
                                   InstrumentorIRBuilderTy &IIRB);
  static Value *getLoopValueRangeInfo(Value &V, Type &Ty,
                                      InstrumentationConfig &IConf,
                                      InstrumentorIRBuilderTy &IIRB);
  static Value *getValue(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB);
  static Value *getValueSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  static Value *getAlignment(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  static Value *getValueTypeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB);
  static Value *getAtomicityOrdering(Value &V, Type &Ty,
                                     InstrumentationConfig &IConf,
                                     InstrumentorIRBuilderTy &IIRB);
  static Value *getSyncScopeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB);
  static Value *isVolatile(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    for (auto IsPRE : {true, false}) {
      auto *AIC = IConf.allocate<StoreIO>(IsPRE);
      AIC->init(IConf, IIRB);
    }
  }
};

struct LoadIO : public InstructionIO<Instruction::Load> {
  LoadIO(bool IsPRE) : InstructionIO(IsPRE) {}
  virtual ~LoadIO() {};

  struct ConfigTy {
    bool PassPointer = true;
    bool ReplacePointer = true;
    bool PassPointerAS = true;
    bool PassBasePointerInfo = true;
    bool PassLoopValueRangeInfo = true;
    bool PassValue = true;
    bool ReplaceValue = true;
    bool PassValueSize = true;
    bool PassAlignment = true;
    bool PassValueTypeId = true;
    bool PassAtomicityOrdering = true;
    bool PassSyncScopeId = true;
    bool PassIsVolatile = true;
  } Config;

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
            ConfigTy *UserConfig = nullptr) {
    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (UserConfig)
      Config = *UserConfig;
    if (Config.PassPointer)
      IRTArgs.push_back(
          IRTArg(IIRB.PtrTy, "pointer", "The accessed pointer.",
                 ((IsPRE && Config.ReplacePointer) ? IRTArg::REPLACABLE
                                                   : IRTArg::NONE),
                 getPointer, setPointer));
    if (Config.PassPointerAS)
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "pointer_as",
                               "The address space of the accessed pointer.",
                               IRTArg::NONE, getPointerAS));
    if (Config.PassBasePointerInfo)
      IRTArgs.push_back(IRTArg(IIRB.PtrTy, "base_pointer_info",
                               "The runtime provided base pointer info.",
                               IRTArg::NONE, getBasePointerInfo));
    if (Config.PassLoopValueRangeInfo)
      IRTArgs.push_back(IRTArg(IIRB.PtrTy, "loop_value_range_info",
                               "The runtime provided loop value range info.",
                               IRTArg::NONE, getLoopValueRangeInfo));
    if (!IsPRE && Config.PassValue)
      IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "value", "The loaded value.",
                               IRTArg::REPLACABLE |
                                   IRTArg::POTENTIALLY_INDIRECT |
                                   IRTArg::INDIRECT_HAS_SIZE,
                               getValue, replaceValue));
    if (Config.PassValueSize)
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "value_size",
                               "The size of the loaded value.", IRTArg::NONE,
                               getValueSize));
    if (Config.PassAlignment)
      IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "alignment",
                               "The known access alignment.", IRTArg::NONE,
                               getAlignment));
    if (Config.PassValueTypeId)
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "value_type_id",
                               "The type id of the loaded value.", IRTArg::NONE,
                               getValueTypeId));
    if (Config.PassAtomicityOrdering)
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "atomicity_ordering",
                               "The atomicity ordering of the load.",
                               IRTArg::NONE, getAtomicityOrdering));
    if (Config.PassSyncScopeId)
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "sync_scope_id",
                               "The sync scope id of the load.", IRTArg::NONE,
                               getSyncScopeId));
    if (Config.PassIsVolatile)
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "is_volatile",
                               "Flag indicating a volatile load.", IRTArg::NONE,
                               isVolatile));

    IConf.addChoice(*this);
  }

  static Value *getPointer(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);
  static Value *setPointer(Value &V, Value &NewV, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);
  static Value *getPointerAS(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  static Value *getBasePointerInfo(Value &V, Type &Ty,
                                   InstrumentationConfig &IConf,
                                   InstrumentorIRBuilderTy &IIRB);
  static Value *getLoopValueRangeInfo(Value &V, Type &Ty,
                                      InstrumentationConfig &IConf,
                                      InstrumentorIRBuilderTy &IIRB);
  static Value *getValue(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB);
  static Value *getValueSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  static Value *getAlignment(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  static Value *getValueTypeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB);
  static Value *getAtomicityOrdering(Value &V, Type &Ty,
                                     InstrumentationConfig &IConf,
                                     InstrumentorIRBuilderTy &IIRB);
  static Value *getSyncScopeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB);
  static Value *isVolatile(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    for (auto IsPRE : {true, false}) {
      auto *AIC = IConf.allocate<LoadIO>(IsPRE);
      AIC->init(IConf, IIRB);
    }
  }
};

struct CallIO : public InstructionIO<Instruction::Call> {
  CallIO(bool IsPRE) : InstructionIO(IsPRE) {}
  virtual ~CallIO() {};

  struct ConfigTy {
    bool PassCallee = true;
    bool PassCalleeName = true;
    bool PassIntrinsicId = true;
    bool PassAllocationInfo = true;
    bool PassReturnedValue = true;
    bool PassReturnedValueSize = true;
    bool PassNumParameters = true;
    bool PassParameters = true;
    bool PassIsDefinition = true;
    std::function<bool(Use &)> ArgFilter;
  } Config;

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            ConfigTy *UserConfig = nullptr) {
    using namespace std::placeholders;
    if (UserConfig)
      Config = *UserConfig;
    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (Config.PassCallee)
      IRTArgs.push_back(
          IRTArg(PointerType::getUnqual(Ctx), "callee",
                 "The callee address, or nullptr if an intrinsic.",
                 IRTArg::NONE, getCallee));
    if (Config.PassCalleeName)
      IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "callee_name",
                               "The callee name (if available).",
                               IRTArg::STRING, getCalleeName));
    if (Config.PassIntrinsicId)
      IRTArgs.push_back(IRTArg(IntegerType::getInt64Ty(Ctx), "intrinsic_id",
                               "The intrinsic id, or 0 if not an intrinsic.",
                               IRTArg::NONE, getIntrinsicId));
    if (Config.PassAllocationInfo)
      IRTArgs.push_back(
          IRTArg(PointerType::getUnqual(Ctx), "allocation_info",
                 "Encoding of the allocation made by the call, if "
                 "any, or nullptr otherwise.",
                 IRTArg::NONE, getAllocationInfo));
    if (!IsPRE) {
      if (Config.PassReturnedValue)
        IRTArgs.push_back(IRTArg(
            IntegerType::getInt64Ty(Ctx), "return_value", "The returned value.",
            IRTArg::REPLACABLE | IRTArg::POTENTIALLY_INDIRECT |
                (Config.PassReturnedValueSize ? IRTArg::INDIRECT_HAS_SIZE
                                              : IRTArg::NONE),
            getValue, replaceValue));
      if (Config.PassReturnedValueSize)
        IRTArgs.push_back(IRTArg(
            IntegerType::getInt32Ty(Ctx), "return_value_size",
            "The size of the returned value", IRTArg::NONE, getValueSize));
    }
    if (Config.PassNumParameters)
      IRTArgs.push_back(IRTArg(
          IntegerType::getInt32Ty(Ctx), "num_parameters",
          "Number of call parameters.", IRTArg::NONE,
          std::bind(&CallIO::getNumCallParameters, this, _1, _2, _3, _4)));
    if (Config.PassParameters)
      IRTArgs.push_back(
          IRTArg(PointerType::getUnqual(Ctx), "parameters",
                 "Description of the call parameters.",
                 IsPRE ? IRTArg::REPLACABLE_CUSTOM : IRTArg::NONE,
                 std::bind(&CallIO::getCallParameters, this, _1, _2, _3, _4),
                 std::bind(&CallIO::setCallParameters, this, _1, _2, _3, _4)));
    if (Config.PassIsDefinition)
      IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "is_definition",
                               "Flag to indicate calls to definitions.",
                               IRTArg::NONE, isDefinition));
    IConf.addChoice(*this);
  }

  static Value *getCallee(Value &V, Type &Ty, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB);
  static Value *getCalleeName(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB);
  static Value *getIntrinsicId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB);
  static Value *getAllocationInfo(Value &V, Type &Ty,
                                  InstrumentationConfig &IConf,
                                  InstrumentorIRBuilderTy &IIRB);
  static Value *getValueSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  Value *getNumCallParameters(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB);
  Value *getCallParameters(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);
  Value *setCallParameters(Value &V, Value &NewV, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);
  static Value *isDefinition(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    for (auto IsPRE : {true, false}) {
      auto *AIC = IConf.allocate<CallIO>(IsPRE);
      AIC->init(IConf, Ctx);
    }
  }
};

struct UnreachableIO : public InstructionIO<Instruction::Unreachable> {
  UnreachableIO() : InstructionIO<Instruction::Unreachable>(/*IsPRE*/ true) {}
  virtual ~UnreachableIO() {};

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    IConf.addChoice(*this);
  }

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    auto *AIC = IConf.allocate<UnreachableIO>();
    AIC->init(IConf, Ctx);
  }
};

struct BranchIO : public InstructionIO<Instruction::Br> {
  BranchIO() : InstructionIO<Instruction::Br>(/*IsPRE*/ true) {}
  virtual ~BranchIO(){};

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "is_conditional",
                             "Flag indicating a conditional branch.",
                             IRTArg::NONE, isConditional));
    IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "value",
                             "Value of condition.", IRTArg::REPLACABLE,
                             getValue, setValue));
    IRTArgs.push_back(IRTArg(PointerType::getInt64Ty(Ctx), "num_successors",
                             "Number of branch successors.", IRTArg::NONE,
                             getNumSuccessors));
    IConf.addChoice(*this);
  }

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    auto *AIC = IConf.allocate<BranchIO>();
    AIC->init(IConf, Ctx);
  }

  static Value *isConditional(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB);
  static Value *getValue(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB);
  static Value *setValue(Value &V, Value &NewV, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB);
  static Value *getNumSuccessors(Value &V, Type &Ty,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB);
};

struct ICmpIO : public InstructionIO<Instruction::ICmp> {
  ICmpIO(bool IsPRE) : InstructionIO<Instruction::ICmp>(IsPRE) {}
  virtual ~ICmpIO() {};

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (!IsPRE)
      IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "value",
                               "Result of an integer compare.",
                               IRTArg::REPLACABLE, getValue, replaceValue));
    IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "is_ptr_cmp",
                             "Flag to indicate a pointer compare.",
                             IRTArg::NONE, isPtrCmp));
    IRTArgs.push_back(IRTArg(IntegerType::getInt32Ty(Ctx), "cmp_predicate_kind",
                             "Predicate kind of an integer compare.",
                             IRTArg::NONE, getCmpPredicate));
    IRTArgs.push_back(IRTArg(IntegerType::getInt64Ty(Ctx), "lhs",
                             "Left hand side of an integer compare.",
                             IRTArg::POTENTIALLY_INDIRECT, getLHS));
    IRTArgs.push_back(IRTArg(IntegerType::getInt64Ty(Ctx), "rhs",
                             "Right hand side of an integer compare.",
                             IRTArg::POTENTIALLY_INDIRECT, getRHS));
    IConf.addChoice(*this);
  }

  static Value *getCmpPredicate(Value &V, Type &Ty,
                                InstrumentationConfig &IConf,
                                InstrumentorIRBuilderTy &IIRB);
  static Value *isPtrCmp(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB);
  static Value *getLHS(Value &V, Type &Ty, InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB);
  static Value *getRHS(Value &V, Type &Ty, InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    for (auto IsPRE : {true, false}) {
      auto *AIC = IConf.allocate<ICmpIO>(IsPRE);
      AIC->init(IConf, Ctx);
    }
  }
};

struct PtrToIntIO : public InstructionIO<Instruction::PtrToInt> {
  PtrToIntIO(bool IsPRE) : InstructionIO<Instruction::PtrToInt>(IsPRE) {}
  virtual ~PtrToIntIO() {};

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "pointer",
                             "Input pointer of the ptr to int.",
                             IRTArg::POTENTIALLY_INDIRECT, getPtr));
    if (!IsPRE)
      IRTArgs.push_back(IRTArg(
          IntegerType::getInt64Ty(Ctx), "value", "Result of the ptr to int.",
          IRTArg::REPLACABLE | IRTArg::POTENTIALLY_INDIRECT, getValue,
          replaceValue));
    IConf.addChoice(*this);
  }

  static Value *getPtr(Value &V, Type &Ty, InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    for (auto IsPRE : {true, false}) {
      auto *AIC = IConf.allocate<PtrToIntIO>(IsPRE);
      AIC->init(IConf, Ctx);
    }
  }
};

struct BasePointerIO : public InstrumentationOpportunity {
  BasePointerIO()
      : InstrumentationOpportunity(
            InstrumentationLocation(InstrumentationLocation::SPECIAL_VALUE)) {}
  virtual ~BasePointerIO() {};

  StringRef getName() const override { return "base_pointer_info"; }

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "base_pointer",
                             "The base pointer in question.",
                             IRTArg::REPLACABLE, getValue, setValueNoop));
    IRTArgs.push_back(IRTArg(
        IntegerType::getInt32Ty(Ctx), "base_pointer_kind",
        "The base pointer kind (argument, global, instruction, unknown).",
        IRTArg::NONE, getPointerKind));
    IConf.addChoice(*this);
  }

  static Value *getPointerKind(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB);
  static Value *setValueNoop(Value &V, Value &NewV,
                             InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB) {
    return &NewV;
  }

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    auto *AIC = IConf.allocate<BasePointerIO>();
    AIC->init(IConf, Ctx);
  }

  virtual Type *getRetTy(LLVMContext &Ctx) const override {
    return PointerType::getUnqual(Ctx);
  }
};

struct LoopValueRangeIO : public InstrumentationOpportunity {
  LoopValueRangeIO()
      : InstrumentationOpportunity(
            InstrumentationLocation(InstrumentationLocation::SPECIAL_VALUE)) {}
  virtual ~LoopValueRangeIO() {};

  StringRef getName() const override { return "loop_value_range"; }

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB) {
    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "initial_loop_val",
                             "The value in the first loop iteration.",
                             IRTArg::NONE, getInitialLoopValue));
    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "final_loop_val",
                             "The value in the last loop iteration.",
                             IRTArg::NONE, getFinalLoopValue));
    IConf.addChoice(*this);
  }

  static Value *getInitialLoopValue(Value &V, Type &Ty,
                                    InstrumentationConfig &IConf,
                                    InstrumentorIRBuilderTy &IIRB);
  static Value *getFinalLoopValue(Value &V, Type &Ty,
                                  InstrumentationConfig &IConf,
                                  InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    auto *LVRIO = IConf.allocate<LoopValueRangeIO>();
    LVRIO->init(IConf, IIRB);
  }

  virtual Value *instrument(Value *&V, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB,
                            InstrumentationCaches &ICaches) override {
    if (CB && !CB(*V))
      return nullptr;
    auto [IP, Success] = IIRB.computeLoopRangeValues(*V, AdditionalSize);
    if (!Success)
      return nullptr;
    IIRB.IRB.SetInsertPoint(IP);
    return InstrumentationOpportunity::instrument(V, IConf, IIRB, ICaches);
  }

  virtual Type *getRetTy(LLVMContext &Ctx) const override {
    return PointerType::getUnqual(Ctx);
  }

  void setAdditionalSize(uint32_t AS) { AdditionalSize = AS; }

  uint32_t AdditionalSize = 0;
};

struct FunctionIO : public InstrumentationOpportunity {
  FunctionIO()
      : InstrumentationOpportunity(
            InstrumentationLocation(InstrumentationLocation::FUNCTION_PRE)) {}
  virtual ~FunctionIO() {};

  StringRef getName() const override { return "function"; }

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "address",
                             "The function address.", IRTArg::NONE,
                             getFunctionAddress));
    IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "name",
                             "The function name.", IRTArg::STRING,
                             getFunctionName));
    IRTArgs.push_back(IRTArg(IntegerType::getInt32Ty(Ctx), "num_arguments",
                             "Number of function arguments (without varargs).",
                             IRTArg::NONE, getNumArguments));
    IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "arguments",
                             "Description of the arguments.",
                             IRTArg::REPLACABLE_CUSTOM, getArguments,
                             setArguments));
    IConf.addChoice(*this);
  }

  static Value *getFunctionAddress(Value &V, Type &Ty,
                                   InstrumentationConfig &IConf,
                                   InstrumentorIRBuilderTy &IIRB);
  static Value *getFunctionName(Value &V, Type &Ty,
                                InstrumentationConfig &IConf,
                                InstrumentorIRBuilderTy &IIRB);
  static Value *getNumArguments(Value &V, Type &Ty,
                                InstrumentationConfig &IConf,
                                InstrumentorIRBuilderTy &IIRB);
  static Value *getArguments(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  static Value *setArguments(Value &V, Value &NewV,
                             InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    auto *AIC = IConf.allocate<FunctionIO>();
    AIC->init(IConf, Ctx);
  }
};

struct ModuleIO : public InstrumentationOpportunity {
  ModuleIO(bool IsPRE)
      : InstrumentationOpportunity(InstrumentationLocation(
            IsPRE ? InstrumentationLocation::MODULE_PRE
                  : InstrumentationLocation::MODULE_POST)) {}
  virtual ~ModuleIO() {};

  StringRef getName() const override { return "module"; }

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "module_name",
                             "The module/translation unit name.",
                             IRTArg::STRING, getModuleName));
    IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "name",
                             "The target triple.", IRTArg::STRING,
                             getTargetTriple));
    IConf.addChoice(*this);
  }

  static Value *getModuleName(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB);
  static Value *getTargetTriple(Value &V, Type &Ty,
                                InstrumentationConfig &IConf,
                                InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    for (auto IsPRE : {true, false}) {
      auto *AIC = IConf.allocate<ModuleIO>(IsPRE);
      AIC->init(IConf, Ctx);
    }
  }
};

struct GlobalIO : public InstrumentationOpportunity {
  GlobalIO()
      : InstrumentationOpportunity(
            InstrumentationLocation(InstrumentationLocation::GLOBAL_PRE)) {}
  virtual ~GlobalIO() {};

  StringRef getName() const override { return "globals"; }

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "address",
                             "The address of the global.", IRTArg::REPLACABLE,
                             getAddress, setAddress));
    IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "name",
                             "The name of the global.", IRTArg::STRING,
                             getSymbolName));
    IRTArgs.push_back(
        IRTArg(IntegerType::getInt64Ty(Ctx), "initial_value",
               "The initial value of the global.",
               IRTArg::POTENTIALLY_INDIRECT | IRTArg::INDIRECT_HAS_SIZE,
               getInitialValue));
    IRTArgs.push_back(IRTArg(IntegerType::getInt32Ty(Ctx), "initial_value_size",
                             "The size of the initial value of the global.",
                             IRTArg::NONE, getInitialValueSize));
    IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "is_constant",
                             "Flag to indicate constant globals.", IRTArg::NONE,
                             isConstant));
    IConf.addChoice(*this);
  }

  static Value *getAddress(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);
  static Value *setAddress(Value &V, Value &NewV, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);
  static Value *getSymbolName(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB);
  static Value *getInitialValue(Value &V, Type &Ty,
                                InstrumentationConfig &IConf,
                                InstrumentorIRBuilderTy &IIRB);
  static Value *getInitialValueSize(Value &V, Type &Ty,
                                    InstrumentationConfig &IConf,
                                    InstrumentorIRBuilderTy &IIRB);
  static Value *isConstant(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    auto *AIC = IConf.allocate<GlobalIO>();
    AIC->init(IConf, Ctx);
  }
};

} // namespace instrumentor

class InstrumentorPass : public PassInfoMixin<InstrumentorPass> {
  using InstrumentationConfig = instrumentor::InstrumentationConfig;
  InstrumentationConfig *UserIConf;

public:
  InstrumentorPass(InstrumentationConfig *IC = nullptr) : UserIConf(IC) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_INSTRUMENTOR_H
