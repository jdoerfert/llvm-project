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
#include "llvm/IR/ValueHandle.h"
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

template <typename IRBuilderTy> void ensureDbgLoc(IRBuilderTy &IRB) {
  if (IRB.getCurrentDebugLocation())
    return;
  auto *BB = IRB.GetInsertBlock();
  if (auto *SP = BB->getParent()->getSubprogram())
    IRB.SetCurrentDebugLocation(DILocation::get(BB->getContext(), 0, 0, SP));
}

Value *getUnderlyingObjectRecursive(Value *Ptr);

template <typename IRBTy>
Value *tryToCast(IRBTy &IRB, Value *V, Type *Ty, const DataLayout &DL,
                 bool AllowTruncate = false) {
  if (!V)
    return Constant::getAllOnesValue(Ty);
  auto *VTy = V->getType();
  if (VTy == Ty)
    return V;
  if (VTy->isAggregateType())
    return V;
  auto RequestedSize = DL.getTypeSizeInBits(Ty);
  auto ValueSize = DL.getTypeSizeInBits(VTy);
  bool IsTruncate = RequestedSize < ValueSize;
  if (IsTruncate && !AllowTruncate)
    return V;
  if (IsTruncate && AllowTruncate)
    return tryToCast(IRB,
                     IRB.CreateIntCast(V, IRB.getIntNTy(RequestedSize),
                                       /*IsSigned=*/false),
                     Ty, DL, AllowTruncate);
  if (VTy->isPointerTy() && Ty->isPointerTy())
    return IRB.CreatePointerBitCastOrAddrSpaceCast(V, Ty);
  if (VTy->isIntegerTy() && Ty->isIntegerTy())
    return IRB.CreateIntCast(V, Ty, /*IsSigned=*/false);
  if (VTy->isFloatingPointTy() && Ty->isIntOrPtrTy()) {
    switch (ValueSize) {
    case 64:
      return tryToCast(IRB, IRB.CreateBitCast(V, IRB.getInt64Ty()), Ty, DL,
                       AllowTruncate);
    case 32:
      return tryToCast(IRB, IRB.CreateBitCast(V, IRB.getInt32Ty()), Ty, DL,
                       AllowTruncate);
    case 16:
      return tryToCast(IRB, IRB.CreateBitCast(V, IRB.getInt16Ty()), Ty, DL,
                       AllowTruncate);
    case 8:
      return tryToCast(IRB, IRB.CreateBitCast(V, IRB.getInt8Ty()), Ty, DL,
                       AllowTruncate);
    default:
      llvm_unreachable("unsupported floating point size");
    }
  }
  return IRB.CreateBitOrPointerCast(V, Ty);
}

struct InstrumentationConfig;
struct InstrumentationOpportunity;

struct InstrumentorIRBuilderTy {

  InstrumentorIRBuilderTy(Module &M, FunctionAnalysisManager &FAM)
      : M(M), Ctx(M.getContext()), FAM(FAM),
        IRB(Ctx, ConstantFolder(),
            IRBuilderCallbackInserter(
                [&](Instruction *I) { NewInsts[I] = Epoche; })) {}
  ~InstrumentorIRBuilderTy() {
    for (auto &V : ToBeErased) {
      if (!V)
	continue;
      if (!V->getType()->isVoidTy())
        V->replaceAllUsesWith(PoisonValue::get(V->getType()));
      cast<Instruction>(V)->eraseFromParent();
    }
    ToBeErased.clear();
  }

  /// Get a temporary alloca to communicate (large) values with the runtime.
  AllocaInst *getAlloca(Function *Fn, Type *Ty, bool MatchType = false) {
    const DataLayout &DL = Fn->getDataLayout();
    auto *&AllocaList = AllocaMap[{Fn, DL.getTypeAllocSize(Ty)}];
    if (!AllocaList)
      AllocaList = new AllocaListTy;
    AllocaInst *AI = nullptr;
    for (auto *&ListAI : *AllocaList) {
      if (MatchType && ListAI->getAllocatedType() != Ty)
        continue;
      AI = ListAI;
      ListAI = *AllocaList->rbegin();
      break;
    }
    if (AI)
      AllocaList->pop_back();
    else
      AI = new AllocaInst(Ty, DL.getAllocaAddrSpace(), "",
                          Fn->getEntryBlock().begin());
    UsedAllocas[AI] = AllocaList;
    return AI;
  }

  /// Return the temporary allocas.
  void returnAllocas() {
    for (auto [AI, List] : UsedAllocas)
      List->push_back(AI);
    UsedAllocas.clear();
  }

  DenseMap<Instruction *, HoistKindTy> HoistedInsts;

  BasicBlock::iterator getBestHoistPoint(BasicBlock::iterator,
                                         HoistKindTy HoistKind);

  DenseMap<Instruction *, std::pair<Value *, Value *>> MinMaxMap;
  static BasicBlock::iterator
  hoistInstructionsAndAdjustIP(Instruction &I, BasicBlock::iterator IP,
                               DominatorTree &DT, bool ForceInitial = false);

  bool isKnownDereferenceableAccess(Instruction &I, Value &Ptr,
                                    uint32_t AccessSize);

  struct LoopRangeInfo {
    Value *Min;
    Value *Max;
    uint32_t AdditionalSize;
  };
  DenseMap<Value *, LoopRangeInfo> LoopRangeInfoMap;

  std::pair<BasicBlock::iterator, bool>
  computeLoopRangeValues(Value &V, uint32_t AdditionalSize);

  SmallVector<Instruction *> GeneratedInsts;

  Value *getInitialLoopValue(Value &V) { return LoopRangeInfoMap[&V].Min; }
  Value *getFinalLoopValue(Value &V) { return LoopRangeInfoMap[&V].Max; }
  Value *getMaxOffset(Value &V, Type &Ty) {
    return ConstantInt::get(&Ty, LoopRangeInfoMap[&V].AdditionalSize);
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
  using AllocaListTy = SmallVector<AllocaInst *>;
  DenseMap<std::pair<Function *, unsigned>, AllocaListTy *> AllocaMap;
  DenseMap<AllocaInst *, SmallVector<AllocaInst *> *> UsedAllocas;

  void eraseLater(Instruction *I) { ToBeErased.insert(I); }
  SmallDenseSet<WeakVH, 32> ToBeErased;

  FunctionAnalysisManager &FAM;

  template <typename T, typename R = typename T::Result>
  R &analysisGetter(Function &F) {
    return FAM.getResult<T>(F);
  }

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
         bool Enabled = true, bool NoCache = false)
      : Enabled(Enabled), Ty(Ty), Name(Name), Description(Description),
        Flags(Flags), GetterCB(std::move(GetterCB)),
        SetterCB(std::move(SetterCB)), NoCache(NoCache) {}

  bool Enabled;
  Type *Ty;
  StringRef Name;
  StringRef Description;
  unsigned Flags;
  GetterCallbackTy GetterCB;
  SetterCallbackTy SetterCB;
  bool NoCache;
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
    FUNCTION_POST,
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
    case FUNCTION_POST:
      return "function_post";
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
        .Case("function_post", FUNCTION_POST)
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
    case FUNCTION_POST:
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

struct InstrumentorIRBuilderTy;
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
    TargetRegex = BaseConfigurationOpportunity::getStringOption(
        *this, "target_regex",
        "Regular expression to be matched against the module target. "
        "Only targets that match this regex will be instrumented",
        "");
    HostEnabled = BaseConfigurationOpportunity::getBoolOption(
        *this, "host_enabled", "Instrument non-GPU targets", true);
    GPUEnabled = BaseConfigurationOpportunity::getBoolOption(
        *this, "gpu_enabled", "Instrument GPU targets", true);
    RuntimeBitcode = BaseConfigurationOpportunity::getStringOption(
        *this, "runtime_bitcode", "Link Runtime Bitcode", "");
    InlineRuntimeEagerly = BaseConfigurationOpportunity::getBoolOption(
        *this, "inline_runtime", "Inline Runtime Eagerly", true);
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
  BaseConfigurationOpportunity *TargetRegex;
  BaseConfigurationOpportunity *HostEnabled;
  BaseConfigurationOpportunity *GPUEnabled;
  BaseConfigurationOpportunity *RuntimeBitcode;
  BaseConfigurationOpportunity *InlineRuntimeEagerly;

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

  DenseMap<Value *, Value *> UnderlyingObjsMap;

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

  virtual void startFunction() {}
};

template <typename EnumTy> struct BaseConfigTy {
  std::bitset<static_cast<int>(EnumTy::NumConfig)> Options;

  BaseConfigTy(bool Enable = true) {
    if (Enable)
      Options.set();
  }

  bool has(EnumTy Opt) const { return Options.test(static_cast<int>(Opt)); }
  void set(EnumTy Opt, bool Value = true) {
    Options.set(static_cast<int>(Opt), Value);
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
    auto *CI = IRTCallDesc.createLLVMCall(V, IConf, IIRB, DL, ICaches);
    return CI;
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

  static Value *getIdPre(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB);
  static Value *getIdPost(Value &V, Type &Ty, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB);

  void addCommonArgs(InstrumentationConfig &IConf, LLVMContext &Ctx,
                     bool PassId) {
    const auto CB = IP.isPRE() ? getIdPre : getIdPost;
    if (PassId)
      IRTArgs.push_back(
          IRTArg(IntegerType::getInt32Ty(Ctx), "id",
                 "A unique ID associated with the given instrumentor call",
                 IRTArg::NONE, CB, nullptr, true, true));
  }
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

  enum ConfigKind {
    PassAddress = 0,
    ReplaceAddress,
    PassSize,
    ReplaceSize,
    PassAlignment,
    PassId,
    NumConfig,
  };

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;

    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (!IsPRE && Config.has(PassAddress))
      IRTArgs.push_back(
          IRTArg(PointerType::getUnqual(Ctx), "address",
                 "The allocated memory address.",
                 Config.has(ReplaceAddress) ? IRTArg::REPLACABLE : IRTArg::NONE,
                 InstrumentationOpportunity::getValue,
                 InstrumentationOpportunity::replaceValue));
    if (Config.has(PassSize))
      IRTArgs.push_back(
          IRTArg(IntegerType::getInt64Ty(Ctx), "size", "The allocation size.",
                 (IsPRE && Config.has(ReplaceSize)) ? IRTArg::REPLACABLE
                                                    : IRTArg::NONE,
                 getSize, setSize));
    if (Config.has(PassAlignment))
      IRTArgs.push_back(IRTArg(IntegerType::getInt64Ty(Ctx), "alignment",
                               "The allocation alignment.", IRTArg::NONE,
                               getAlignment));

    addCommonArgs(IConf, Ctx, Config.has(PassId));
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

  enum ConfigKind {
    PassPointer = 0,
    ReplacePointer,
    PassPointerAS,
    PassBasePointerInfo,
    PassLoopValueRangeInfo,
    PassStoredValue,
    PassStoredValueSize,
    PassAlignment,
    PassValueTypeId,
    PassAtomicityOrdering,
    PassSyncScopeId,
    PassIsVolatile,
    PassId,
    NumConfig,
  };

  virtual Type *getValueType(LLVMContext &Ctx) const {
    return IntegerType::getInt64Ty(Ctx);
  }

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;

    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (Config.has(PassPointer))
      IRTArgs.push_back(
          IRTArg(IIRB.PtrTy, "pointer", "The accessed pointer.",
                 ((IsPRE && Config.has(ReplacePointer)) ? IRTArg::REPLACABLE
                                                        : IRTArg::NONE),
                 getPointer, setPointer));
    if (Config.has(PassPointerAS))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "pointer_as",
                               "The address space of the accessed pointer.",
                               IRTArg::NONE, getPointerAS));
    if (Config.has(PassBasePointerInfo))
      IRTArgs.push_back(IRTArg(IIRB.PtrTy, "base_pointer_info",
                               "The runtime provided base pointer info.",
                               IRTArg::NONE, getBasePointerInfo));
    if (Config.has(PassLoopValueRangeInfo))
      IRTArgs.push_back(IRTArg(IIRB.PtrTy, "loop_value_range_info",
                               "The runtime provided loop value range info.",
                               IRTArg::NONE, getLoopValueRangeInfo));
    if (Config.has(PassStoredValue))
      IRTArgs.push_back(
          IRTArg(getValueType(IIRB.Ctx), "value", "The stored value.",
                 IRTArg::POTENTIALLY_INDIRECT | (Config.has(PassStoredValueSize)
                                                     ? IRTArg::INDIRECT_HAS_SIZE
                                                     : IRTArg::NONE),
                 getValue));
    if (Config.has(PassStoredValueSize))
      IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "value_size",
                               "The size of the stored value.", IRTArg::NONE,
                               getValueSize));
    if (Config.has(PassAlignment))
      IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "alignment",
                               "The known access alignment.", IRTArg::NONE,
                               getAlignment));
    if (Config.has(PassValueTypeId))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "value_type_id",
                               "The type id of the stored value.", IRTArg::NONE,
                               getValueTypeId));
    if (Config.has(PassAtomicityOrdering))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "atomicity_ordering",
                               "The atomicity ordering of the store.",
                               IRTArg::NONE, getAtomicityOrdering));
    if (Config.has(PassSyncScopeId))
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "sync_scope_id",
                               "The sync scope id of the store.", IRTArg::NONE,
                               getSyncScopeId));
    if (Config.has(PassIsVolatile))
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "is_volatile",
                               "Flag indicating a volatile store.",
                               IRTArg::NONE, isVolatile));

    addCommonArgs(IConf, IIRB.Ctx, Config.has(PassId));
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

  enum ConfigKind {
    PassPointer = 0,
    ReplacePointer,
    PassPointerAS,
    PassBasePointerInfo,
    PassLoopValueRangeInfo,
    PassValue,
    ReplaceValue,
    PassValueSize,
    PassAlignment,
    PassValueTypeId,
    PassAtomicityOrdering,
    PassSyncScopeId,
    PassIsVolatile,
    PassId,
    NumConfig,
  };

  virtual Type *getValueType(LLVMContext &Ctx) const {
    return IntegerType::getInt64Ty(Ctx);
  }

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
            ConfigTy *UserConfig = nullptr) {
    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (UserConfig)
      Config = *UserConfig;
    if (Config.has(PassPointer))
      IRTArgs.push_back(
          IRTArg(IIRB.PtrTy, "pointer", "The accessed pointer.",
                 ((IsPRE && Config.has(ReplacePointer)) ? IRTArg::REPLACABLE
                                                        : IRTArg::NONE),
                 getPointer, setPointer));
    if (Config.has(PassPointerAS))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "pointer_as",
                               "The address space of the accessed pointer.",
                               IRTArg::NONE, getPointerAS));
    if (Config.has(PassBasePointerInfo))
      IRTArgs.push_back(IRTArg(IIRB.PtrTy, "base_pointer_info",
                               "The runtime provided base pointer info.",
                               IRTArg::NONE, getBasePointerInfo));
    if (Config.has(PassLoopValueRangeInfo))
      IRTArgs.push_back(IRTArg(IIRB.PtrTy, "loop_value_range_info",
                               "The runtime provided loop value range info.",
                               IRTArg::NONE, getLoopValueRangeInfo));
    if (!IsPRE && Config.has(PassValue))
      IRTArgs.push_back(IRTArg(
          getValueType(IIRB.Ctx), "value", "The loaded value.",
          Config.has(ReplaceValue)
              ? IRTArg::REPLACABLE | IRTArg::POTENTIALLY_INDIRECT |
                    (Config.has(PassValueSize) ? IRTArg::INDIRECT_HAS_SIZE
                                               : IRTArg::NONE)
              : IRTArg::NONE,
          getValue, Config.has(ReplaceValue) ? replaceValue : nullptr));
    if (Config.has(PassValueSize))
      IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "value_size",
                               "The size of the loaded value.", IRTArg::NONE,
                               getValueSize));
    if (Config.has(PassAlignment))
      IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "alignment",
                               "The known access alignment.", IRTArg::NONE,
                               getAlignment));
    if (Config.has(PassValueTypeId))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "value_type_id",
                               "The type id of the loaded value.", IRTArg::NONE,
                               getValueTypeId));
    if (Config.has(PassAtomicityOrdering))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "atomicity_ordering",
                               "The atomicity ordering of the load.",
                               IRTArg::NONE, getAtomicityOrdering));
    if (Config.has(PassSyncScopeId))
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "sync_scope_id",
                               "The sync scope id of the load.", IRTArg::NONE,
                               getSyncScopeId));
    if (Config.has(PassIsVolatile))
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "is_volatile",
                               "Flag indicating a volatile load.", IRTArg::NONE,
                               isVolatile));
    addCommonArgs(IConf, IIRB.Ctx, Config.has(PassId));
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

struct AtomicRMWIO : public InstructionIO<Instruction::AtomicRMW> {
  AtomicRMWIO(bool IsPRE) : InstructionIO(IsPRE) {}
  virtual ~AtomicRMWIO() {};

  enum ConfigKind {
    PassPointer = 0,
    ReplacePointer,
    PassPointerAS,
    PassBasePointerInfo,
    PassLoopValueRangeInfo,
    PassStoredValue,
    PassStoredValueSize,
    PassAlignment,
    PassValueTypeId,
    PassAtomicityOrdering,
    PassSyncScopeId,
    PassIsVolatile,
    PassId,
    NumConfig,
  };

  virtual Type *getValueType(LLVMContext &Ctx) const {
    return IntegerType::getInt64Ty(Ctx);
  }

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;

    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (Config.has(PassPointer))
      IRTArgs.push_back(
          IRTArg(IIRB.PtrTy, "pointer", "The accessed pointer.",
                 ((IsPRE && Config.has(ReplacePointer)) ? IRTArg::REPLACABLE
                                                        : IRTArg::NONE),
                 getPointer, setPointer));
    if (Config.has(PassPointerAS))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "pointer_as",
                               "The address space of the accessed pointer.",
                               IRTArg::NONE, getPointerAS));
    if (Config.has(PassBasePointerInfo))
      IRTArgs.push_back(IRTArg(IIRB.PtrTy, "base_pointer_info",
                               "The runtime provided base pointer info.",
                               IRTArg::NONE, getBasePointerInfo));
    if (Config.has(PassLoopValueRangeInfo))
      IRTArgs.push_back(IRTArg(IIRB.PtrTy, "loop_value_range_info",
                               "The runtime provided loop value range info.",
                               IRTArg::NONE, getLoopValueRangeInfo));
    if (Config.has(PassStoredValue))
      IRTArgs.push_back(
          IRTArg(getValueType(IIRB.Ctx), "value", "The stored value.",
                 IRTArg::POTENTIALLY_INDIRECT | (Config.has(PassStoredValueSize)
                                                     ? IRTArg::INDIRECT_HAS_SIZE
                                                     : IRTArg::NONE),
                 getValue));
    if (Config.has(PassStoredValueSize))
      IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "value_size",
                               "The size of the stored value.", IRTArg::NONE,
                               getValueSize));
    if (Config.has(PassAlignment))
      IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "alignment",
                               "The known access alignment.", IRTArg::NONE,
                               getAlignment));
    if (Config.has(PassValueTypeId))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "value_type_id",
                               "The type id of the stored value.", IRTArg::NONE,
                               getValueTypeId));
    if (Config.has(PassAtomicityOrdering))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "atomicity_ordering",
                               "The atomicity ordering of the store.",
                               IRTArg::NONE, getAtomicityOrdering));
    if (Config.has(PassSyncScopeId))
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "sync_scope_id",
                               "The sync scope id of the store.", IRTArg::NONE,
                               getSyncScopeId));
    if (Config.has(PassIsVolatile))
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "is_volatile",
                               "Flag indicating a volatile store.",
                               IRTArg::NONE, isVolatile));

    addCommonArgs(IConf, IIRB.Ctx, Config.has(PassId));
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
      auto *AIC = IConf.allocate<AtomicRMWIO>(IsPRE);
      AIC->init(IConf, IIRB);
    }
  }
};

struct CallIO : public InstructionIO<Instruction::Call> {
  CallIO(bool IsPRE) : InstructionIO(IsPRE) {}
  virtual ~CallIO() {};

  enum ConfigKind {
    PassCallee,
    PassCalleeName,
    PassIntrinsicId,
    PassAllocationInfo,
    PassReturnedValue,
    PassReturnedValueSize,
    PassNumParameters,
    PassParameters,
    PassIsDefinition,
    PassId,
    NumConfig,
  };

  struct ConfigTy final : public BaseConfigTy<ConfigKind> {
    std::function<bool(Use &)> ArgFilter;

    ConfigTy(bool Enable = true) : BaseConfigTy(Enable) {}
  } Config;

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            ConfigTy *UserConfig = nullptr) {
    using namespace std::placeholders;
    if (UserConfig)
      Config = *UserConfig;
    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (Config.has(PassCallee))
      IRTArgs.push_back(
          IRTArg(PointerType::getUnqual(Ctx), "callee",
                 "The callee address, or nullptr if an intrinsic.",
                 IRTArg::NONE, getCallee));
    if (Config.has(PassCalleeName))
      IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "callee_name",
                               "The callee name (if available).",
                               IRTArg::STRING, getCalleeName));
    if (Config.has(PassIntrinsicId))
      IRTArgs.push_back(IRTArg(IntegerType::getInt64Ty(Ctx), "intrinsic_id",
                               "The intrinsic id, or 0 if not an intrinsic.",
                               IRTArg::NONE, getIntrinsicId));
    if (Config.has(PassAllocationInfo))
      IRTArgs.push_back(
          IRTArg(PointerType::getUnqual(Ctx), "allocation_info",
                 "Encoding of the allocation made by the call, if "
                 "any, or nullptr otherwise.",
                 IRTArg::NONE, getAllocationInfo));
    if (!IsPRE) {
      if (Config.has(PassReturnedValue)) {
        Type *ReturnValueTy = IntegerType::getInt64Ty(Ctx);
        if (auto *RTy = getRetTy(Ctx))
          ReturnValueTy = RTy;
        IRTArgs.push_back(IRTArg(
            ReturnValueTy, "return_value", "The returned value.",
            IRTArg::REPLACABLE | IRTArg::POTENTIALLY_INDIRECT |
                (Config.has(PassReturnedValueSize) ? IRTArg::INDIRECT_HAS_SIZE
                                                   : IRTArg::NONE),
            getValue, replaceValue));
      }
      if (Config.has(PassReturnedValueSize))
        IRTArgs.push_back(IRTArg(
            IntegerType::getInt32Ty(Ctx), "return_value_size",
            "The size of the returned value", IRTArg::NONE, getValueSize));
    }
    if (Config.has(PassNumParameters))
      IRTArgs.push_back(IRTArg(
          IntegerType::getInt32Ty(Ctx), "num_parameters",
          "Number of call parameters.", IRTArg::NONE,
          std::bind(&CallIO::getNumCallParameters, this, _1, _2, _3, _4)));
    if (Config.has(PassParameters))
      IRTArgs.push_back(
          IRTArg(PointerType::getUnqual(Ctx), "parameters",
                 "Description of the call parameters.",
                 IsPRE ? IRTArg::REPLACABLE_CUSTOM : IRTArg::NONE,
                 std::bind(&CallIO::getCallParameters, this, _1, _2, _3, _4),
                 std::bind(&CallIO::setCallParameters, this, _1, _2, _3, _4)));
    if (Config.has(PassIsDefinition))
      IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "is_definition",
                               "Flag to indicate calls to definitions.",
                               IRTArg::NONE, isDefinition));
    addCommonArgs(IConf, Ctx, Config.has(PassId));
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

struct InvokeIO : public InstructionIO<Instruction::Invoke> {
  InvokeIO(bool IsPRE) : InstructionIO(IsPRE) {}
  virtual ~InvokeIO() {};

  enum ConfigKind {
    PassCallee,
    PassCalleeName,
    PassIntrinsicId,
    PassAllocationInfo,
    PassReturnedValue,
    PassReturnedValueSize,
    PassNumParameters,
    PassParameters,
    PassIsDefinition,
    PassId,
    NumConfig,
  };

  struct ConfigTy final : public BaseConfigTy<ConfigKind> {
    std::function<bool(Use &)> ArgFilter;

    ConfigTy(bool Enable = true) : BaseConfigTy(Enable) {}
  } Config;

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            ConfigTy *UserConfig = nullptr) {
    using namespace std::placeholders;
    if (UserConfig)
      Config = *UserConfig;
    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (Config.has(PassCallee))
      IRTArgs.push_back(
          IRTArg(PointerType::getUnqual(Ctx), "callee",
                 "The callee address, or nullptr if an intrinsic.",
                 IRTArg::NONE, getCallee));
    if (Config.has(PassCalleeName))
      IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "callee_name",
                               "The callee name (if available).",
                               IRTArg::STRING, getCalleeName));
    if (Config.has(PassIntrinsicId))
      IRTArgs.push_back(IRTArg(IntegerType::getInt64Ty(Ctx), "intrinsic_id",
                               "The intrinsic id, or 0 if not an intrinsic.",
                               IRTArg::NONE, getIntrinsicId));
    if (Config.has(PassAllocationInfo))
      IRTArgs.push_back(
          IRTArg(PointerType::getUnqual(Ctx), "allocation_info",
                 "Encoding of the allocation made by the call, if "
                 "any, or nullptr otherwise.",
                 IRTArg::NONE, getAllocationInfo));
    if (!IsPRE) {
      if (Config.has(PassReturnedValue)) {
        Type *ReturnValueTy = IntegerType::getInt64Ty(Ctx);
        if (auto *RTy = getRetTy(Ctx))
          ReturnValueTy = RTy;
        IRTArgs.push_back(IRTArg(
            ReturnValueTy, "return_value", "The returned value.",
            IRTArg::REPLACABLE | IRTArg::POTENTIALLY_INDIRECT |
                (Config.has(PassReturnedValueSize) ? IRTArg::INDIRECT_HAS_SIZE
                                                   : IRTArg::NONE),
            getValue, replaceValue));
      }
      if (Config.has(PassReturnedValueSize))
        IRTArgs.push_back(IRTArg(
            IntegerType::getInt32Ty(Ctx), "return_value_size",
            "The size of the returned value", IRTArg::NONE, getValueSize));
    }
    if (Config.has(PassNumParameters))
      IRTArgs.push_back(IRTArg(
          IntegerType::getInt32Ty(Ctx), "num_parameters",
          "Number of call parameters.", IRTArg::NONE,
          std::bind(&InvokeIO::getNumCallParameters, this, _1, _2, _3, _4)));
    if (Config.has(PassParameters))
      IRTArgs.push_back(IRTArg(
          PointerType::getUnqual(Ctx), "parameters",
          "Description of the call parameters.",
          IsPRE ? IRTArg::REPLACABLE_CUSTOM : IRTArg::NONE,
          std::bind(&InvokeIO::getCallParameters, this, _1, _2, _3, _4),
          std::bind(&InvokeIO::setCallParameters, this, _1, _2, _3, _4)));
    if (Config.has(PassIsDefinition))
      IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "is_definition",
                               "Flag to indicate calls to definitions.",
                               IRTArg::NONE, isDefinition));
    addCommonArgs(IConf, Ctx, Config.has(PassId));
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
    auto *AIC = IConf.allocate<InvokeIO>(/*IsPRE=*/true);
    AIC->init(IConf, Ctx);
  }
};

struct UnreachableIO : public InstructionIO<Instruction::Unreachable> {
  UnreachableIO() : InstructionIO<Instruction::Unreachable>(/*IsPRE*/ true) {}
  virtual ~UnreachableIO() {};

  enum ConfigKind {
    PassId,
    NumConfig,
  };

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;
    addCommonArgs(IConf, Ctx, Config.has(PassId));
    IConf.addChoice(*this);
  }

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx,
                       ConfigTy *UserConfig = nullptr) {
    auto *AIC = IConf.allocate<UnreachableIO>();
    AIC->init(IConf, Ctx, UserConfig);
  }
};

struct BranchIO : public InstructionIO<Instruction::Br> {
  BranchIO() : InstructionIO<Instruction::Br>(/*IsPRE*/ true) {}
  virtual ~BranchIO() {};

  enum ConfigKind {
    PassId,
    NumConfig,
  };

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;
    IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "is_conditional",
                             "Flag indicating a conditional branch.",
                             IRTArg::NONE, isConditional));
    IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "value",
                             "Value of condition.", IRTArg::REPLACABLE,
                             getValue, setValue));
    IRTArgs.push_back(IRTArg(PointerType::getInt64Ty(Ctx), "num_successors",
                             "Number of branch successors.", IRTArg::NONE,
                             getNumSuccessors));
    addCommonArgs(IConf, Ctx, Config.has(PassId));
    IConf.addChoice(*this);
  }

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx,
                       ConfigTy *UserConfig = nullptr) {
    auto *AIC = IConf.allocate<BranchIO>();
    AIC->init(IConf, Ctx, UserConfig);
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

  enum ConfigKind {
    PassValue,
    ReplaceValue,
    PassIsPtrCmp,
    PassCmpPredicate,
    PassLHS,
    PassRHS,
    PassId,
    NumConfig,
  };

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  virtual Type *getValueType(LLVMContext &Ctx) const {
    return IntegerType::getInt64Ty(Ctx);
  }

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;
    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (!IsPRE && Config.has(PassValue))
      IRTArgs.push_back(
          IRTArg(IntegerType::getInt8Ty(Ctx), "value",
                 "Result of an integer compare.", IRTArg::REPLACABLE, getValue,
                 Config.has(ReplaceValue) ? replaceValue : nullptr));
    if (Config.has(PassIsPtrCmp))
      IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "is_ptr_cmp",
                               "Flag to indicate a pointer compare.",
                               IRTArg::NONE, isPtrCmp));
    if (Config.has(PassCmpPredicate))
      IRTArgs.push_back(IRTArg(IntegerType::getInt32Ty(Ctx),
                               "cmp_predicate_kind",
                               "Predicate kind of an integer compare.",
                               IRTArg::NONE, getCmpPredicate));
    if (Config.has(PassLHS))
      IRTArgs.push_back(IRTArg(getValueType(Ctx), "lhs",
                               "Left hand side of an integer compare.",
                               IRTArg::POTENTIALLY_INDIRECT, getLHS));
    if (Config.has(PassRHS))
      IRTArgs.push_back(IRTArg(getValueType(Ctx), "rhs",
                               "Right hand side of an integer compare.",
                               IRTArg::POTENTIALLY_INDIRECT, getRHS));
    addCommonArgs(IConf, Ctx, Config.has(PassId));
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

struct VAArgIO : public InstructionIO<Instruction::VAArg> {
  VAArgIO(bool IsPRE) : InstructionIO<Instruction::VAArg>(IsPRE) {}
  virtual ~VAArgIO() {};

  enum ConfigKind {
    PassPointer,
    ReplacePointer,
    PassId,
    NumConfig,
  };

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;
    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (Config.has(PassPointer))
      IRTArgs.push_back(IRTArg(
          PointerType::get(Ctx, 0), "pointer",
          "Pointer of the va_arg instruction.", IRTArg::REPLACABLE, getPointer,
          IsPRE && Config.has(ReplacePointer) ? setPointer : nullptr));
    addCommonArgs(IConf, Ctx, Config.has(PassId));
    IConf.addChoice(*this);
  }

  static Value *getPointer(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);
  static Value *setPointer(Value &V, Value &NewV, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    for (auto IsPRE : {true, false}) {
      auto *AIC = IConf.allocate<VAArgIO>(IsPRE);
      AIC->init(IConf, Ctx);
    }
  }
};

struct PtrToIntIO : public InstructionIO<Instruction::PtrToInt> {
  PtrToIntIO(bool IsPRE) : InstructionIO<Instruction::PtrToInt>(IsPRE) {}
  virtual ~PtrToIntIO() {};

  enum ConfigKind {
    PassPointer,
    PassResult,
    ReplaceResult,
    PassId,
    NumConfig,
  };

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;
    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (Config.has(PassPointer))
      IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "pointer",
                               "Input pointer of the ptr to int.",
                               IRTArg::POTENTIALLY_INDIRECT, getPtr));
    if (!IsPRE && Config.has(PassResult))
      IRTArgs.push_back(IRTArg(
          IntegerType::getInt64Ty(Ctx), "value", "Result of the ptr to int.",
          IRTArg::REPLACABLE | IRTArg::POTENTIALLY_INDIRECT, getValue,
          Config.has(ReplaceResult) ? replaceValue : nullptr));
    addCommonArgs(IConf, Ctx, Config.has(PassId));
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

  enum ConfigKind {
    PassPointer = 0,
    PassPointerKind,
    PassId,
    NumConfig,
  };

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  StringRef getName() const override { return "base_pointer_info"; }

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;
    if (Config.has(PassPointer))
      IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "base_pointer",
                               "The base pointer in question.",
                               IRTArg::REPLACABLE, getValue, setValueNoop));
    if (Config.has(PassPointerKind))
      IRTArgs.push_back(IRTArg(
          IntegerType::getInt32Ty(Ctx), "base_pointer_kind",
          "The base pointer kind (argument, global, instruction, unknown).",
          IRTArg::NONE, getPointerKind));
    addCommonArgs(IConf, Ctx, Config.has(PassId));
    IConf.addChoice(*this);
  }

  static Value *getPointerKind(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB);
  static Value *setValueNoop(Value &V, Value &NewV,
                             InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB) {
    return &NewV;
  }

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx,
                       ConfigTy *UserConfig = nullptr) {
    auto *AIC = IConf.allocate<BasePointerIO>();
    AIC->init(IConf, Ctx, UserConfig);
  }
};

struct LoopValueRangeIO : public InstrumentationOpportunity {
  LoopValueRangeIO()
      : InstrumentationOpportunity(
            InstrumentationLocation(InstrumentationLocation::SPECIAL_VALUE)) {}
  virtual ~LoopValueRangeIO() {};

  enum ConfigKind {
    PassId,
    NumConfig,
  };

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  StringRef getName() const override { return "loop_value_range"; }

  virtual Type *getValueType(LLVMContext &Ctx) const {
    return IntegerType::getInt64Ty(Ctx);
  }

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;
    IRTArgs.push_back(IRTArg(getValueType(IIRB.Ctx), "initial_loop_val",
                             "The value in the first loop iteration.",
                             IRTArg::NONE, getInitialLoopValue));
    IRTArgs.push_back(IRTArg(getValueType(IIRB.Ctx), "final_loop_val",
                             "The value in the last loop iteration.",
                             IRTArg::NONE, getFinalLoopValue));
    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "max_offset",
                             "The maximal offset inside the loop.",
                             IRTArg::NONE, getMaxOffset));
    addCommonArgs(IConf, IIRB.Ctx, Config.has(PassId));
    IConf.addChoice(*this);
  }

  static Value *getInitialLoopValue(Value &V, Type &Ty,
                                    InstrumentationConfig &IConf,
                                    InstrumentorIRBuilderTy &IIRB);
  static Value *getFinalLoopValue(Value &V, Type &Ty,
                                  InstrumentationConfig &IConf,
                                  InstrumentorIRBuilderTy &IIRB);
  static Value *getMaxOffset(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB,
                       ConfigTy *UserConfig = nullptr) {
    auto *LVRIO = IConf.allocate<LoopValueRangeIO>();
    LVRIO->init(IConf, IIRB, UserConfig);
  }

  virtual Value *instrument(Value *&V, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB,
                            InstrumentationCaches &ICaches) override {
    if (CB && !CB(*V))
      return nullptr;
    auto [IP, Success] = IIRB.computeLoopRangeValues(*V, AdditionalSize);
    if (!Success)
      return nullptr;
    IRBuilderBase::InsertPointGuard IPG(IIRB.IRB);
    IIRB.IRB.SetInsertPoint(IP);
    ensureDbgLoc(IIRB.IRB);
    return InstrumentationOpportunity::instrument(V, IConf, IIRB, ICaches);
  }

  virtual Type *getRetTy(LLVMContext &Ctx) const override {
    return PointerType::getUnqual(Ctx);
  }

  void setAdditionalSize(uint32_t AS) { AdditionalSize = AS; }

  uint32_t AdditionalSize = 0;
};

struct FunctionIO : public InstrumentationOpportunity {
  FunctionIO(bool IsPRE)
      : InstrumentationOpportunity(
            InstrumentationLocation(InstrumentationLocation(
                IsPRE ? InstrumentationLocation::FUNCTION_PRE
                      : InstrumentationLocation::FUNCTION_POST))) {}
  virtual ~FunctionIO() {};

  enum ConfigKind {
    PassAddress = 0,
    PassName,
    PassNumArguments,
    PassArguments,
    ReplaceArguments,
    PassIsMain,
    PassId,
    NumConfig,
  };

  struct ConfigTy final : public BaseConfigTy<ConfigKind> {
    std::function<bool(Argument &)> ArgFilter;

    ConfigTy(bool Enable = true) : BaseConfigTy(Enable) {}
  } Config;

  StringRef getName() const override { return "function"; }

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            ConfigTy *UserConfig = nullptr) {
    using namespace std::placeholders;
    if (UserConfig)
      Config = *UserConfig;

    if (Config.has(PassAddress))
      IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "address",
                               "The function address.", IRTArg::NONE,
                               getFunctionAddress));
    if (Config.has(PassName))
      IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "name",
                               "The function name.", IRTArg::STRING,
                               getFunctionName));
    if (Config.has(PassNumArguments))
      IRTArgs.push_back(IRTArg(
          IntegerType::getInt32Ty(Ctx), "num_arguments",
          "Number of function arguments (without varargs).", IRTArg::NONE,
          std::bind(&FunctionIO::getNumArguments, this, _1, _2, _3, _4)));
    if (Config.has(PassArguments))
      IRTArgs.push_back(
          IRTArg(PointerType::getUnqual(Ctx), "arguments",
                 "Description of the arguments.",
                 Config.has(ReplaceArguments) ? IRTArg::REPLACABLE_CUSTOM
                                              : IRTArg::NONE,
                 std::bind(&FunctionIO::getArguments, this, _1, _2, _3, _4),
                 std::bind(&FunctionIO::setArguments, this, _1, _2, _3, _4)));
    if (Config.has(PassIsMain))
      IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "is_main",
                               "Flag to indicate it is the main function.",
                               IRTArg::NONE, isMainFunction));
    addCommonArgs(IConf, Ctx, Config.has(PassId));
    IConf.addChoice(*this);
  }

  static Value *getFunctionAddress(Value &V, Type &Ty,
                                   InstrumentationConfig &IConf,
                                   InstrumentorIRBuilderTy &IIRB);
  static Value *getFunctionName(Value &V, Type &Ty,
                                InstrumentationConfig &IConf,
                                InstrumentorIRBuilderTy &IIRB);
  Value *getNumArguments(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB);
  Value *getArguments(Value &V, Type &Ty, InstrumentationConfig &IConf,
                      InstrumentorIRBuilderTy &IIRB);
  Value *setArguments(Value &V, Value &NewV, InstrumentationConfig &IConf,
                      InstrumentorIRBuilderTy &IIRB);
  static Value *isMainFunction(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    for (auto IsPRE : {true, false}) {
      auto *AIC = IConf.allocate<FunctionIO>(IsPRE);
      AIC->init(IConf, Ctx);
    }
  }
};

struct ModuleIO : public InstrumentationOpportunity {
  ModuleIO(bool IsPRE)
      : InstrumentationOpportunity(InstrumentationLocation(
            IsPRE ? InstrumentationLocation::MODULE_PRE
                  : InstrumentationLocation::MODULE_POST)) {}
  virtual ~ModuleIO() {};

  enum ConfigKind {
    PassId,
    NumConfig,
  };

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  StringRef getName() const override { return "module"; }

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;

    IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "module_name",
                             "The module/translation unit name.",
                             IRTArg::STRING, getModuleName));
    IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "name",
                             "The target triple.", IRTArg::STRING,
                             getTargetTriple));

    addCommonArgs(IConf, Ctx, Config.has(PassId));
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

  enum ConfigKind {
    PassAddress = 0,
    PassName,
    PassInitialValue,
    PassInitialValueSize,
    PassIsConstant,
    PassIsDefinition,
    PassId,
    ReplaceAddress,
    NumConfig,
  };

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  StringRef getName() const override { return "global"; }

  void init(InstrumentationConfig &IConf, LLVMContext &Ctx,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;
    if (Config.has(PassAddress))
      IRTArgs.push_back(IRTArg(
          PointerType::getUnqual(Ctx), "address", "The address of the global.",
          Config.has(ReplaceAddress) ? IRTArg::REPLACABLE : IRTArg::NONE,
          getAddress, setAddress));
    if (Config.has(PassName))
      IRTArgs.push_back(IRTArg(PointerType::getUnqual(Ctx), "name",
                               "The name of the global.", IRTArg::STRING,
                               getSymbolName));
    if (Config.has(PassInitialValue))
      IRTArgs.push_back(
          IRTArg(IntegerType::getInt64Ty(Ctx), "initial_value",
                 "The initial value of the global.",
                 IRTArg::POTENTIALLY_INDIRECT | IRTArg::INDIRECT_HAS_SIZE,
                 getInitialValue));
    if (Config.has(PassInitialValueSize))
      IRTArgs.push_back(IRTArg(IntegerType::getInt32Ty(Ctx),
                               "initial_value_size",
                               "The size of the initial value of the global.",
                               IRTArg::NONE, getInitialValueSize));
    if (Config.has(PassIsConstant))
      IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "is_constant",
                               "Flag to indicate constant globals.",
                               IRTArg::NONE, isConstant));
    if (Config.has(PassIsDefinition))
      IRTArgs.push_back(IRTArg(IntegerType::getInt8Ty(Ctx), "is_definition",
                               "Flag to indicate global definitions.",
                               IRTArg::NONE, isDefinition));
    addCommonArgs(IConf, Ctx, Config.has(PassId));
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
  static Value *isDefinition(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf, LLVMContext &Ctx) {
    auto *AIC = IConf.allocate<GlobalIO>();
    AIC->init(IConf, Ctx);
  }
};

} // namespace instrumentor

class InstrumentorPass : public PassInfoMixin<InstrumentorPass> {
  using InstrumentationConfig = instrumentor::InstrumentationConfig;
  using InstrumentorIRBuilderTy = instrumentor::InstrumentorIRBuilderTy;
  InstrumentationConfig *UserIConf;
  InstrumentorIRBuilderTy *UserIIRB;

public:
  InstrumentorPass(InstrumentationConfig *IC = nullptr,
                   InstrumentorIRBuilderTy *IIRB = nullptr)
      : UserIConf(IC), UserIIRB(IIRB) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_INSTRUMENTOR_H
