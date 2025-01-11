//===-- Instrumentor.cpp - Highly configurable instrumentation pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/Instrumentor.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/ConstantFolder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>

using namespace llvm;

#define DEBUG_TYPE "instrumentor"

cl::opt<std::string> WriteJSONConfig(
    "instrumentor-write-config-file",
    cl::desc(
        "Write the instrumentor configuration into the specified JSON file"),
    cl::init(""));
cl::opt<std::string> ReadJSONConfig(
    "instrumentor-read-config-file",
    cl::desc(
        "Read the instrumentor configuration from the specified JSON file"),
    cl::init(""));

namespace {

struct RTArgument {

  RTArgument(Type *Ty, StringRef Name, InstrumentorKindTy Kind = PLAIN)
      : Ty(Ty), Name(Name), Kind(Kind) {}
  RTArgument(Value *V, StringRef Name, InstrumentorKindTy Kind = PLAIN)
      : V(V), Ty(V->getType()), Name(Name), Kind(Kind) {}

  Value *V = nullptr;
  Type *Ty = nullptr;
  std::string Name;
  InstrumentorKindTy Kind;
};

template <typename... Targs>
void dumpObject(json::OStream &J, Targs... Fargs) {}

void writeInstrumentorConfig(InstrumentorConfig &IC) {
  if (WriteJSONConfig.empty())
    return;

  std::error_code EC;
  raw_fd_stream OS(WriteJSONConfig, EC);
  if (EC) {
    errs() << "WARNING: Failed to open instrumentor configuration file for "
              "writing: "
           << EC.message() << "\n";
    return;
  }

  json::OStream J(OS, 2);
  J.objectBegin();

#define SECTION_START(SECTION, POSITION)                                       \
  J.attributeBegin(#SECTION);                                                  \
  J.objectBegin();                                                             \
  J.attribute("Enabled", IC.SECTION.Enabled);
#define CVALUE_INTERNAL(SECTION, TYPE, NAME, DEFAULT_VALUE)
#define CVALUE(SECTION, TYPE, NAME, DEFAULT_VALUE)                             \
  J.attribute(#NAME, IC.SECTION.NAME);
#define RTVALUE(SECTION, NAME, DEFAULT_VALUE, VALUE_TYPE_STR, PROPERTIES)      \
  J.attribute(#NAME, IC.SECTION.NAME.Enabled);
#define SECTION_END(SECTION)                                                   \
  J.objectEnd();                                                               \
  J.attributeEnd();

#include "llvm/Transforms/Instrumentation/InstrumentorConfig.def"

  J.objectEnd();
}

bool readInstrumentorConfigFromJSON(InstrumentorConfig &IC) {
  if (ReadJSONConfig.empty())
    return true;

  std::error_code EC;
  auto BufferOrErr = MemoryBuffer::getFileOrSTDIN(ReadJSONConfig);
  if (std::error_code EC = BufferOrErr.getError()) {
    errs() << "WARNING: Failed to open instrumentor configuration file for "
              "reading: "
           << EC.message() << "\n";
    return false;
  }
  auto Buffer = std::move(BufferOrErr.get());
  json::Path::Root NullRoot;
  auto Parsed = json::parse(Buffer->getBuffer());
  if (!Parsed) {
    errs() << "WARNING: Failed to parse the instrumentor configuration file: "
           << Parsed.takeError() << "\n";
    return false;
  }
  auto *Config = Parsed->getAsObject();
  if (!Config) {
    errs() << "WARNING: Failed to parse the instrumentor configuration file: "
              "Expected "
              "an object '{ ... }'\n";
    return false;
  }

  auto End = Config->end(), It = Config->begin();

  bool Read;
  json::Object *Obj = nullptr;

#define CVALUE(SECTION, TYPE, NAME, DEFAULT_VALUE)                             \
  Read = false;                                                                \
  if (Obj)                                                                     \
    if (auto *Val = Obj->get(#NAME))                                           \
      Read = json::fromJSON(*Val, IC.SECTION.NAME, NullRoot);                  \
  if (!Read)                                                                   \
    errs() << "WARNING: Failed to read " #SECTION "." #NAME " as " #TYPE       \
           << "\n";

#define RTVALUE(SECTION, NAME, DEFAULT_VALUE, VALUE_TYPE_STR, PROPERTIES)      \
  if (Obj) {                                                                   \
    if (auto *Val = Obj->get(#NAME))                                           \
      if (!json::fromJSON(*Val, IC.SECTION.NAME.Enabled, NullRoot))            \
        errs() << "WARNING: Failed to read " #SECTION "." #NAME " as bool\n";  \
  }

#define SECTION_START(SECTION, POSITION)                                       \
  Obj = nullptr;                                                               \
  It = Config->find(#SECTION);                                                 \
  if (It != End) {                                                             \
    Obj = It->second.getAsObject();                                            \
    if (auto *Val = Obj->get("Enabled"))                                       \
      if (!json::fromJSON(*Val, IC.SECTION.Enabled, NullRoot))                 \
        errs() << "WARNING: Failed to read " #SECTION ".Enabled as bool\n";    \
  }

#define CVALUE_INTERNAL(SECTION, TYPE, NAME, DEFAULT_VALUE)
#define SECTION_END(SECTION)

#include "llvm/Transforms/Instrumentation/InstrumentorConfig.def"

  return true;
}

raw_ostream &printAsCType(raw_ostream &OS, Type *Ty,
                          InstrumentorKindTy Kind = InstrumentorKindTy::PLAIN,
                          bool IsIndirect = false) {
  // TODO: filter ORIGINAL_VALUE and such
  switch (Kind) {
  case InstrumentorKindTy::STRING:
    return OS << "char* ";
  case InstrumentorKindTy::INT32_PTR:
    return OS << "int* ";
  case InstrumentorKindTy::PTR_PTR:
    return OS << "void** ";
  default:
    break;
  };
  if (Ty->isPointerTy())
    OS << "void*";
  else if (Ty->isIntegerTy())
    OS << "int" << Ty->getIntegerBitWidth() << "_t";
  else
    OS << *Ty;
  if (IsIndirect)
    OS << "*";
  return OS << " ";
}

bool printAsPrintfFormat(raw_ostream &OS, InstrumentorConfig::ConfigValue &CV,
                         LLVMContext &Ctx, bool IsIndirect) {
  // TODO: filter ORIGINAL_VALUE and such
  switch (CV.getKind()) {
  default:
    break;
  case InstrumentorKindTy::STRING:
    return OS << "%s", true;
  case InstrumentorKindTy::TYPE_ID:
    return OS << "%i (%s)", true;
  case InstrumentorKindTy::INITIALIZER_KIND:
    return OS << "%i (%s)", true;
  case InstrumentorKindTy::BOOLEAN:
    return OS << "%s", true;
  case InstrumentorKindTy::INT32_PTR:
    return OS << "%p (%i)", true;
  case InstrumentorKindTy::PTR_PTR:
    return OS << "%p (%p)", true;
  };

  Type *Ty = CV.getType(Ctx);
  if (IsIndirect)
    OS << "%p -> ";
  if (Ty->isPointerTy())
    return OS << "%p", true;
  if (Ty->isIntegerTy()) {
    if (Ty->getIntegerBitWidth() <= 32)
      return OS << "%i", true;
    if (Ty->getIntegerBitWidth() <= 64)
      return OS << "%li", true;
  }
  if (Ty->isFloatTy())
    return OS << "%f", true;
  if (Ty->isDoubleTy())
    return OS << "%lf", true;
  return false;
}

void printInitializerInfo(raw_ostream &OS, bool Definition) {
  OS << "char *getInitializerKindStr(int32_t InitializerKind)";
  if (!Definition) {
    OS << ";\n\n";
    return;
  }
  OS << " {\n";
  OS << "  switch (InitializerKind) {\n";
#define INITKIND(ID, STR)                                                      \
  OS << "    case " << ID << ": return \"" << STR << "\";\n";

  INITKIND(-1, "unknown")
  INITKIND(0, "zeroinit")
  INITKIND(1, "undefval")
  INITKIND(2, "complex")
  OS << "  default: return \"unknown\";";
  OS << "  }\n";
  OS << "}\n\n";
}

void printTypeIDSwitch(raw_ostream &OS, bool Definition) {
  OS << "char *getTypeIDStr(int32_t TypeID)";
  if (!Definition) {
    OS << ";\n\n";
    return;
  }
  OS << " {\n";
  OS << "  switch (TypeID) {\n";
#define TYPEID(ID, STR)                                                        \
  OS << "    case " << ID << ": return \"" << STR << "\";\n";

  TYPEID(Type::HalfTyID, "16-bit floating point type")
  TYPEID(Type::BFloatTyID, "16-bit floating point type (7-bit significand)")
  TYPEID(Type::FloatTyID, "32-bit floating point type")
  TYPEID(Type::DoubleTyID, "64-bit floating point type")
  TYPEID(Type::X86_FP80TyID, "80-bit floating point type (X87)")
  TYPEID(Type::FP128TyID, "128-bit floating point type (112-bit significand)")
  TYPEID(Type::PPC_FP128TyID,
         "128-bit floating point type (two 64-bits, PowerPC)")
  TYPEID(Type::VoidTyID, "type with no size")
  TYPEID(Type::LabelTyID, "Labels")
  TYPEID(Type::MetadataTyID, "Metadata")
  TYPEID(Type::X86_AMXTyID, "AMX vectors (8192 bits, X86 specific)")
  TYPEID(Type::TokenTyID, "Tokens")
  TYPEID(Type::IntegerTyID, "Arbitrary bit width integers")
  TYPEID(Type::FunctionTyID, "Functions")
  TYPEID(Type::PointerTyID, "Pointers")
  TYPEID(Type::StructTyID, "Structures")
  TYPEID(Type::ArrayTyID, "Arrays")
  TYPEID(Type::FixedVectorTyID, "Fixed width SIMD vector type")
  TYPEID(Type::ScalableVectorTyID, "Scalable SIMD vector type")
  TYPEID(Type::TypedPointerTyID, "Typed pointer used by some GPU targets")
  TYPEID(Type::TargetExtTyID, "Target extension type")
  OS << "  default: return \"unknown\";";
  OS << "  }\n";
  OS << "}\n\n";
}

template <typename IRBTy>
Value *tryToCast(IRBTy &IRB, Value *V, Type *Ty, const DataLayout &DL,
                 bool CheckOnly = false) {
  if (!V)
    return Constant::getAllOnesValue(Ty);
  auto *VTy = V->getType();
  if (VTy == Ty)
    return V;
  if (CastInst::isBitOrNoopPointerCastable(VTy, Ty, DL))
    return CheckOnly ? V : IRB.CreateBitOrPointerCast(V, Ty);
  if (VTy->isPointerTy() && Ty->isPointerTy())
    return CheckOnly ? V : IRB.CreatePointerBitCastOrAddrSpaceCast(V, Ty);
  if (VTy->isPointerTy() && Ty->isIntegerTy())
    return CheckOnly ? V : IRB.CreatePtrToInt(V, Ty);
  if (VTy->isIntegerTy() && Ty->isPointerTy())
    return CheckOnly ? V : IRB.CreateIntToPtr(V, Ty);
  if (VTy->isIntegerTy() && Ty->isIntegerTy())
    return CheckOnly ? V : IRB.CreateIntCast(V, Ty, /* isSigned */ false);
  if (VTy->isFloatingPointTy() && Ty->isIntOrPtrTy()) {
    switch (DL.getTypeSizeInBits(VTy)) {
    case 64:
      V = CheckOnly ? IRB.getInt64(0) : IRB.CreateBitCast(V, IRB.getInt64Ty());
      break;
    case 32:
      V = CheckOnly ? IRB.getInt32(0) : IRB.CreateBitCast(V, IRB.getInt32Ty());
      break;
    case 16:
      V = CheckOnly ? IRB.getInt16(0) : IRB.CreateBitCast(V, IRB.getInt16Ty());
      break;
    case 8:
      V = CheckOnly ? IRB.getInt8(0) : IRB.CreateBitCast(V, IRB.getInt8Ty());
      break;
    default:
      return nullptr;
    }
    return tryToCast(IRB, V, Ty, DL);
  }
  return nullptr;
}

class InstrumentorImpl final {
public:
  InstrumentorImpl(InstrumentorConfig &IC, Module &M,
                   ModuleAnalysisManager &MAM)
      : IC(IC), M(M), MAM(MAM),
        FAM(MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager()),
        GetTLI([this](Function &F) -> TargetLibraryInfo & {
          return FAM.getResult<TargetLibraryAnalysis>(F);
        }),
        Ctx(M.getContext()), IRB(Ctx, ConstantFolder(),
                                 IRBuilderCallbackInserter([&](Instruction *I) {
                                   NewInsts[I] = Epoche;
                                 })) {}

  void printRuntimeSignatures() {
    auto *DeclOut = IC.Base.PrintRuntimeSignatures ? &outs() : nullptr;
    auto *StubRTOut = getStubRuntimeOut();
    if (!DeclOut && !StubRTOut)
      return;

    SmallVector<InstrumentorConfig::ConfigValue *> ConfigValues;
    InstrumentorConfig::Position Position;
    int HasPotentiallyIndirect = 0;
    StringRef Name;

#define CVALUE(SECTION, TYPE, NAME, DEFAULT_VALUE)
#define CVALUE_INTERNAL(SECTION, TYPE, NAME, DEFAULT_VALUE)
#define RTVALUE(SECTION, NAME, DEFAULT_VALUE, VALUE_TYPE_STR, PROPERTIES)      \
  ConfigValues.push_back(&IC.SECTION.NAME);                                    \
  HasPotentiallyIndirect += bool(IC.SECTION.NAME.getKind() &                   \
                                 InstrumentorKindTy::POTENTIALLY_INDIRECT);

#define SECTION_START(SECTION, POSITION)                                       \
  Name = IC.SECTION.SectionName;                                               \
  HasPotentiallyIndirect = 0;                                                  \
  Position = POSITION;

#define SECTION_END(SECTION)                                                   \
  for (auto &Pos : {InstrumentorConfig::PRE, InstrumentorConfig::POST})        \
    if (Position & Pos) {                                                      \
      if (HasPotentiallyIndirect < 2)                                          \
        printStubRTDefinitions(DeclOut, StubRTOut, Name, ConfigValues, Pos,    \
                               /*Indirect=*/false);                            \
      if (HasPotentiallyIndirect)                                              \
        printStubRTDefinitions(DeclOut, StubRTOut, Name, ConfigValues, Pos,    \
                               /*Indirect=*/true);                             \
    }                                                                          \
  ConfigValues.clear();

#include "llvm/Transforms/Instrumentation/InstrumentorConfig.def"
  }

  ~InstrumentorImpl() {
    if (StubRuntimeOut) {
      printTypeIDSwitch(*StubRuntimeOut, /*Definition=*/true);
      printInitializerInfo(*StubRuntimeOut, /*Definition=*/true);
      delete StubRuntimeOut;
    }
  }

  /// Instrument the module, public entry point.
  bool instrument();

private:
  bool shouldInstrumentFunction(Function *Fn);
  bool shouldInstrumentGlobalVariable(GlobalVariable *GV);
  bool hasAnnotation(Instruction *I, StringRef Annotation);

  bool instrumentFunction(Function &Fn);
  bool instrumentAlloca(InstrumentorConfig::allocaObj &Section, AllocaInst &I);
  bool instrumentCall(CallBase &I);
  bool instrumentCallArgs(CallInst &I);
  bool instrumentAllocationCall(CallBase &I, const AllocationCallInfo &ACI);
  bool instrumentMemoryIntrinsic(IntrinsicInst &I);
  bool instrumentGeneralIntrinsic(IntrinsicInst &I);
  bool instrumentLoad(LoadInst &I, bool After);
  bool instrumentStore(StoreInst &I);
  bool instrumentUnreachable(UnreachableInst &I);
  bool instrumentMainFunction(Function &MainFn);
  bool instrumentModule(bool After);

  DenseMap<Value *, CallInst *> BasePtrMap;
  bool instrumentBasePointer(Value &ArgOrInst);
  bool removeUnusedBasePointers();
  Value *findBasePointer(Value *V);

  template <typename MemoryInstTy> bool analyzeAccess(MemoryInstTy &I);

  SmallVector<GlobalVariable *> Globals;
  bool prepareGlobalVariables();
  bool instrumentGlobalVariables();

  void addCtorOrDtor(bool Ctor);

  /// Mapping to remember temporary allocas for reuse.
  DenseMap<std::pair<Function *, unsigned>, AllocaInst *> AllocaMap;

  /// Return a temporary alloca to communicate (large) values with the runtime.
  AllocaInst *getAlloca(Function *Fn, Type *Ty) {
    AllocaInst *&AI = AllocaMap[{Fn, DL.getTypeAllocSize(Ty)}];
    if (!AI)
      AI = new AllocaInst(Ty, DL.getAllocaAddrSpace(), "",
                          Fn->getEntryBlock().getFirstNonPHIOrDbgOrAlloca());
    return AI;
  }

  /// Mapping to remember global strings passed to the runtime.
  DenseMap<StringRef, Value *> GlobalStringsMap;
  Value *getGlobalString(StringRef S) {
    Value *&V = GlobalStringsMap[S];
    if (!V) {
      Twine Name = Twine(IC.Base.RuntimeName) + "str";
      V = IRB.CreateGlobalString(S, Name, DL.getDefaultGlobalsAddressSpace(),
                                 &M);
    }
    return V;
  }

  template <typename Ty> Constant *getCI(Type *IT, Ty Val) {
    return ConstantInt::get(IT, Val);
  }

  std::string getRTName(StringRef Prefix, StringRef Name,
                        StringRef Suffix = "") {
    return (IC.Base.RuntimeName + Prefix + Name + Suffix).str();
  }
  std::string getRTName(InstrumentorConfig::Position Position, StringRef Name,
                        StringRef Suffix = "") {
    switch (Position) {
    case InstrumentorConfig::NONE:
      return getRTName("", Name, Suffix);
    case InstrumentorConfig::PRE:
      return getRTName("pre_", Name, Suffix);
    case InstrumentorConfig::POST:
      return getRTName("post_", Name, Suffix);
    default:
      llvm_unreachable("Invalid position");
    }
  }

  raw_fd_ostream *StubRuntimeOut = nullptr;

  raw_fd_ostream *getStubRuntimeOut() {
    if (!IC.Base.StubRuntimePath.empty()) {
      std::error_code EC;
      StubRuntimeOut = new raw_fd_ostream(IC.Base.StubRuntimePath, EC);
      if (EC) {
        errs() << "WARNING: Failed to open instrumentor stub runtime "
                  "file for "
                  "writing: "
               << EC.message() << "\n";
        delete StubRuntimeOut;
        StubRuntimeOut = nullptr;
      } else {
        *StubRuntimeOut << "// LLVM Instrumentor stub runtime\n\n";
        *StubRuntimeOut << "#include <stdint.h>\n";
        *StubRuntimeOut << "#include <stdio.h>\n\n";
        printTypeIDSwitch(*StubRuntimeOut, /*Definition=*/false);
        printInitializerInfo(*StubRuntimeOut, /*Definition=*/false);
      }
    }
    return StubRuntimeOut;
  }

  /// Map to remember instrumentation functions for a specific opcode and
  /// pre/post position.
  StringMap<FunctionCallee> InstrumentorCallees;

  CallInst *getCall(StringRef Name, SmallVectorImpl<RTArgument> &RTArgs,
                    bool After) {

    bool UsesIndirection = false, DirectReturn = false;
    Type *RetTy = VoidTy;
    SmallVector<Type *> RTArgTypes;
    SmallVector<Value *> CallArgs;
    for (auto &RTA : RTArgs) {
      bool PreOnly = (RTA.Kind & InstrumentorKindTy::PRE_ONLY);
      bool PostOnly = (RTA.Kind & InstrumentorKindTy::POST_ONLY);
      if (((PostOnly && !After) || (PreOnly && After)))
        continue;
      RTArgTypes.push_back(RTA.Ty);
      CallArgs.push_back(RTA.V);
      if (RTA.Kind & InstrumentorKindTy::POTENTIALLY_INDIRECT)
        UsesIndirection |= RTA.Ty->isPointerTy();
      if ((RTA.Kind & InstrumentorKindTy::REPLACABLE_PRE) && !After) {
        if (RetTy != VoidTy)
          UsesIndirection = true;
        if (!(RTA.Kind & InstrumentorKindTy::POTENTIALLY_INDIRECT)) {
          assert(!DirectReturn);
          DirectReturn = true;
        }
        RetTy = RTA.Ty;
      } else if ((RTA.Kind & InstrumentorKindTy::REPLACABLE_POST) && After) {
        if (RetTy != VoidTy)
          UsesIndirection = true;
        if (!(RTA.Kind & InstrumentorKindTy::POTENTIALLY_INDIRECT)) {
          assert(!DirectReturn);
          DirectReturn = true;
        }
        RetTy = RTA.Ty;
      }
    }
    if (UsesIndirection && !DirectReturn)
      RetTy = VoidTy;

    std::string CompleteName = getRTName(After ? "post_" : "pre_", Name,
                                         UsesIndirection ? "_ind" : "");
    FunctionCallee &FC = InstrumentorCallees[CompleteName];

    if (!FC.getFunctionType())
      FC = M.getOrInsertFunction(
          CompleteName,
          FunctionType::get(RetTy, RTArgTypes, /*IsVarArgs*/ false));

    return IRB.CreateCall(FC, CallArgs);
  }

  bool shouldSkipCV(InstrumentorConfig::ConfigValue &CV, bool After) {
    bool PreOnly = (CV.getKind() & InstrumentorKindTy::PRE_ONLY);
    bool PostOnly = (CV.getKind() & InstrumentorKindTy::POST_ONLY);
    return !CV || ((PostOnly && !After) || (PreOnly && After));
  }

  void printStubRTDefinitions(
      raw_ostream *SignatureOut, raw_ostream *StubRTOut, StringRef Name,
      SmallVectorImpl<InstrumentorConfig::ConfigValue *> &ConfigValues,
      InstrumentorConfig::Position Position, bool Indirect) {

    [[maybe_unused]] bool DirectReturn = false;
    StringRef ReturnedVariable;
    Type *RetTy = VoidTy;
    for (auto *CV : ConfigValues) {
      if (shouldSkipCV(*CV, (Position & InstrumentorConfig::POST)))
        continue;
      bool ReplaceablePre =
          (CV->getKind() & InstrumentorKindTy::REPLACABLE_PRE);
      bool ReplaceablePost =
          (CV->getKind() & InstrumentorKindTy::REPLACABLE_POST);
      if (((!ReplaceablePre && (Position & InstrumentorConfig::PRE)) ||
           (!ReplaceablePost && (Position & InstrumentorConfig::POST))))
        continue;
      if ((ReplaceablePre && (Position & InstrumentorConfig::PRE)) ||
          (ReplaceablePost && (Position & InstrumentorConfig::POST))) {
        bool CanBeIndirect =
            (Indirect &&
             (CV->getKind() & InstrumentorKindTy::POTENTIALLY_INDIRECT));
        assert(!DirectReturn || CanBeIndirect);
        if (!CanBeIndirect) {
          RetTy = CV->getType(Ctx);
          ReturnedVariable = CV->getName();
        }
        DirectReturn |= !CanBeIndirect;
      }
    }

    bool First = true;
    std::string Str;
    raw_string_ostream StrOut(Str);
    printAsCType(StrOut, RetTy);
    auto CompleteName = getRTName(Position, Name, Indirect ? "_ind" : "");
    StrOut << CompleteName << "(";
    for (auto *CV : ConfigValues) {
      if (shouldSkipCV(*CV, (Position & InstrumentorConfig::POST)))
        continue;
      if (!First)
        StrOut << ", ";
      First = false;
      bool IsIndirect = Indirect && (CV->getKind() &
                                     InstrumentorKindTy::POTENTIALLY_INDIRECT);
      printAsCType(StrOut, CV->getType(Ctx), CV->getKind(), IsIndirect)
          << CV->getName();
      if (Indirect &&
          (CV->getKind() & InstrumentorKindTy::POTENTIALLY_INDIRECT))
        StrOut << "Ind";
    }
    StrOut << ")";
    if (SignatureOut)
      *SignatureOut << Str << ";\n";
    if (!StubRTOut)
      return;
    (*StubRTOut) << Str << " {\n";
    Str.clear();
    (*StubRTOut) << "  printf(\"" << CompleteName << " - ";
    for (auto *CV : ConfigValues) {
      if (shouldSkipCV(*CV, (Position & InstrumentorConfig::POST)))
        continue;
      bool IsIndirect = Indirect && (CV->getKind() &
                                     InstrumentorKindTy::POTENTIALLY_INDIRECT);
      if (IsIndirect)
        (*StubRTOut) << CV->getName() << "Ind: ";
      else
        (*StubRTOut) << CV->getName() << ": ";
      if (printAsPrintfFormat(*StubRTOut, *CV, Ctx, IsIndirect)) {
        if (!Str.empty())
          StrOut << ", ";
        // TODO: filter ORIGINAL_VALUE and such
        switch (CV->getKind()) {
        case InstrumentorKindTy::TYPE_ID:
          StrOut << CV->getName() << ", getTypeIDStr(" << CV->getName() << ")";
          break;
        case InstrumentorKindTy::INITIALIZER_KIND:
          StrOut << CV->getName() << ", getInitializerKindStr(" << CV->getName()
                 << ")";
          break;
        case InstrumentorKindTy::BOOLEAN:
          StrOut << CV->getName() << " ? \"true\" : \"false\"";
          break;
        case InstrumentorKindTy::INT32_PTR:
        case InstrumentorKindTy::PTR_PTR:
          StrOut << CV->getName() << ", *" << CV->getName();
          break;
        default:
          if (IsIndirect)
            StrOut << "(void*)";
          StrOut << CV->getName();
          if (IsIndirect)
            StrOut << "Ind, *" << CV->getName() << "Ind";
        }
      } else {
        (*StubRTOut) << "unknown";
      }
      (*StubRTOut) << " - ";
    }
    if (ConfigValues.empty())
      (*StubRTOut) << "\\n\");\n";
    else
      (*StubRTOut) << "\\n\", " << Str << ");\n";
    if (!ReturnedVariable.empty())
      (*StubRTOut) << "  return " << ReturnedVariable << ";\n";
    (*StubRTOut) << "}\n\n";
  }

  void addVal(SmallVectorImpl<RTArgument> &RTArgs,
              InstrumentorConfig::ConfigValue &Obj, Value *V, bool After) {
    if (shouldSkipCV(Obj, After))
      return;
    Type *Ty = Obj.getType(Ctx);
    V = tryToCast(IRB, V, Ty, DL);
    assert(V->getType() == Ty);
    RTArgs.emplace_back(V, Obj.getName(), Obj.getKind());
  }

  AllocaInst *addIndVal(SmallVectorImpl<RTArgument> &RTArgs,
                        InstrumentorConfig::ConfigValue &Obj, Value *V,
                        Function *F, bool After,
                        bool ForceIndirection = false) {
    if (shouldSkipCV(Obj, After))
      return nullptr;
    if (!ForceIndirection &&
        tryToCast(IRB, V, Obj.getType(Ctx), DL, /*CheckOnly=*/true)) {
      addVal(RTArgs, Obj, V, After);
      return nullptr;
    }
    auto *IndirectionAI = getAlloca(F, V->getType());
    IRB.CreateStore(V, IndirectionAI);
    RTArgs.emplace_back(IndirectionAI, std::string(Obj.getName()) + "Ind",
                        Obj.getKind());
    return IndirectionAI;
  }

  void addValCB(SmallVectorImpl<RTArgument> &RTArgs,
                InstrumentorConfig::ConfigValue &Obj,
                llvm::function_ref<Value *(Type *)> ValFn, bool After) {
    if (shouldSkipCV(Obj, After))
      return;
    Type *Ty = Obj.getType(Ctx);
    Value *V = ValFn(Ty);
    assert(V->getType() == Ty);
    RTArgs.emplace_back(tryToCast(IRB, V, Ty, DL), Obj.getName(),
                        Obj.getKind());
  }

  void addCI(SmallVectorImpl<RTArgument> &RTArgs,
             InstrumentorConfig::ConfigValue &Obj, uint64_t Value, bool After) {
    if (shouldSkipCV(Obj, After))
      return;
    RTArgs.emplace_back(getCI(Obj.getType(Ctx), Value), Obj.getName(),
                        Obj.getKind());
  }

  /// Each instrumentation, i.a., of an instruction, is happening in a dedicated
  /// epoche. The epoche allows to determine if instrumentation instructions
  /// were already around, due to prior instrumentations, or have been
  /// introduced to support the current instrumentation, i.a., compute
  /// information about the current instruction.
  unsigned Epoche = 0;

  /// A mapping from instrumentation instructions to the epoche they have been
  /// created.
  DenseMap<Instruction *, unsigned> NewInsts;

  /// The instrumentor configuration.
  InstrumentorConfig &IC;

  /// The underlying module.
  Module &M;

  ModuleAnalysisManager &MAM;
  FunctionAnalysisManager &FAM;

  std::function<TargetLibraryInfo &(Function &F)> GetTLI;

  /// The underying LLVM context.
  LLVMContext &Ctx;

  /// A special IR builder that keeps track of the inserted instructions.
  IRBuilder<ConstantFolder, IRBuilderCallbackInserter> IRB;

  /// The module's ctor and dtor functions.
  Function *CtorFn = nullptr;
  Function *DtorFn = nullptr;

  static constexpr StringRef SkipAnnotation = "__instrumentor_skip";
  static constexpr StringRef COAnnotation = "__instrumentor_constoffset";

  /// Commonly used values for IR inspection and creation.
  ///{

  const DataLayout &DL = M.getDataLayout();

  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *IntptrTy = M.getDataLayout().getIntPtrType(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);
  IntegerType *Int8Ty = Type::getInt8Ty(Ctx);
  IntegerType *Int32Ty = Type::getInt32Ty(Ctx);
  IntegerType *Int64Ty = Type::getInt64Ty(Ctx);
  Constant *NullPtrVal = Constant::getNullValue(PtrTy);
  ///}
};

} // end anonymous namespace

bool InstrumentorImpl::shouldInstrumentFunction(Function *Fn) {
  if (!Fn || Fn->isDeclaration())
    return false;
  return !Fn->getName().starts_with(IC.Base.RuntimeName);
}

bool InstrumentorImpl::shouldInstrumentGlobalVariable(GlobalVariable *GV) {
  if (!GV || GV->hasGlobalUnnamedAddr())
    return false;
  if (GV->getName().starts_with("llvm."))
    return false;
  return !GV->getName().starts_with(IC.Base.RuntimeName);
}

bool InstrumentorImpl::hasAnnotation(Instruction *I, StringRef Annotation) {
  if (!I || !I->hasMetadata(LLVMContext::MD_annotation))
    return false;

  return any_of(I->getMetadata(LLVMContext::MD_annotation)->operands(),
                [&](const MDOperand &Op) {
                  StringRef AnnotationStr =
                      isa<MDString>(Op.get())
                          ? cast<MDString>(Op.get())->getString()
                          : cast<MDString>(
                                cast<MDTuple>(Op.get())->getOperand(0).get())
                                ->getString();
                  return (AnnotationStr == Annotation);
                });
}

bool InstrumentorImpl::instrumentAlloca(InstrumentorConfig::allocaObj &Section,
                                        AllocaInst &I) {
  if (!Section.Enabled)
    return false;
  if (Section.CB && !Section.CB(I))
    return false;

  Instruction *IP = I.getNextNonDebugInstruction();
  while (isa<AllocaInst>(IP))
    IP = IP->getNextNonDebugInstruction();
  IRB.SetInsertPoint(IP);

  auto CalculateAllocaSize = [&](Type *Ty) {
    Value *SizeValue = nullptr;
    TypeSize TypeSize = DL.getTypeAllocSize(I.getAllocatedType());
    if (TypeSize.isFixed()) {
      SizeValue = getCI(Ty, TypeSize.getFixedValue());
    } else {
      SizeValue = IRB.CreateSub(
          IRB.CreatePtrToInt(
              IRB.CreateGEP(I.getAllocatedType(), &I, {getCI(Int32Ty, 1)}), Ty),
          IRB.CreatePtrToInt(&I, Ty));
    }
    if (I.isArrayAllocation())
      SizeValue = IRB.CreateMul(SizeValue,
                                IRB.CreateZExtOrBitCast(I.getArraySize(), Ty));
    return SizeValue;
  };

  SmallVector<RTArgument> RTArgs;
  addVal(RTArgs, Section.Value, &I, /*After=*/true);
  addValCB(RTArgs, Section.AllocationSize, CalculateAllocaSize,
           /*After=*/true);
  addCI(RTArgs, Section.Alignment, I.getAlign().value(), /*After=*/true);

  auto *CI = getCall(Section.SectionName, RTArgs, /*After=*/true);
  if (Section.ReplaceValue) {
    IRB.SetInsertPoint(CI->getNextNonDebugInstruction());
    I.replaceUsesWithIf(tryToCast(IRB, CI, I.getType(), DL), [&](Use &U) {
      if (NewInsts.lookup(cast<Instruction>(U.getUser())) == Epoche)
        return false;
      if (auto *LI = dyn_cast<LoadInst>(U.getUser()))
        return !hasAnnotation(LI, COAnnotation);
      if (auto *SI = dyn_cast<StoreInst>(U.getUser()))
        return SI->getPointerOperand() == &I &&
               !hasAnnotation(cast<Instruction>(U.getUser()), COAnnotation);
      return true;
    });
  }

  return true;
}

bool InstrumentorImpl::instrumentCall(CallBase &I) {
  if (!IC.allocation_call.Enabled && !IC.memory_intrinsic.Enabled &&
      !IC.intrinsic.Enabled)
    return false;
  auto &TLI = GetTLI(*I.getFunction());
  auto ACI = getAllocationCallInfo(&I, &TLI);

  if (ACI && IC.allocation_call.Enabled)
    return instrumentAllocationCall(I, *ACI);

  if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::memcpy:
    case Intrinsic::memcpy_element_unordered_atomic:
    case Intrinsic::memcpy_inline:
    case Intrinsic::memmove:
    case Intrinsic::memmove_element_unordered_atomic:
    case Intrinsic::memset:
    case Intrinsic::memset_element_unordered_atomic:
    case Intrinsic::memset_inline:
      if (IC.memory_intrinsic.Enabled)
        return instrumentMemoryIntrinsic(*II);
      break;
    case Intrinsic::trap:
    case Intrinsic::debugtrap:
    case Intrinsic::ubsantrap:
        return instrumentGeneralIntrinsic(*II);
      break;
    default:
      break;
    }
  }

  return false;
}

bool InstrumentorImpl::instrumentCallArgs(CallInst &I) {
  Function *CalledFn = I.getCalledFunction();
  // TODO: This should be more generic
  if (shouldInstrumentFunction(CalledFn))
    return false;

  SmallVector<Use *> InstrUses;

  for (size_t A = 0; A < I.arg_size(); ++A) {
    Use &U = I.getArgOperandUse(A);
    Value *Arg = U.get();
    // TODO: This should be more generic. For now, we are only
    // considering pointer arguments
    if (Arg->getType() != PtrTy || Arg == NullPtrVal)
      continue;
    if (auto *LI = dyn_cast<LoadInst>(Arg))
      if (hasAnnotation(LI, SkipAnnotation))
        continue;
    APInt Offset(64, 0);
    Value *BasePtr = Arg->stripAndAccumulateConstantOffsets(DL, Offset, true);
    if (auto *GV = dyn_cast<GlobalVariable>(BasePtr))
      if (!shouldInstrumentGlobalVariable(GV))
        continue;

    InstrUses.push_back(&U);
  }

  if (InstrUses.empty())
    return false;

  IRB.SetInsertPoint(&I);

  for (Use *U : InstrUses) {
    SmallVector<RTArgument> RTArgs;
    auto *IndirectionAI = addIndVal(RTArgs, IC.call_arg.Value, U->get(),
                                    I.getFunction(), /*After=*/false);
    addCI(RTArgs, IC.call_arg.ValueTypeId, U->get()->getType()->getTypeID(),
          /*After=*/false);

    auto *CI = getCall(IC.call_arg.SectionName, RTArgs, /*After=*/false);
    if (IC.call_arg.ReplaceValue) {
      Value *NewV = IndirectionAI
                        ? IRB.CreateLoad(U->get()->getType(), IndirectionAI)
                        : tryToCast(IRB, CI, U->get()->getType(), DL);
      U->set(NewV);
    }
  }

  return true;
}

bool InstrumentorImpl::instrumentAllocationCall(CallBase &I,
                                                const AllocationCallInfo &ACI) {
  if (IC.allocation_call.CB && !IC.allocation_call.CB(I))
    return false;

  Instruction *IP = I.getNextNonDebugInstruction();
  IRB.SetInsertPoint(IP);

  SmallVector<RTArgument> RTArgs;
  addVal(RTArgs, IC.allocation_call.MemoryPointer, &I, /*After=*/true);

  auto GetMemorySize = [&](Type *Ty) {
    auto *LHS = tryToCast(IRB, ACI.SizeLHS, Ty, DL);
    auto *RHS = tryToCast(IRB, ACI.SizeRHS, Ty, DL);
    bool LHSIsUnknown =
        isa<ConstantInt>(LHS) && cast<ConstantInt>(LHS)->isAllOnesValue();
    bool RHSIsUnknown =
        isa<ConstantInt>(RHS) && cast<ConstantInt>(RHS)->isAllOnesValue();
    if (!LHSIsUnknown && !RHSIsUnknown)
      return IRB.CreateMul(LHS, RHS);
    if (LHSIsUnknown)
      return RHS;
    return LHS;
  };
  addValCB(RTArgs, IC.allocation_call.MemorySize, GetMemorySize,
           /*After=*/true);
  addVal(RTArgs, IC.allocation_call.Alignment, ACI.Alignment, /*After=*/true);
  addValCB(
      RTArgs, IC.allocation_call.Family,
      [&](Type *) { return getGlobalString(ACI.Family.value_or("")); },
      /*After=*/true);

  auto GetInitializerKind = [&](Type *) {
    if (!ACI.InitialValue)
      return getCI(Int8Ty, -1);
    if (ACI.InitialValue->isZeroValue())
      return getCI(Int8Ty, 0);
    if (isa<UndefValue>(ACI.InitialValue))
      return getCI(Int8Ty, 1);
    return getCI(Int8Ty, 2);
  };
  addValCB(RTArgs, IC.allocation_call.InitializerKind, GetInitializerKind,
           /*After=*/true);

  auto *CI = getCall(IC.allocation_call.SectionName, RTArgs, /*After=*/true);
  if (IC.allocation_call.ReplaceValue) {
    IRB.SetInsertPoint(CI->getNextNonDebugInstruction());
    I.replaceUsesWithIf(tryToCast(IRB, CI, I.getType(), DL), [&](Use &U) {
      return NewInsts.lookup(cast<Instruction>(U.getUser())) != Epoche;
    });
  }

  return true;
}

bool InstrumentorImpl::instrumentMemoryIntrinsic(IntrinsicInst &I) {
  if (IC.memory_intrinsic.CB && !IC.memory_intrinsic.CB(I))
    return false;

  IRB.SetInsertPoint(&I);

  unsigned KindId = I.getIntrinsicID();
  Value *SrcPtr = nullptr;
  Value *DestPtr = nullptr;
  Value *Length = nullptr;
  Value *MemsetValue = nullptr;
  uint32_t AtomicElementSize = -1;

  switch (KindId) {
  case Intrinsic::memcpy_element_unordered_atomic:
  case Intrinsic::memmove_element_unordered_atomic:
  case Intrinsic::memset_element_unordered_atomic:
    AtomicElementSize = cast<AtomicMemCpyInst>(I).getElementSizeInBytes();
    break;
  default:
    break;
  }

  switch (KindId) {
  case Intrinsic::memcpy:
  case Intrinsic::memcpy_inline:
  case Intrinsic::memcpy_element_unordered_atomic:
  case Intrinsic::memmove:
  case Intrinsic::memmove_element_unordered_atomic: {
    auto &AMC = cast<AnyMemTransferInst>(I);
    SrcPtr = AMC.getRawSource();
    DestPtr = AMC.getRawDest();
    Length = AMC.getLength();
    break;
  }
  case Intrinsic::memset:
  case Intrinsic::memset_inline:
  case Intrinsic::memset_element_unordered_atomic: {
    auto &AMS = cast<AnyMemSetInst>(I);
    DestPtr = AMS.getRawDest();
    Length = AMS.getLength();
    MemsetValue = AMS.getValue();
    break;
  }
  default:
    llvm_unreachable("Unexpected intrinsic");
  }

  SmallVector<RTArgument> RTArgs;
  addCI(RTArgs, IC.memory_intrinsic.KindId, KindId, /*After=*/false);
  addVal(RTArgs, IC.memory_intrinsic.DestinationPointer, DestPtr,
         /*After=*/false);
  addCI(RTArgs, IC.memory_intrinsic.DestinationPointerAddressSpace,
        DestPtr->getType()->getPointerAddressSpace(), /*After=*/false);
  addVal(RTArgs, IC.memory_intrinsic.SourcePointer,
         SrcPtr ? SrcPtr : Constant::getAllOnesValue(PtrTy), /*After=*/false);
  addCI(RTArgs, IC.memory_intrinsic.SourcePointerAddressSpace,
        SrcPtr ? SrcPtr->getType()->getPointerAddressSpace() : -1,
        /*After=*/false);

  addVal(RTArgs, IC.memory_intrinsic.MemsetValue,
         MemsetValue ? MemsetValue
                     : Constant::getAllOnesValue(
                           IC.memory_intrinsic.MemsetValue.getType(Ctx)),
         /*After=*/false);
  addVal(RTArgs, IC.memory_intrinsic.Length, Length, /*After=*/false);
  addCI(RTArgs, IC.memory_intrinsic.IsVolatile, I.isVolatile(),
        /*After=*/false);
  addCI(RTArgs, IC.memory_intrinsic.AtomicElementSize, AtomicElementSize,
        /*After=*/false);

  getCall(IC.memory_intrinsic.SectionName, RTArgs, /*After=*/false);
  return true;
}

bool InstrumentorImpl::instrumentGeneralIntrinsic(IntrinsicInst &I) {
  if (!IC.intrinsic.Enabled)
    return false;
  if (IC.intrinsic.CB && !IC.intrinsic.CB(I))
    return false;

  IRB.SetInsertPoint(&I);

  SmallVector<RTArgument> RTArgs;
  addCI(RTArgs, IC.intrinsic.KindId, I.getIntrinsicID(),
        /*After=*/false);

  getCall(IC.intrinsic.SectionName, RTArgs, /*After=*/false);

  return true;
}

template <typename MemoryInstTy>
bool InstrumentorImpl::analyzeAccess(MemoryInstTy &I) {
  Value *Ptr = I.getPointerOperand();

  APInt Offset(DL.getIndexSizeInBits(0), 0);
  Value *BasePtr = Ptr->stripAndAccumulateConstantOffsets(DL, Offset, true);

  Type *AllocTy = nullptr;
  if (auto *AI = dyn_cast<AllocaInst>(BasePtr))
    AllocTy = AI->getAllocatedType();
#if 0
  // TODO: Implement constant offset optimization for globals
  else if (auto *GV = dyn_cast<GlobalVariable>(BasePtr))
    AllocTy = GV->getValueType();
#endif

  if (!AllocTy)
    return false;

  APInt OffsetPlusSize = Offset + DL.getTypeStoreSize(I.getAccessType());
  APInt AllocSize(OffsetPlusSize.getBitWidth(), DL.getTypeStoreSize(AllocTy));
  if (AllocSize.ult(OffsetPlusSize))
    return false;

  I.addAnnotationMetadata(COAnnotation);

  return true;
}

bool InstrumentorImpl::instrumentStore(StoreInst &I) {
  if (!IC.store.Enabled)
    return false;
  if (IC.store.CB && !IC.store.CB(I))
    return false;
  if (hasAnnotation(&I, SkipAnnotation))
    return false;
  if (hasAnnotation(&I, COAnnotation))
    return false;

  IRB.SetInsertPoint(&I);

  SmallVector<RTArgument> RTArgs;
  addVal(RTArgs, IC.store.PointerOperand, I.getPointerOperand(),
         /*After=*/false);
  addCI(RTArgs, IC.store.PointerOperandAddressSpace, I.getPointerAddressSpace(),
        /*After=*/false);
  addIndVal(RTArgs, IC.store.ValueOperand, I.getValueOperand(), I.getFunction(),
            /*After=*/false);
  addCI(RTArgs, IC.store.ValueOperandSize,
        DL.getTypeSizeInBits(I.getValueOperand()->getType()), /*After=*/false);
  addCI(RTArgs, IC.store.ValueOperandTypeId,
        I.getValueOperand()->getType()->getTypeID(), /*After=*/false);
  addCI(RTArgs, IC.store.Alignment, I.getAlign().value(), /*After=*/false);
  addCI(RTArgs, IC.store.AtomicityOrdering, uint64_t(I.getOrdering()),
        /*After=*/false);
  addCI(RTArgs, IC.store.SyncScopeId, uint64_t(I.getSyncScopeID()),
        /*After=*/false);
  addCI(RTArgs, IC.store.IsVolatile, I.isVolatile(), /*After=*/false);

  addValCB(
      RTArgs, IC.store.BasePointerInfo,
      [&](Type *Ty) -> Value * {
        return findBasePointer(I.getPointerOperand());
      },
      /*After=*/false);

  auto *CI = getCall(IC.store.SectionName, RTArgs, /*After=*/false);
  if (IC.store.ReplacePointerOperand)
    I.setOperand(I.getPointerOperandIndex(), CI);

  return true;
}

bool InstrumentorImpl::instrumentLoad(LoadInst &I, bool After) {
  if (!IC.load.Enabled)
    return false;
  if (!After && !IC.load.InstrumentBefore)
    return false;
  if (After && !IC.load.InstrumentAfter)
    return false;
  if (IC.load.CB && !IC.load.CB(I))
    return false;
  if (hasAnnotation(&I, SkipAnnotation))
    return false;
  if (hasAnnotation(&I, COAnnotation))
    return false;

  if (After)
    IRB.SetInsertPoint(I.getNextNonDebugInstruction());
  else
    IRB.SetInsertPoint(&I);

  SmallVector<RTArgument> RTArgs;
  addVal(RTArgs, IC.load.PointerOperand, I.getPointerOperand(), After);
  addCI(RTArgs, IC.load.PointerOperandAddressSpace, I.getPointerAddressSpace(),
        After);

  AllocaInst *IndirectionAI =
      addIndVal(RTArgs, IC.load.Value, &I, I.getFunction(), After,
                /*ForceIndirection=*/false);

  addCI(RTArgs, IC.load.ValueSize, DL.getTypeStoreSize(I.getType()), After);
  addCI(RTArgs, IC.load.ValueTypeId, I.getType()->getTypeID(), After);
  addCI(RTArgs, IC.load.Alignment, I.getAlign().value(), After);
  addCI(RTArgs, IC.load.AtomicityOrdering, uint64_t(I.getOrdering()), After);
  addCI(RTArgs, IC.load.SyncScopeId, uint64_t(I.getSyncScopeID()), After);
  addCI(RTArgs, IC.load.IsVolatile, I.isVolatile(), After);

  addValCB(
      RTArgs, IC.store.BasePointerInfo,
      [&](Type *Ty) -> Value * {
        return findBasePointer(I.getPointerOperand());
      },
      After);

  bool ReplaceValue = IC.load.ReplaceValue && After;
  bool ReplacePointerOperand = IC.load.ReplacePointerOperand && !After;

  auto *CI = getCall(IC.load.SectionName, RTArgs, After);
  if (ReplaceValue) {
    IRB.SetInsertPoint(CI->getNextNonDebugInstruction());
    Value *NewV = IndirectionAI ? IRB.CreateLoad(I.getType(), IndirectionAI)
                                : tryToCast(IRB, CI, I.getType(), DL);
    I.replaceUsesWithIf(NewV, [&](Use &U) {
      return NewInsts.lookup(cast<Instruction>(U.getUser())) != Epoche;
    });
  } else if (ReplacePointerOperand) {
    I.setOperand(I.getPointerOperandIndex(), CI);
  }

  return true;
}

bool InstrumentorImpl::instrumentUnreachable(UnreachableInst &I) {
  if (!IC.unreachable.Enabled)
    return false;
  if (IC.unreachable.CB && !IC.unreachable.CB(I))
    return false;

  IRB.SetInsertPoint(&I);
  SmallVector<RTArgument> RTArgs;
  getCall(IC.unreachable.SectionName, RTArgs,
          /*After=*/false);

  return true;
}

bool InstrumentorImpl::instrumentFunction(Function &Fn) {
  bool Changed = false;
  if (!shouldInstrumentFunction(&Fn))
    return Changed;

  if (IC.load.SkipSafeAccess || IC.load.SkipSafeAccess) {
    // TODO: Merge this into the main loop with RPOT
    for (auto &BB : Fn) {
      for (auto &I : BB) {
        switch (I.getOpcode()) {
        case Instruction::Load:
          if (IC.load.SkipSafeAccess)
            Changed |= analyzeAccess(cast<LoadInst>(I));
          break;
        case Instruction::Store:
          if (IC.store.SkipSafeAccess)
            Changed |= analyzeAccess(cast<StoreInst>(I));
          break;
        default:
          break;
        }
      }
    }
  }

  if (IC.base_pointer.Enabled)
    for (auto &Arg : Fn.args())
      if (Arg.getType() == PtrTy)
        instrumentBasePointer(Arg);

  ReversePostOrderTraversal<Function *> RPOT(&Fn);
  for (auto &It : RPOT) {
    for (auto &I : *It) {
      // Skip instrumentation instructions.
      if (NewInsts.contains(&I))
        continue;

      // Count epochs eagerly.
      ++Epoche;

      switch (I.getOpcode()) {
      case Instruction::Alloca:
      case Instruction::Call:
      case Instruction::Load:
      case Instruction::PHI:
          if (I.getType() == PtrTy)
            Changed |= instrumentBasePointer(I);
        break;
      default:
        break;
      }

      switch (I.getOpcode()) {
      case Instruction::Alloca:
        Changed |= instrumentAlloca(IC.alloca, cast<AllocaInst>(I));
        break;
      case Instruction::Call:
          Changed |= instrumentCall(cast<CallBase>(I));
        break;
      case Instruction::Load:
          Changed |= instrumentLoad(cast<LoadInst>(I), /*After=*/false);
          Changed |= instrumentLoad(cast<LoadInst>(I), /*After=*/true);
        break;
      case Instruction::Store:
          Changed |= instrumentStore(cast<StoreInst>(I));
        break;
      case Instruction::Unreachable:
          Changed |= instrumentUnreachable(cast<UnreachableInst>(I));
        break;
      default:
        break;
      }
    }
  }

  // TODO: Can be merged into the previous loop
  if (IC.call_arg.Enabled) {
    for (auto &It : RPOT)
      for (auto &I : *It)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (!NewInsts.contains(CI))
            Changed |= instrumentCallArgs(*CI);
  }

  return Changed;
}

bool InstrumentorImpl::instrumentMainFunction(Function &MainFn) {
  if (!shouldInstrumentFunction(&MainFn))
    return false;
  if (IC.main_function.CB && !IC.main_function.CB(MainFn))
    return false;

  std::string MainFnName = getRTName("", "main");
  MainFn.setName(MainFnName);

  Function *InstMainFn = Function::Create(
      MainFn.getFunctionType(), GlobalValue::ExternalLinkage, "main", M);

  auto *EntryBB = BasicBlock::Create(Ctx, "entry", InstMainFn);
  IRB.SetInsertPoint(EntryBB, EntryBB->getFirstNonPHIOrDbgOrAlloca());

  SmallVector<RTArgument> RTArgs;
  AllocaInst *IndirectionAIArgC = nullptr, *IndirectionAIArgV = nullptr;
  IndirectionAIArgC = addIndVal(
      RTArgs, IC.main_function.ArgC,
      InstMainFn->arg_size() ? cast<Value>(InstMainFn->arg_begin())
                             : getCI(IC.main_function.ArgC.getType(Ctx), 0),
      InstMainFn,
      /*After=*/false,
      /*ForceIndirection=*/IC.main_function.ReplaceArgumentValues);
  IndirectionAIArgV = addIndVal(
      RTArgs, IC.main_function.ArgV,
      InstMainFn->arg_size() > 1 ? cast<Value>(InstMainFn->arg_begin() + 1)
                                 : NullPtrVal,
      InstMainFn,
      /*After=*/false,
      /*ForceIndirection=*/IC.main_function.ReplaceArgumentValues);

  if (IC.main_function.InstrumentBefore)
    getCall(IC.main_function.SectionName, RTArgs, /*After=*/false);

  SmallVector<Value *> UserMainArgs;
  if (IC.main_function.ReplaceArgumentValues) {
    if (InstMainFn->arg_size())
      UserMainArgs.push_back(IRB.CreateLoad(IC.main_function.ArgC.getType(Ctx),
                                            IndirectionAIArgC));
    if (InstMainFn->arg_size() > 1)
      UserMainArgs.push_back(IRB.CreateLoad(IC.main_function.ArgV.getType(Ctx),
                                            IndirectionAIArgV));
  } else {
    for (auto &A : InstMainFn->args())
      UserMainArgs.push_back(&A);
  }

  FunctionCallee FnCallee =
      M.getOrInsertFunction(MainFnName, MainFn.getFunctionType());
  Value *ReturnValue = IRB.CreateCall(FnCallee, UserMainArgs);

  if (IC.main_function.InstrumentAfter) {
    addVal(RTArgs, IC.main_function.ReturnValue, ReturnValue, /*After=*/true);

    auto *CI = getCall(IC.main_function.SectionName, RTArgs, /*After=*/true);
    if (IC.main_function.ReplaceReturnValue)
      ReturnValue = CI;
  }
  IRB.CreateRet(ReturnValue);

  return true;
}

bool InstrumentorImpl::instrumentModule(bool After) {
  Function *YtorFn = After ? DtorFn : CtorFn;
  assert(YtorFn);

  IRB.SetInsertPointPastAllocas(YtorFn);

  SmallVector<RTArgument> RTArgs;
  addValCB(
      RTArgs, IC.module.ModuleName,
      [&](Type *) { return getGlobalString(M.getName()); }, After);
  addValCB(
      RTArgs, IC.module.ModuleTargetTriple,
      [&](Type *) { return getGlobalString(M.getTargetTriple()); }, After);

  getCall(IC.module.SectionName, RTArgs, After);

  return true;
}

bool InstrumentorImpl::prepareGlobalVariables() {
  bool Changed = false;

  for (GlobalVariable &GV : M.globals()) {
    if (!shouldInstrumentGlobalVariable(&GV))
      continue;

    if (GV.isDSOLocal()) {
      Globals.push_back(&GV);
    } else {
      // TODO: Find a better way to skip external globals (e.g., stderr, stdout)
      for (Use &U : GV.uses()) {
        assert(isa<Instruction>(U.getUser()));
        Instruction *I = cast<Instruction>(U.getUser());
        I->addAnnotationMetadata(SkipAnnotation);
        Changed = true;
      }
    }
  }

  for (GlobalVariable *GV : Globals) {
    SmallVector<Use *> Uses;
    for (Use &U : GV->uses())
      Uses.push_back(&U);

    for (Use *U : Uses) {
      auto *E = dyn_cast<ConstantExpr>(U->getUser());
      if (!E)
        continue;

      SmallVector<Use *> CEUses;
      for (Use &CEU : E->uses())
        CEUses.push_back(&CEU);

      for (Use *CEU : CEUses) {
        if (auto *UI = dyn_cast<Instruction>(CEU->getUser())) {
          if (shouldInstrumentFunction(UI->getFunction())) {
            Instruction *NewI = E->getAsInstruction();
            NewI->insertBefore(UI);
            CEU->set(NewI);
            Changed = true;
          }
        }
      }
    }
  }

  return Changed;
}

bool InstrumentorImpl::instrumentGlobalVariables() {
  IRB.SetInsertPointPastAllocas(CtorFn);

  if (Globals.empty())
    return false;

  SmallVector<GlobalVariable *> InstrGlobals;
  for (GlobalVariable *GV : Globals) {
    GlobalVariable *InstrGV =
        new GlobalVariable(M, PtrTy, false, GlobalValue::PrivateLinkage,
                           NullPtrVal, getRTName("", GV->getName()));

    SmallVector<RTArgument> RTArgs;
    addVal(RTArgs, IC.global_var.Value, GV, /*After=*/true);
    addCI(RTArgs, IC.global_var.Size, DL.getTypeAllocSize(GV->getValueType()),
          /*After=*/true);
    addCI(RTArgs, IC.global_var.Alignment, GV->getAlign().valueOrOne().value(),
          /*After=*/true);
    addCI(RTArgs, IC.global_var.IsConstant, GV->isConstant(), /*After=*/true);
    addCI(RTArgs, IC.global_var.UnnamedAddress, int64_t(GV->getUnnamedAddr()),
          /*After=*/true);
    addValCB(
        RTArgs, IC.global_var.Name,
        [&](Type *) { return getGlobalString(GV->getName()); }, /*After=*/true);
    auto GetInitializerKind = [&](Type *) {
      auto *InitialValue = GV->getInitializer();
      if (!InitialValue)
        return getCI(Int8Ty, -1);
      if (InitialValue->isZeroValue())
        return getCI(Int8Ty, 0);
      if (isa<UndefValue>(InitialValue))
        return getCI(Int8Ty, 1);
      return getCI(Int8Ty, 2);
    };
    addValCB(RTArgs, IC.global_var.InitializerKind, GetInitializerKind,
             /*After=*/true);

    auto *CI = getCall(IC.global_var.SectionName, RTArgs, /*After=*/true);
    if (IC.global_var.ReplaceValue)
      IRB.CreateStore(CI, InstrGV);

    InstrGlobals.push_back(InstrGV);
  }

  for (size_t G = 0; G < Globals.size(); ++G) {
    GlobalVariable *GV = Globals[G];

    SmallVector<Use *> Uses;
    for (Use &U : GV->uses())
      Uses.push_back(&U);

    SmallVector<Use *> ReplaceUses;
    for (Use *U : Uses) {
      if (auto *I = dyn_cast<Instruction>(U->getUser())) {
        if (shouldInstrumentFunction(I->getFunction()))
          ReplaceUses.push_back(U);
      } else if (auto *E = dyn_cast<ConstantExpr>(U->getUser())) {
      } else {
        llvm_unreachable("Unexpected global variable use");
      }
    }

    for (Use *U : ReplaceUses) {
#if 0
      // TODO: Implement constant offset optimization for globals
      Instruction *LoadI = nullptr;
      Instruction *StoreI = nullptr;
      Instruction *UserI = cast<Instruction>(U->getUser());
      if (auto *LI = dyn_cast<LoadInst>(UserI)) {
        MI = LI;
      } else if (auto *SI = dyn_cast<StoreInst>(UserI)) {
        if (SI->getPointerOperand() == GV)
        MemI = nullptr;
      } else if (isa<GetElementPtrInst>(UserI)) {
        SmallVector<Use *> GEPUses;
        for (Use &GEPU : UserI->uses())
          GEPUses.push_back(&GEPU);
        assert(GEPUses.size() == 1);

        Value *Val = GEPUses[0]->getUser();
        assert(isa<LoadInst>(Val) || isa<StoreInst>(Val));
        MemI = cast<Instruction>(Val);
      }
      assert(MemI);

      if (hasAnnotation(MemI, COAnnotation))
        continue;
#endif
      Instruction *UserI = cast<Instruction>(U->getUser());
      IRB.SetInsertPoint(UserI);
      auto *LoadPtr = IRB.CreateLoad(PtrTy, InstrGlobals[G]);
      U->set(LoadPtr);
    }
  }

  return true;
}

bool InstrumentorImpl::instrumentBasePointer(Value &ArgOrInst) {
  if (!IC.base_pointer.Enabled)
    return false;
  if (auto *Arg = dyn_cast<Argument>(&ArgOrInst)) {
    IRB.SetInsertPointPastAllocas(Arg->getParent());
  } else if (auto *I = dyn_cast<Instruction>(&ArgOrInst)) {
    do {
      I = I->getNextNonDebugInstruction();
    } while (isa<AllocaInst>(I));
    IRB.SetInsertPoint(I);
  }

  SmallVector<RTArgument> RTArgs;
  addVal(RTArgs, IC.base_pointer.Value, &ArgOrInst, /*After=*/true);

  auto *CI = getCall(IC.base_pointer.SectionName, RTArgs, /*After=*/true);
  BasePtrMap[&ArgOrInst] = CI;

  return true;
}

bool InstrumentorImpl::removeUnusedBasePointers() {
  bool Changed = false;

  for (auto &Entry : BasePtrMap) {
    CallInst *CI = Entry.second;
    if (!CI->getNumUses()) {
      CI->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

Value *InstrumentorImpl::findBasePointer(Value *V) {
  if (!IC.base_pointer.Enabled)
    return NullPtrVal;

  while (auto *I = dyn_cast<Instruction>(V)) {
    bool KeepSearching = false;
    switch (I->getOpcode()) {
    case Instruction::GetElementPtr:
      V = cast<GetElementPtrInst>(I)->getPointerOperand();
      KeepSearching = true;
      break;
    case Instruction::AddrSpaceCast:
      V = cast<AddrSpaceCastInst>(I)->getPointerOperand();
      KeepSearching = true;
      break;
    default:
      break;
    }

    if (!KeepSearching)
      break;
  }

  auto It = BasePtrMap.find(V);
  if (It == BasePtrMap.end())
    return NullPtrVal;
  return It->second;
}

void InstrumentorImpl::addCtorOrDtor(bool Ctor) {
  Function *YtorFn = Function::Create(FunctionType::get(VoidTy, false),
                                      GlobalValue::PrivateLinkage,
                                      getRTName("", Ctor ? "ctor" : "dtor"), M);

  auto *EntryBB = BasicBlock::Create(Ctx, "entry", YtorFn);
  IRB.SetInsertPoint(EntryBB, EntryBB->getFirstNonPHIOrDbgOrAlloca());
  IRB.CreateRetVoid();

  if (Ctor) {
    appendToGlobalCtors(M, YtorFn, 0);
    CtorFn = YtorFn;
  } else {
    appendToGlobalDtors(M, YtorFn, 0);
    DtorFn = YtorFn;
  }
}

bool InstrumentorImpl::instrument() {
  bool Changed = false;
  printRuntimeSignatures();

  Function *MainFn = nullptr;

  if (IC.module.InstrumentBefore || IC.global_var.Enabled)
    addCtorOrDtor(/*Ctor=*/true);
  if (IC.module.InstrumentAfter)
    addCtorOrDtor(/*Ctor=*/false);

  if (IC.module.InstrumentBefore)
    Changed |= instrumentModule(/*After=*/false);
  if (IC.module.InstrumentAfter)
    Changed |= instrumentModule(/*After=*/true);

  if (IC.global_var.Enabled)
    Changed |= prepareGlobalVariables();

  for (Function &Fn : M) {
    Changed |= instrumentFunction(Fn);

    if (Fn.getName() == "main")
      MainFn = &Fn;
  }

  if (IC.global_var.Enabled)
    Changed |= instrumentGlobalVariables();

  if (MainFn &&
      (IC.main_function.InstrumentBefore || IC.main_function.InstrumentAfter))
    Changed |= instrumentMainFunction(*MainFn);

  if (IC.base_pointer.SkipUnused)
    Changed |= removeUnusedBasePointers();

  return Changed;
}

PreservedAnalyses InstrumentorPass::run(Module &M, ModuleAnalysisManager &MAM) {
  InstrumentorImpl Impl(IC, M, MAM);
  if (!readInstrumentorConfigFromJSON(IC))
    return PreservedAnalyses::all();
  writeInstrumentorConfig(IC);

  if (!Impl.instrument())
    return PreservedAnalyses::all();

  if (verifyModule(M))
    M.dump();
  assert(!verifyModule(M, &errs()));
  return PreservedAnalyses::none();
}
