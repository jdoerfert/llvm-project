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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
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
#include "llvm/IR/Instruction.h"
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
#include <filesystem>
#include <fstream>
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

cl::opt<std::string> ReadDatabaseDir(
    "instrumentor-read-database-dir",
    cl::desc(
        "Read the externally instrumented entities from the directory"),
    cl::init(""));

cl::opt<std::string> WriteDatabaseDir(
    "instrumentor-write-database-dir",
    cl::desc(
        "Write the locally instrumented entities into the directory"),
    cl::init(""));

namespace {

struct RTArgumentPack;

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
  if (IC.SECTION.canRunPre())                                                  \
    J.attribute("EnabledPre", IC.SECTION.EnabledPre);                          \
  if (IC.SECTION.canRunPost())                                                 \
    J.attribute("EnabledPost", IC.SECTION.EnabledPost);
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
    if (POSITION & InstrumentorConfig::PRE)                                    \
      if (auto *Val = Obj->get("EnabledPre"))                                  \
        if (!json::fromJSON(*Val, IC.SECTION.EnabledPre, NullRoot))            \
          errs() << "WARNING: Failed to read " #SECTION                        \
                    ".EnabledPre as bool\n";                                   \
    if (POSITION & InstrumentorConfig::POST)                                   \
      if (auto *Val = Obj->get("EnabledPost"))                                 \
        if (!json::fromJSON(*Val, IC.SECTION.EnabledPost, NullRoot))           \
          errs() << "WARNING: Failed to read " #SECTION                        \
                    ".EnabledPost as bool\n";                                  \
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
    return CheckOnly ? V : IRB.CreateIntCast(V, Ty, /*IsSigned=*/ false);
  if (VTy->isIntegerTy() && Ty->isFloatingPointTy()) {
    if (DL.getTypeSizeInBits(VTy) > DL.getTypeSizeInBits(Ty))
      return tryToCast(IRB,
                       CheckOnly
                           ? IRB.getIntN(DL.getTypeSizeInBits(VTy) / 2, 0)
                           : IRB.CreateIntCast(
                                 V, IRB.getIntNTy(DL.getTypeSizeInBits(VTy) / 2), /*IsSigned=*/false),
                       Ty, DL);
    return nullptr;
  }
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

template <typename Ty> Constant *getCI(Type *IT, Ty Val) {
  return ConstantInt::get(IT, Val);
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
    int HasPotentiallyIndirect[3] = {0, 0, 0};
    StringRef Name;

#define CVALUE(SECTION, TYPE, NAME, DEFAULT_VALUE)
#define CVALUE_INTERNAL(SECTION, TYPE, NAME, DEFAULT_VALUE)
#define RTVALUE(SECTION, NAME, DEFAULT_VALUE, VALUE_TYPE_STR, PROPERTIES)      \
  ConfigValues.push_back(&IC.SECTION.NAME);                                    \
  for (auto &Pos : {InstrumentorConfig::PRE, InstrumentorConfig::POST})        \
    if (IC.SECTION.NAME.isEnabled(Pos))                                        \
      HasPotentiallyIndirect[Pos] +=                                           \
          bool(IC.SECTION.NAME.getKind() &                                     \
               InstrumentorKindTy::POTENTIALLY_INDIRECT);

#define SECTION_START(SECTION, POSITION)                                       \
  Name = IC.SECTION.SectionName;                                               \
  HasPotentiallyIndirect[1] = HasPotentiallyIndirect[2] = 0;                   \
  Position = POSITION;

#define SECTION_END(SECTION)                                                   \
  for (auto &Pos : {InstrumentorConfig::PRE, InstrumentorConfig::POST}) {      \
    if (Position & Pos) {                                                      \
      if (HasPotentiallyIndirect[Pos] < 2)                                     \
        printStubRTDefinitions(DeclOut, StubRTOut, Name, ConfigValues, Pos,    \
                               /*Indirect=*/false);                            \
      if (HasPotentiallyIndirect[Pos])                                         \
        printStubRTDefinitions(DeclOut, StubRTOut, Name, ConfigValues, Pos,    \
                               /*Indirect=*/true);                             \
    }                                                                          \
  }                                                                            \
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

  void loadInstrumentorDatabase();
  void updateInstrumentorDatabase();

  /// Get a temporary alloca to communicate (large) values with the runtime.
  AllocaInst *getAlloca(Function *Fn, Type *Ty) {
    auto &AllocaList = AllocaMap[{Fn, DL.getTypeAllocSize(Ty)}];
    if (AllocaList.empty())
      return new AllocaInst(Ty, DL.getAllocaAddrSpace(), "",
                            Fn->getEntryBlock().getFirstNonPHIOrDbgOrAlloca());
    return AllocaList.pop_back_val();
  }

  /// Return the temporary allocas.
  void returnAllocas(SmallVector<AllocaInst *> &TmpAllocas) {
    if (TmpAllocas.empty())
      return;

    for (AllocaInst *AI : TmpAllocas) {
      auto &AllocaList = AllocaMap[{AI->getFunction(), DL.getTypeAllocSize(AI->getAllocatedType())}];
      AllocaList.push_back(AI);
    }
    TmpAllocas.clear();
  }

private:
  bool shouldInstrumentFunction(Function *Fn);
  bool shouldInstrumentGlobalVariable(GlobalVariable *GV);
  bool shouldInstrumentAccess(Value *Ptr, Instruction *MI = nullptr);

  bool instrumentFunction(Function &Fn);
  bool instrumentAlloca(AllocaInst &I, InstrumentorConfig::Position P);
  bool instrumentCall(CallBase &I, InstrumentorConfig::Position P);
  bool instrumentGenericCall(CallBase &I, InstrumentorConfig::Position P);
  bool instrumentAllocationCall(CallBase &I, const AllocationCallInfo &ACI,
                                InstrumentorConfig::Position P);
  bool instrumentMemoryIntrinsic(IntrinsicInst &I,
                                 InstrumentorConfig::Position P);
  bool instrumentGeneralIntrinsic(IntrinsicInst &I,
                                  InstrumentorConfig::Position P);
  bool instrumentLoad(LoadInst &I, InstrumentorConfig::Position P);
  bool instrumentStore(StoreInst &I, InstrumentorConfig::Position P);
  bool instrumentUnreachable(UnreachableInst &I,
                             InstrumentorConfig::Position P);
  bool instrumentMainFunction(Function &MainFn);
  bool instrumentModule(InstrumentorConfig::Position P);

  DenseMap<Value *, CallInst *> BasePtrMap;
  bool instrumentBasePointer(Value &ArgOrInst);
  bool removeUnusedBasePointers();
  Value *findBasePointer(Value *V);

  template <typename MemoryInstTy> bool analyzeAccess(MemoryInstTy &I);

  bool prepareGlobalVariables();
  bool instrumentGlobalVariables();

  void addCtorOrDtor(bool Ctor);

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

  CallInst *getCall(StringRef Name, RTArgumentPack &RTArgPack,
                    InstrumentorConfig::Position P);

  void setInsertPoint(Instruction &I, InstrumentorConfig::Position P) {
    if (P & InstrumentorConfig::PRE) {
      IRB.SetInsertPoint(&I);
      return;
    }
    Instruction *IP = I.getNextNonDebugInstruction();
    if (isa<AllocaInst>(I))
      while (isa<AllocaInst>(IP))
        IP = IP->getNextNonDebugInstruction();
    IRB.SetInsertPoint(IP);
  }

  void printStubRTDefinitions(
      raw_ostream *SignatureOut, raw_ostream *StubRTOut, StringRef Name,
      SmallVectorImpl<InstrumentorConfig::ConfigValue *> &ConfigValues,
      InstrumentorConfig::Position P, bool Indirect) {

    [[maybe_unused]] bool DirectReturn = false;
    StringRef ReturnedVariable;
    Type *RetTy = VoidTy;
    for (auto *CV : ConfigValues) {
      if (!CV->isEnabled(P))
        continue;
      bool ReplaceablePre =
          (CV->getKind() & InstrumentorKindTy::REPLACABLE_PRE);
      bool ReplaceablePost =
          (CV->getKind() & InstrumentorKindTy::REPLACABLE_POST);
      if (((!ReplaceablePre && (P & InstrumentorConfig::PRE)) ||
           (!ReplaceablePost && (P & InstrumentorConfig::POST))))
        continue;
      if ((ReplaceablePre && (P & InstrumentorConfig::PRE)) ||
          (ReplaceablePost && (P & InstrumentorConfig::POST))) {
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
    auto CompleteName = getRTName(P, Name, Indirect ? "_ind" : "");
    StrOut << CompleteName << "(";
    for (auto *CV : ConfigValues) {
      if (!CV->isEnabled(P))
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
      if (!CV->isEnabled(P))
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

  /// Mapping to remember temporary allocas for reuse.
  DenseMap<std::pair<Function *, unsigned>, SmallVector<AllocaInst *>> AllocaMap;

  /// The module's ctor and dtor functions.
  Function *CtorFn = nullptr;
  Function *DtorFn = nullptr;

  // The functions and globals that are instrumented within the module.
  SmallVector<Function *> Functions;
  SmallVector<GlobalVariable *> Globals;

  // The original globals that are not instrumented.
  SmallSet<GlobalVariable *, 20> SkippedGlobals;

  // The original memory instructions that should not be instrumented. For
  // instance, the accesses that are safe. See load.SkipSafeAccess and
  // store.SkipSafeAccess.
  SmallSet<Instruction *, 20> SkippedMemInsts;

  StringSet<> ExternalInstrFunctions;
  StringSet<> ExternalInstrGlobals;

  static constexpr StringRef DatabaseExt = ".instrdb";
  static constexpr StringRef DatabaseSectionFunctions = "[functions]";
  static constexpr StringRef DatabaseSectionGlobals = "[globals]";

protected:
  friend struct RTArgumentPack;

  /// Commonly used values for IR inspection and creation.
  ///{

  /// The underying LLVM context.
  LLVMContext &Ctx;

  /// A special IR builder that keeps track of the inserted instructions.
  IRBuilder<ConstantFolder, IRBuilderCallbackInserter> IRB;

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

struct RTArgument final {
  RTArgument(Type *Ty, StringRef Name, InstrumentorKindTy Kind = PLAIN)
      : Ty(Ty), Name(Name), Kind(Kind) {}
  RTArgument(Value *V, StringRef Name, InstrumentorKindTy Kind = PLAIN)
      : V(V), Ty(V->getType()), Name(Name), Kind(Kind) {}

  Value *V = nullptr;
  Type *Ty = nullptr;
  std::string Name;
  InstrumentorKindTy Kind;
};

struct RTArgumentPack final {
  RTArgumentPack(InstrumentorImpl &I) : Impl(I) {}

  ~RTArgumentPack() {
    if (!TmpAllocas.empty())
      Impl.returnAllocas(TmpAllocas);
  }

  auto &getArgs() { return Args; }

  void addVal(InstrumentorConfig::ConfigValue &Obj, Value *V,
              InstrumentorConfig::Position P, AllocaInst *AI = nullptr) {
    if (AI)
      TmpAllocas.push_back(AI);
    if (!Obj.isEnabled(P))
      return;
    Type *Ty = Obj.getType(Impl.Ctx);
    V = tryToCast(Impl.IRB, V, Ty, Impl.DL);
    assert(V->getType() == Ty);
    Args.emplace_back(V, Obj.getName(), Obj.getKind());
  }

  AllocaInst *addIndVal(InstrumentorConfig::ConfigValue &Obj, Value *V,
                        Function *F, InstrumentorConfig::Position P,
                        bool ForceIndirection = false) {
    if (!Obj.isEnabled(P))
      return nullptr;
    if (!ForceIndirection &&
        tryToCast(Impl.IRB, UndefValue::get(Obj.getType(Impl.Ctx)), V->getType(), Impl.DL,
                  /*CheckOnly=*/true) &&
        tryToCast(Impl.IRB, V, Obj.getType(Impl.Ctx), Impl.DL, /*CheckOnly=*/true)) {
      addVal(Obj, V, P);
      return nullptr;
    }
    auto *IndirectionAI = Impl.getAlloca(F, V->getType());
    Impl.IRB.CreateStore(V, IndirectionAI);
    Args.emplace_back(IndirectionAI, std::string(Obj.getName()) + "Ind",
                      Obj.getKind());
    TmpAllocas.push_back(IndirectionAI);
    return IndirectionAI;
  }

  void addValCB(InstrumentorConfig::ConfigValue &Obj,
                function_ref<Value *(Type *)> ValFn,
                InstrumentorConfig::Position P) {
    if (!Obj.isEnabled(P))
      return;
    Type *Ty = Obj.getType(Impl.Ctx);
    Value *V = ValFn(Ty);
    assert(V->getType() == Ty);
    Args.emplace_back(tryToCast(Impl.IRB, V, Ty, Impl.DL), Obj.getName(),
                      Obj.getKind());
  }

  void addCI(InstrumentorConfig::ConfigValue &Obj, uint64_t Value,
             InstrumentorConfig::Position P) {
    if (!Obj.isEnabled(P))
      return;
    Args.emplace_back(getCI(Obj.getType(Impl.Ctx), Value), Obj.getName(),
                      Obj.getKind());
  }

private:
  InstrumentorImpl &Impl;
  SmallVector<RTArgument> Args;
  SmallVector<AllocaInst *> TmpAllocas;
};

} // end anonymous namespace

namespace fs = std::filesystem;

void InstrumentorImpl::loadInstrumentorDatabase() {
  const std::string &DatabaseDir = ReadDatabaseDir;
  if (DatabaseDir.empty())
    return;

  std::string CurrentFileName = M.getSourceFileName() + DatabaseExt.data();

  fs::path Directory(DatabaseDir);
  if (!fs::exists(Directory) || !fs::is_directory(Directory)) {
    errs() << "Instrumentor database directory does not exists\n";
    return;
  }

  for (const auto& entry : fs::directory_iterator(Directory)) {
    if (!fs::is_regular_file(entry.status()))
      continue;

    const fs::path &FilePath = entry.path();
    fs::path FileName = FilePath.filename();
    StringRef FileNameStr(FileName.c_str());
    if (!FileNameStr.ends_with(DatabaseExt))
      continue;
    if (FileNameStr == StringRef(CurrentFileName))
      continue;

    std::ifstream File(FilePath.string());
    if (!File.is_open())
      continue;

    errs() << "Loading instrumented entities from " << FilePath << "\n";

    std::string Line;
    std::getline(File, Line);
    assert(Line == DatabaseSectionFunctions);

    while (std::getline(File, Line) && Line != DatabaseSectionGlobals)
      ExternalInstrFunctions.insert(Line.c_str());
    assert(Line == DatabaseSectionGlobals);

    while (std::getline(File, Line))
      ExternalInstrGlobals.insert(Line.c_str());

    File.close();
  }
}

void InstrumentorImpl::updateInstrumentorDatabase() {
  const std::string &DatabaseDir = WriteDatabaseDir;
  if (DatabaseDir.empty())
    return;

  fs::path Directory(DatabaseDir);
  fs::create_directory(Directory);

  std::string FileName = M.getSourceFileName() + DatabaseExt.data();

  fs::path FilePath = Directory / FileName;
  if (fs::exists(FilePath) && !fs::is_regular_file(FilePath)) {
    errs() << "Instrumentor database file " << FileName << " is not a regular file\n";
    return;
  }

  std::ofstream File(FilePath.string());
  assert(File.is_open());

  errs() << "Writing instrumented entities to " << FilePath << "\n";

  File << DatabaseSectionFunctions.data() << std::endl;
  for (const Function *Fn : Functions)
    File << Fn->getName().data() << std::endl;

  File << DatabaseSectionGlobals.data() << std::endl;
  for (const GlobalVariable *GV : Globals)
    File << GV->getName().data() << std::endl;
}

bool InstrumentorImpl::shouldInstrumentFunction(Function *Fn) {
  if (!Fn || Fn->isDeclaration())
    return false;
  return !Fn->getName().starts_with(IC.Base.RuntimeName);
}

bool InstrumentorImpl::shouldInstrumentGlobalVariable(GlobalVariable *GV) {
  if (!GV || GV->hasGlobalUnnamedAddr())
    return false;
  return !GV->getName().starts_with("llvm.");
}

bool InstrumentorImpl::shouldInstrumentAccess(Value *Ptr, Instruction *MI) {
  if (!Ptr || Ptr == NullPtrVal)
    return false;

  // If the pointer is used by a memory instruction, check whether that
  // instruction should be skipped (e.g., a memory instruction has a constant
  // offset access).
  if (MI && SkippedMemInsts.contains(MI))
    return false;

  // Some global variables could be skipped also.
  Value *UO = getUnderlyingObject(Ptr, 10);
  if (auto *GV = dyn_cast<GlobalVariable>(UO)) {
    if (SkippedGlobals.contains(GV))
      return false;
  } else if (auto *LI = dyn_cast<LoadInst>(UO)) {
    if (auto *GV = dyn_cast<GlobalVariable>(LI->getPointerOperand()))
      if (SkippedGlobals.contains(GV))
        return false;
  }
  return true;
}

CallInst *InstrumentorImpl::getCall(StringRef Name, RTArgumentPack &RTArgPack,
                                    InstrumentorConfig::Position P) {
  assert(P != InstrumentorConfig::PRE_AND_POST);

  bool UsesIndirection = false, DirectReturn = false;
  Type *RetTy = VoidTy;
  SmallVector<Type *> RTArgTypes;
  SmallVector<Value *> CallArgs;
  for (auto &RTA : RTArgPack.getArgs()) {
    bool PreOnly = (RTA.Kind & InstrumentorKindTy::PRE_ONLY);
    bool PostOnly = (RTA.Kind & InstrumentorKindTy::POST_ONLY);
    if (((PostOnly && (P == InstrumentorConfig::PRE)) ||
         (PreOnly && (P == InstrumentorConfig::POST))))
      continue;
    RTArgTypes.push_back(RTA.Ty);
    CallArgs.push_back(RTA.V);
    if (RTA.Kind & InstrumentorKindTy::POTENTIALLY_INDIRECT)
      UsesIndirection |= RTA.Ty->isPointerTy();
    if ((RTA.Kind & InstrumentorKindTy::REPLACABLE_PRE) &&
        (P == InstrumentorConfig::PRE)) {
      if (RetTy != VoidTy)
        UsesIndirection = true;
      if (!(RTA.Kind & InstrumentorKindTy::POTENTIALLY_INDIRECT)) {
        assert(!DirectReturn);
        DirectReturn = true;
      }
      RetTy = RTA.Ty;
    } else if ((RTA.Kind & InstrumentorKindTy::REPLACABLE_POST) &&
               (P == InstrumentorConfig::POST)) {
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

  std::string CompleteName =
      getRTName((P == InstrumentorConfig::POST) ? "post_" : "pre_", Name,
                UsesIndirection ? "_ind" : "");
  FunctionCallee &FC = InstrumentorCallees[CompleteName];

  if (!FC.getFunctionType())
    FC = M.getOrInsertFunction(
        CompleteName,
        FunctionType::get(RetTy, RTArgTypes, /*IsVarArgs*/ false));

  return IRB.CreateCall(FC, CallArgs);
}

bool InstrumentorImpl::instrumentAlloca(AllocaInst &I,
                                        InstrumentorConfig::Position P) {
  if (!IC.alloca.isEnabled(P))
    return false;
  if (IC.alloca.CB && !IC.alloca.CB(I))
    return false;

  setInsertPoint(I, P);

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

  RTArgumentPack RTArgPack(*this);
  RTArgPack.addVal(IC.alloca.Value, &I, P);
  RTArgPack.addValCB(IC.alloca.AllocationSize, CalculateAllocaSize, P);
  RTArgPack.addCI(IC.alloca.Alignment, I.getAlign().value(), P);

  auto *CI = getCall(IC.alloca.SectionName, RTArgPack, P);
  if (IC.alloca.ReplaceValue) {
    IRB.SetInsertPoint(CI->getNextNonDebugInstruction());
    I.replaceUsesWithIf(tryToCast(IRB, CI, I.getType(), DL), [&](Use &U) {
      if (NewInsts.lookup(cast<Instruction>(U.getUser())) == Epoche)
        return false;
      if (auto *LI = dyn_cast<LoadInst>(U.getUser()))
        return !SkippedMemInsts.contains(LI);
      if (auto *SI = dyn_cast<StoreInst>(U.getUser()))
        return SI->getPointerOperand() != &I ||
               !SkippedMemInsts.contains(SI);
      return true;
    });

    // This generates a base pointer.
    instrumentBasePointer(*CI);
  }

  return true;
}

bool InstrumentorImpl::instrumentCall(CallBase &I,
                                      InstrumentorConfig::Position P) {
  bool Changed = false;

  if (IC.allocation_call.isEnabled(P)) {
    auto &TLI = GetTLI(*I.getFunction());
    auto ACI = getAllocationCallInfo(&I, &TLI);

    if (ACI)
      Changed |= instrumentAllocationCall(I, *ACI, P);
  }

  switch (I.getIntrinsicID()) {
  case Intrinsic::memcpy:
  case Intrinsic::memcpy_element_unordered_atomic:
  case Intrinsic::memcpy_inline:
  case Intrinsic::memmove:
  case Intrinsic::memmove_element_unordered_atomic:
  case Intrinsic::memset:
  case Intrinsic::memset_element_unordered_atomic:
  case Intrinsic::memset_inline:
    Changed |= instrumentMemoryIntrinsic(cast<IntrinsicInst>(I), P);
    break;
  case Intrinsic::trap:
  case Intrinsic::debugtrap:
  case Intrinsic::ubsantrap:
    Changed |= instrumentGeneralIntrinsic(cast<IntrinsicInst>(I), P);
    break;
  case Intrinsic::not_intrinsic:
    Changed |= instrumentGenericCall(I, P);
  default:
    break;
  }

  return Changed;
}

bool InstrumentorImpl::instrumentGenericCall(CallBase &I,
                                             InstrumentorConfig::Position P) {
  if (!IC.call.isEnabled(P))
    return false;

  Function *CalledFn = I.getCalledFunction();
  if (IC.call.SkipInstrumentedLocally)
    // TODO: This will not work for indirect calls.
    if (shouldInstrumentFunction(CalledFn))
      return false;

  if (IC.call.SkipInstrumentedExternally)
    // TODO: This will not work for indirect calls.
    if (CalledFn && ExternalInstrFunctions.contains(CalledFn->getName()))
      return false;

  SmallVector<Value *> CallArgs;
  SmallVector<uint32_t> CallArgsPos;
  for (auto [Idx, AU] : enumerate(I.args()))
    if (!IC.call.OnlyReplacedPointerValues || (AU.get()->getType() == PtrTy &&
        shouldInstrumentAccess(AU.get()))) {
      CallArgs.push_back(AU.get());
      CallArgsPos.push_back(Idx);
  }

  setInsertPoint(I, P);

  RTArgumentPack RTArgPack(*this);
  SmallVector<Type *> ArgTypes;
  SmallVector<Value *> ValueGEPs;
  for (Value *AV : CallArgs)
    ArgTypes.push_back(AV->getType());

  RTArgPack.addVal(IC.call.CalleePtr, I.getCalledOperand(), P);
  RTArgPack.addValCB(
      IC.call.CalleeName,
      [&](Type *) {
        return getGlobalString(CalledFn ? CalledFn->getName() : "<unknown>");
      },
      P);

  if (IC.call.Values.isEnabled(P)) {
    if (ArgTypes.empty()) {
      RTArgPack.addVal(IC.call.Values, NullPtrVal, P);
    } else {
      StructType *ArgStructTy = StructType::create(ArgTypes, "argstype");
      auto *ArgStructAI = getAlloca(I.getFunction(), ArgStructTy);
      for (auto [Idx, AV] : enumerate(CallArgs)) {
        ValueGEPs.push_back(IRB.CreateStructGEP(ArgStructTy, ArgStructAI, Idx));
        IRB.CreateStore(AV, ValueGEPs.back());
      }
      RTArgPack.addVal(IC.call.Values, ArgStructAI, P, ArgStructAI);
    }
  }

  if (IC.call.ValueSizes.isEnabled(P)) {
    if (ArgTypes.empty()) {
      RTArgPack.addVal(IC.call.ValueSizes, NullPtrVal, P);
    } else {
      ArrayType *SizesTy = ArrayType::get(Int32Ty, ArgTypes.size());
      auto *SizesAI = getAlloca(I.getFunction(), SizesTy);
      for (auto [Idx, ATy] : enumerate(ArgTypes)) {
        IRB.CreateStore(getCI(Int32Ty, DL.getTypeStoreSize(ATy)),
                        IRB.CreateConstGEP2_32(SizesTy, SizesAI, 0, Idx));
      }
      RTArgPack.addVal(IC.call.ValueSizes, SizesAI, P, SizesAI);
    }
  }

  if (IC.call.ValueTypeIds.isEnabled(P)) {
    if (ArgTypes.empty()) {
      RTArgPack.addVal(IC.call.ValueTypeIds, NullPtrVal, P);
    } else {
      ArrayType *TypeIdsTy = ArrayType::get(Int32Ty, ArgTypes.size());
      auto *TypeIdsAI = getAlloca(I.getFunction(), TypeIdsTy);
      for (auto [Idx, ATy] : enumerate(ArgTypes))
        IRB.CreateStore(getCI(Int32Ty, ATy->getTypeID()),
                        IRB.CreateConstGEP2_32(TypeIdsTy, TypeIdsAI, 0, Idx));
      RTArgPack.addVal(IC.call.ValueTypeIds, TypeIdsAI, P, TypeIdsAI);
    }
  }

  RTArgPack.addCI(IC.call.NumValues, ArgTypes.size(), P);

  auto *CI = getCall(IC.call.SectionName, RTArgPack, P);

  if (IC.call.ReplaceValues && IC.call.Values.isEnabled(P)) {
    for (auto [Idx, ATy] : enumerate(ArgTypes))
      I.setArgOperand(CallArgsPos[Idx], IRB.CreateLoad(ATy, ValueGEPs[Idx]));
  }

  return true;
}

bool InstrumentorImpl::instrumentAllocationCall(
    CallBase &I, const AllocationCallInfo &ACI,
    InstrumentorConfig::Position P) {
  if (!IC.allocation_call.isEnabled(P))
    return false;
  if (IC.allocation_call.CB && !IC.allocation_call.CB(I))
    return false;

  setInsertPoint(I, P);

  RTArgumentPack RTArgPack(*this);
  RTArgPack.addVal(IC.allocation_call.MemoryPointer, &I, P);

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
  RTArgPack.addValCB(IC.allocation_call.MemorySize, GetMemorySize, P);
  RTArgPack.addVal(IC.allocation_call.Alignment, ACI.Alignment, P);
  RTArgPack.addValCB(
      IC.allocation_call.Family,
      [&](Type *) { return getGlobalString(ACI.Family.value_or("")); }, P);

  auto GetInitializerKind = [&](Type *) {
    if (!ACI.InitialValue)
      return getCI(Int8Ty, -1);
    if (ACI.InitialValue->isZeroValue())
      return getCI(Int8Ty, 0);
    if (isa<UndefValue>(ACI.InitialValue))
      return getCI(Int8Ty, 1);
    return getCI(Int8Ty, 2);
  };
  RTArgPack.addValCB(IC.allocation_call.InitializerKind, GetInitializerKind, P);

  auto *CI = getCall(IC.allocation_call.SectionName, RTArgPack, P);
  if (IC.allocation_call.ReplaceValue) {
    IRB.SetInsertPoint(CI->getNextNonDebugInstruction());
    I.replaceUsesWithIf(tryToCast(IRB, CI, I.getType(), DL), [&](Use &U) {
      return NewInsts.lookup(cast<Instruction>(U.getUser())) != Epoche;
    });

    // This generates a base pointer.
    instrumentBasePointer(*CI);
  }

  return true;
}

bool InstrumentorImpl::instrumentMemoryIntrinsic(
    IntrinsicInst &I, InstrumentorConfig::Position P) {
  if (!IC.memory_intrinsic.isEnabled(P))
    return false;
  if (IC.memory_intrinsic.CB && !IC.memory_intrinsic.CB(I))
    return false;

  setInsertPoint(I, P);

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

  RTArgumentPack RTArgPack(*this);
  RTArgPack.addCI(IC.memory_intrinsic.KindId, KindId, P);
  RTArgPack.addVal(IC.memory_intrinsic.DestinationPointer, DestPtr, P);
  RTArgPack.addCI(IC.memory_intrinsic.DestinationPointerAddressSpace,
        DestPtr->getType()->getPointerAddressSpace(), P);
  RTArgPack.addVal(IC.memory_intrinsic.SourcePointer,
         SrcPtr ? SrcPtr : Constant::getAllOnesValue(PtrTy), P);
  RTArgPack.addCI(IC.memory_intrinsic.SourcePointerAddressSpace,
        SrcPtr ? SrcPtr->getType()->getPointerAddressSpace() : -1, P);

  RTArgPack.addVal(IC.memory_intrinsic.MemsetValue,
         MemsetValue ? MemsetValue
                     : Constant::getAllOnesValue(
                           IC.memory_intrinsic.MemsetValue.getType(Ctx)),
         P);
  RTArgPack.addVal(IC.memory_intrinsic.Length, Length, P);
  RTArgPack.addCI(IC.memory_intrinsic.IsVolatile, I.isVolatile(), P);
  RTArgPack.addCI(IC.memory_intrinsic.AtomicElementSize, AtomicElementSize, P);

  getCall(IC.memory_intrinsic.SectionName, RTArgPack, P);
  return true;
}

bool InstrumentorImpl::instrumentGeneralIntrinsic(
    IntrinsicInst &I, InstrumentorConfig::Position P) {
  if (!IC.intrinsic.isEnabled(P))
    return false;
  if (IC.intrinsic.CB && !IC.intrinsic.CB(I))
    return false;

  setInsertPoint(I, P);

  RTArgumentPack RTArgPack(*this);
  RTArgPack.addCI(IC.intrinsic.KindId, I.getIntrinsicID(), P);

  getCall(IC.intrinsic.SectionName, RTArgPack, P);

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

  SkippedMemInsts.insert(&I);

  return true;
}

bool InstrumentorImpl::instrumentStore(StoreInst &I,
                                       InstrumentorConfig::Position P) {
  if (!IC.store.isEnabled(P))
    return false;
  if (IC.store.CB && !IC.store.CB(I))
    return false;
  if (!shouldInstrumentAccess(I.getPointerOperand(), &I))
    return false;

  setInsertPoint(I, P);

  RTArgumentPack RTArgPack(*this);
  RTArgPack.addVal(IC.store.PointerOperand, I.getPointerOperand(), P);
  RTArgPack.addCI(IC.store.PointerOperandAddressSpace, I.getPointerAddressSpace(),
        P);
  RTArgPack.addIndVal(IC.store.ValueOperand, I.getValueOperand(), I.getFunction(),
            P);
  RTArgPack.addCI(IC.store.ValueOperandSize,
        DL.getTypeStoreSize(I.getValueOperand()->getType()), P);
  RTArgPack.addCI(IC.store.ValueOperandTypeId,
        I.getValueOperand()->getType()->getTypeID(), P);
  RTArgPack.addCI(IC.store.Alignment, I.getAlign().value(), P);
  RTArgPack.addCI(IC.store.AtomicityOrdering, uint64_t(I.getOrdering()), P);
  RTArgPack.addCI(IC.store.SyncScopeId, uint64_t(I.getSyncScopeID()), P);
  RTArgPack.addCI(IC.store.IsVolatile, I.isVolatile(), P);

  RTArgPack.addValCB(
      IC.store.BasePointerInfo,
      [&](Type *Ty) -> Value * {
        return findBasePointer(I.getPointerOperand());
      },
      P);

  auto *CI = getCall(IC.store.SectionName, RTArgPack, P);
  if (IC.store.ReplacePointerOperand)
    I.setOperand(I.getPointerOperandIndex(), CI);

  return true;
}

bool InstrumentorImpl::instrumentLoad(LoadInst &I,
                                      InstrumentorConfig::Position P) {
  if (!IC.load.isEnabled(P))
    return false;
  if (IC.load.CB && !IC.load.CB(I))
    return false;
  if (!shouldInstrumentAccess(I.getPointerOperand(), &I))
    return false;

  setInsertPoint(I, P);

  RTArgumentPack RTArgPack(*this);
  RTArgPack.addVal(IC.load.PointerOperand, I.getPointerOperand(), P);
  RTArgPack.addCI(IC.load.PointerOperandAddressSpace, I.getPointerAddressSpace(),
        P);

  AllocaInst *IndirectionAI =
      RTArgPack.addIndVal(IC.load.Value, &I, I.getFunction(), P,
                /*ForceIndirection=*/false);

  RTArgPack.addCI(IC.load.ValueSize, DL.getTypeStoreSize(I.getType()), P);
  RTArgPack.addCI(IC.load.ValueTypeId, I.getType()->getTypeID(), P);
  RTArgPack.addCI(IC.load.Alignment, I.getAlign().value(), P);
  RTArgPack.addCI(IC.load.AtomicityOrdering, uint64_t(I.getOrdering()), P);
  RTArgPack.addCI(IC.load.SyncScopeId, uint64_t(I.getSyncScopeID()), P);
  RTArgPack.addCI(IC.load.IsVolatile, I.isVolatile(), P);

  RTArgPack.addValCB(
      IC.store.BasePointerInfo,
      [&](Type *Ty) -> Value * {
        return findBasePointer(I.getPointerOperand());
      },
      P);

  bool ReplaceValue = IC.load.ReplaceValue && (P == InstrumentorConfig::POST);
  bool ReplacePointerOperand =
      IC.load.ReplacePointerOperand && (P == InstrumentorConfig::PRE);

  auto *CI = getCall(IC.load.SectionName, RTArgPack, P);
  if (ReplaceValue) {
    IRB.SetInsertPoint(CI->getNextNonDebugInstruction());
    Value *NewV = IndirectionAI ? IRB.CreateLoad(I.getType(), IndirectionAI)
                                : tryToCast(IRB, CI, I.getType(), DL);
    if (!NewV)
      I.getFunction()->dump();
    assert(NewV);
    I.replaceUsesWithIf(NewV, [&](Use &U) {
      return NewInsts.lookup(cast<Instruction>(U.getUser())) != Epoche;
    });
  } else if (ReplacePointerOperand) {
    I.setOperand(I.getPointerOperandIndex(), CI);
  }

  return true;
}

bool InstrumentorImpl::instrumentUnreachable(UnreachableInst &I,
                                             InstrumentorConfig::Position P) {
  if (!IC.unreachable.isEnabled(P))
    return false;
  if (IC.unreachable.CB && !IC.unreachable.CB(I))
    return false;

  setInsertPoint(I, P);

  RTArgumentPack RTArgPack(*this);
  getCall(IC.unreachable.SectionName, RTArgPack, P);

  return true;
}

bool InstrumentorImpl::instrumentFunction(Function &Fn) {
  bool Changed = false;
  if (!shouldInstrumentFunction(&Fn))
    return Changed;

  Functions.push_back(&Fn);

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

  for (auto &Arg : Fn.args())
    if (Arg.getType() == PtrTy)
      Changed |= instrumentBasePointer(Arg);

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
      case Instruction::IntToPtr:
        if (I.getType() == PtrTy)
          Changed |= instrumentBasePointer(I);
        break;
      default:
        break;
      }

      switch (I.getOpcode()) {
      case Instruction::Alloca:
        Changed |=
            instrumentAlloca(cast<AllocaInst>(I), InstrumentorConfig::PRE);
        Changed |=
            instrumentAlloca(cast<AllocaInst>(I), InstrumentorConfig::POST);
        break;
      case Instruction::Call:
        Changed |= instrumentCall(cast<CallBase>(I), InstrumentorConfig::PRE);
        Changed |= instrumentCall(cast<CallBase>(I), InstrumentorConfig::POST);
        break;
      case Instruction::Load:
        Changed |= instrumentLoad(cast<LoadInst>(I), InstrumentorConfig::PRE);
        Changed |= instrumentLoad(cast<LoadInst>(I), InstrumentorConfig::POST);
        break;
      case Instruction::Store:
        Changed |= instrumentStore(cast<StoreInst>(I), InstrumentorConfig::PRE);
        Changed |=
            instrumentStore(cast<StoreInst>(I), InstrumentorConfig::POST);
        break;
      case Instruction::Unreachable:
        Changed |= instrumentUnreachable(cast<UnreachableInst>(I),
                                         InstrumentorConfig::PRE);
        Changed |= instrumentUnreachable(cast<UnreachableInst>(I),
                                         InstrumentorConfig::POST);
        break;
      default:
        break;
      }
    }
  }

  return Changed;
}

bool InstrumentorImpl::instrumentMainFunction(Function &MainFn) {
  auto AvailP = InstrumentorConfig::PRE_AND_POST;
  if (!IC.main_function.isEnabled(AvailP))
    return false;
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

  RTArgumentPack RTArgPack(*this);
  AllocaInst *IndirectionAIArgC = nullptr, *IndirectionAIArgV = nullptr;
  IndirectionAIArgC = RTArgPack.addIndVal(
      IC.main_function.ArgC,
      InstMainFn->arg_size() ? cast<Value>(InstMainFn->arg_begin())
                             : getCI(IC.main_function.ArgC.getType(Ctx), 0),
      InstMainFn, AvailP,
      /*ForceIndirection=*/IC.main_function.ReplaceArgumentValues);
  IndirectionAIArgV = RTArgPack.addIndVal(
      IC.main_function.ArgV,
      InstMainFn->arg_size() > 1 ? cast<Value>(InstMainFn->arg_begin() + 1)
                                 : NullPtrVal,
      InstMainFn, AvailP,
      /*ForceIndirection=*/IC.main_function.ReplaceArgumentValues);

  if (IC.main_function.isEnabled(InstrumentorConfig::PRE))
    getCall(IC.main_function.SectionName, RTArgPack, InstrumentorConfig::PRE);

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

  if (IC.main_function.isEnabled(InstrumentorConfig::POST)) {
    RTArgPack.addVal(IC.main_function.ReturnValue, ReturnValue, AvailP);

    auto *CI =
        getCall(IC.main_function.SectionName, RTArgPack, InstrumentorConfig::POST);
    if (IC.main_function.ReplaceReturnValue)
      ReturnValue = CI;
  }
  IRB.CreateRet(ReturnValue);

  return true;
}

bool InstrumentorImpl::instrumentModule(InstrumentorConfig::Position P) {
  assert(P != InstrumentorConfig::PRE_AND_POST);
  if (!IC.module.isEnabled(P))
    return false;

  Function *YtorFn = (P & InstrumentorConfig::POST) ? DtorFn : CtorFn;
  assert(YtorFn);

  IRB.SetInsertPointPastAllocas(YtorFn);

  RTArgumentPack RTArgPack(*this);
  RTArgPack.addValCB(
      IC.module.ModuleName,
      [&](Type *) { return getGlobalString(M.getName()); }, P);
  RTArgPack.addValCB(
      IC.module.ModuleTargetTriple,
      [&](Type *) { return getGlobalString(M.getTargetTriple()); }, P);

  getCall(IC.module.SectionName, RTArgPack, P);

  return true;
}

bool InstrumentorImpl::prepareGlobalVariables() {
  bool Changed = false;
  if (!IC.global_var.isEnabled(InstrumentorConfig::PRE_AND_POST))
    return Changed;

  for (GlobalVariable &GV : M.globals()) {
    if (!shouldInstrumentGlobalVariable(&GV)) {
      SkippedGlobals.insert(&GV);
    } else if (GV.isDeclaration() && ExternalInstrGlobals.contains(GV.getName())) {
      Globals.push_back(&GV);
    } else if (GV.isDeclaration() && IC.global_var.SkipDeclaration) {
      SkippedGlobals.insert(&GV);
    } else {
      Globals.push_back(&GV);
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
  auto AvailP = InstrumentorConfig::POST;
  if (!IC.global_var.isEnabled(AvailP))
    return false;
  if (Globals.empty())
    return false;

  IRB.SetInsertPointPastAllocas(CtorFn);

  SmallVector<GlobalVariable *> InstrGlobals;
  for (GlobalVariable *GV : Globals) {
    GlobalVariable *InstrGV =
        new GlobalVariable(M, PtrTy, false, GlobalValue::ExternalLinkage,
                           GV->isDeclaration() ? nullptr : NullPtrVal,
                           getRTName("", GV->getName()));

    InstrGlobals.push_back(InstrGV);

    // Do not register the global as another module will do it.
    if (GV->isDeclaration())
      continue;

    RTArgumentPack RTArgPack(*this);
    RTArgPack.addVal(IC.global_var.Value, GV, AvailP);
    RTArgPack.addCI(IC.global_var.Size, DL.getTypeAllocSize(GV->getValueType()),
          AvailP);
    RTArgPack.addCI(IC.global_var.Alignment, GV->getAlign().valueOrOne().value(),
          AvailP);
    RTArgPack.addCI(IC.global_var.IsConstant, GV->isConstant(), AvailP);
    RTArgPack.addCI(IC.global_var.UnnamedAddress, int64_t(GV->getUnnamedAddr()),
          AvailP);
    RTArgPack.addValCB(
        IC.global_var.Name,
        [&](Type *) { return getGlobalString(GV->getName()); }, AvailP);
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
    RTArgPack.addValCB(IC.global_var.InitializerKind, GetInitializerKind, AvailP);

    auto *CI = getCall(IC.global_var.SectionName, RTArgPack, AvailP);
    if (IC.global_var.ReplaceValue)
      IRB.CreateStore(CI, InstrGV);
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
      Instruction *UserI = cast<Instruction>(U->getUser());
      IRB.SetInsertPoint(UserI);
      auto *LoadPtr = IRB.CreateLoad(PtrTy, InstrGlobals[G]);
      U->set(LoadPtr);
    }
  }

  return true;
}

bool InstrumentorImpl::instrumentBasePointer(Value &ArgOrInst) {
  auto P = InstrumentorConfig::POST;
  if (!IC.base_pointer.isEnabled(P))
    return false;
  if (auto *Arg = dyn_cast<Argument>(&ArgOrInst)) {
    IRB.SetInsertPointPastAllocas(Arg->getParent());
  } else if (auto *I = dyn_cast<Instruction>(&ArgOrInst)) {
    setInsertPoint(*I, P);
  }

  RTArgumentPack RTArgPack(*this);
  RTArgPack.addVal(IC.base_pointer.Value, &ArgOrInst, P);

  auto *CI = getCall(IC.base_pointer.SectionName, RTArgPack, P);
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
  if (!IC.base_pointer.isEnabled(InstrumentorConfig::PRE_AND_POST))
    return NullPtrVal;

  Value *UO = getUnderlyingObject(V, 10);

  auto It = BasePtrMap.find(UO);
  if (It != BasePtrMap.end())
    return It->second;

  // It must be a global variable. Otherwise, there are pointer
  // originators that are not instrumented yet.
  assert(isa<GlobalVariable>(UO));
  return NullPtrVal;
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

  if (IC.module.isEnabled(InstrumentorConfig::PRE) ||
      IC.global_var.isEnabled(InstrumentorConfig::PRE_AND_POST))
    addCtorOrDtor(/*Ctor=*/true);
  if (IC.module.isEnabled(InstrumentorConfig::POST))
    addCtorOrDtor(/*Ctor=*/false);

  Changed |= instrumentModule(InstrumentorConfig::PRE);
  Changed |= instrumentModule(InstrumentorConfig::POST);

  Changed |= prepareGlobalVariables();

  for (Function &Fn : M) {
    Changed |= instrumentFunction(Fn);

    if (Fn.getName() == "main")
      MainFn = &Fn;
  }

  Changed |= instrumentGlobalVariables();

  if (MainFn)
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

  Impl.loadInstrumentorDatabase();

  if (!Impl.instrument())
    return PreservedAnalyses::all();

  if (verifyModule(M))
    M.dump();
  assert(!verifyModule(M, &errs()));

  Impl.updateInstrumentorDatabase();

  return PreservedAnalyses::none();
}
