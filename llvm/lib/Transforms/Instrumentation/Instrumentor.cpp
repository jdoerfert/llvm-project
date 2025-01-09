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
#include "llvm/IR/InstIterator.h"
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

#define SECTION_START(SECTION, CLASS)                                          \
  J.attributeBegin(#SECTION);                                                  \
  J.objectBegin();
#define CONFIG_INTERNAL(SECTION, TYPE, NAME, DEFAULT_VALUE)
#define CONFIG(SECTION, TYPE, NAME, DEFAULT_VALUE)                             \
  J.attribute(#NAME, IC.SECTION.NAME);
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

#define CONFIG(SECTION, TYPE, NAME, DEFAULT_VALUE)                             \
  It = Config->find(#SECTION);                                                 \
  if (It != End) {                                                             \
    if (auto *InstObj = It->second.getAsObject()) {                            \
      if (auto *Val = InstObj->get(#NAME)) {                                   \
        if (!json::fromJSON(*Val, IC.SECTION.NAME, NullRoot))                  \
          errs() << "WARNING: Failed to read " #SECTION "." #NAME " as " #TYPE \
                 << "\n";                                                      \
      }                                                                        \
    }                                                                          \
  }

#define SECTION_START(SECTION, CLASS)
#define CONFIG_INTERNAL(SECTION, TYPE, NAME, DEFAULT_VALUE)
#define SECTION_END(SECTION)

#include "llvm/Transforms/Instrumentation/InstrumentorConfig.def"

  return true;
}

raw_ostream &printAsCType(raw_ostream &OS, Type *T) {
  if (T->isPointerTy())
    return OS << "void* ";
  if (T->isIntegerTy())
    return OS << "int" << T->getIntegerBitWidth() << "_t ";
  return OS << *T << " ";
}
bool printAsPrintfFormat(raw_ostream &OS, Type *T) {
  if (T->isPointerTy())
    return OS << "%p", true;
  if (T->isIntegerTy()) {
    if (T->getIntegerBitWidth() <= 32)
      return OS << "%i", true;
    if (T->getIntegerBitWidth() <= 64)
      return OS << "%li", true;
  }
  if (T->isFloatTy())
    return OS << "%f", true;
  if (T->isDoubleTy())
    return OS << "%lf", true;
  return false;
}

template <typename IRBTy>
Value *tryToCast(IRBTy &IRB, Value *V, Type *Ty, const DataLayout &DL) {
  if (!V)
    return Constant::getAllOnesValue(Ty);
  auto *VTy = V->getType();
  if (VTy == Ty)
    return V;
  if (CastInst::isBitOrNoopPointerCastable(VTy, Ty, DL))
    return IRB.CreateBitOrPointerCast(V, Ty);
  if (VTy->isPointerTy() && Ty->isPointerTy())
    return IRB.CreatePointerBitCastOrAddrSpaceCast(V, Ty);
  if (VTy->isPointerTy() && Ty->isIntegerTy())
    return IRB.CreateIntToPtr(V, Ty);
  if (VTy->isIntegerTy() && Ty->isIntegerTy())
    return IRB.CreateIntCast(V, Ty, /* isSigned */ false);
  if (VTy->isFloatingPointTy() && Ty->isIntOrPtrTy()) {
    switch (DL.getTypeSizeInBits(VTy)) {
    case 64:
      V = IRB.CreateBitCast(V, IRB.getInt64Ty());
      break;
    case 32:
      V = IRB.CreateBitCast(V, IRB.getInt32Ty());
      break;
    case 16:
      V = IRB.CreateBitCast(V, IRB.getInt16Ty());
      break;
    case 8:
      V = IRB.CreateBitCast(V, IRB.getInt8Ty());
      break;
    default:
      return Constant::getAllOnesValue(Ty);
    }
    return tryToCast(IRB, V, Ty, DL);
  }
  return Constant::getAllOnesValue(Ty);
}

class InstrumentorImpl final {
public:
  InstrumentorImpl(const InstrumentorConfig &IC, Module &M,
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

  ~InstrumentorImpl() { delete StubRuntimeOut; }

  /// Instrument the module, public entry point.
  bool instrument();

private:
  bool shouldInstrumentFunction(Function *Fn);
  bool shouldInstrumentGlobalVariable(GlobalVariable *GV);
  bool hasAnnotation(Instruction *I, StringRef Annotation);

  bool instrumentFunction(Function &Fn);
  bool instrumentAlloca(AllocaInst &I);
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

  template <typename MemoryInstTy>
  bool analyzeAccess(MemoryInstTy &I);

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
                          Fn->getEntryBlock().getFirstInsertionPt());
    return AI;
  }

  /// Mapping to remember global strings passed to the runtime.
  DenseMap<StringRef, Value *> GlobalStringsMap;
  Value *getGlobalString(StringRef S) {
    Value *&V = GlobalStringsMap[S];
    if (!V) {
      Twine Name = Twine(IC.Base.RuntimeName) + "str";
      V = IRB.CreateGlobalString(S, Name, DL.getDefaultGlobalsAddressSpace(), &M);
    }
    return V;
  }

  template <typename Ty> Constant *getCI(Type *IT, Ty Val) {
    return ConstantInt::get(IT, Val);
  }

  std::string getRTName(StringRef Prefix, StringRef Position,
                        StringRef Suffix = "") {
    return (IC.Base.RuntimeName + Prefix + Position + Suffix).str();
  }

  raw_fd_ostream *StubRuntimeOut = nullptr;

  raw_fd_ostream *getStubRuntimeOut() {
    if (!StubRuntimeOut) {
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
      }
    }
    return StubRuntimeOut;
  }

  /// Map to remember instrumentation functions for a specific opcode and
  /// pre/post position.
  StringMap<FunctionCallee> InstrumentorCallees;

  FunctionCallee getCallee(StringRef Name, SmallVectorImpl<Type *> &RTArgTypes,
                           SmallVectorImpl<std::string> &RTArgNames, bool After,
                           bool Indirection, Type *RetTy = nullptr) {
    std::string CompleteName = getRTName(After ? "post_" : "pre_", Name, Indirection ? "_ind" : "");
    FunctionCallee &FC = InstrumentorCallees[CompleteName];

    if (!FC.getFunctionType()) {
      FC = M.getOrInsertFunction(CompleteName, FunctionType::get(RetTy ? RetTy : VoidTy,
          RTArgTypes, /*IsVarArgs*/ false));

      if (IC.Base.PrintRuntimeSignatures || !IC.Base.StubRuntimePath.empty()) {
        std::string Str;
        raw_string_ostream StrOut(Str);
        printAsCType(StrOut, FC.getFunctionType()->getReturnType());
        StrOut << FC.getCallee()->getName() << "(";
        auto *FT = FC.getFunctionType();
        for (int I = 0, E = RTArgNames.size(); I != E; ++I) {
          if (I != 0)
            StrOut << ", ";
          printAsCType(StrOut, FT->getParamType(I)) << RTArgNames[I];
        }
        StrOut << ")";
        if (IC.Base.PrintRuntimeSignatures)
          outs() << Str << ";\n";
        if (!IC.Base.StubRuntimePath.empty()) {
          if (auto *SROut = getStubRuntimeOut()) {
            (*SROut) << Str << " {\n";
            Str.clear();
            (*SROut) << "  printf(\"" << CompleteName << " - ";
            for (int I = 0, E = RTArgNames.size(); I != E; ++I) {
              (*SROut) << RTArgNames[I] << ": ";
              if (printAsPrintfFormat(*SROut, RTArgTypes[I])) {
                if (!Str.empty())
                  StrOut << ", ";
                StrOut << RTArgNames[I];
              } else {
                (*SROut) << "<unknown>";
              }
              (*SROut) << " - ";
            }
            (*SROut) << "\\n\", " << Str << ");\n";
            if (RetTy) {
              if (RetTy == RTArgTypes[0])
                (*SROut) << "  return " << RTArgNames[0] << ";\n";
              else
                (*SROut) << "  return 0;\n";
            }
            (*SROut) << "}\n";
          }
        }
      }
    }
    return FC;
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
  const InstrumentorConfig &IC;

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

bool InstrumentorImpl::instrumentAlloca(AllocaInst &I) {
  if (IC.Alloca.CB && !IC.Alloca.CB(I))
    return false;

  Instruction *IP = I.getNextNonDebugInstruction();
  while (isa<AllocaInst>(IP))
    IP = IP->getNextNonDebugInstruction();
  IRB.SetInsertPoint(IP);

  SmallVector<Type *> RTArgTypes;
  SmallVector<Value *> RTArgs;
  SmallVector<std::string> RTArgNames;

  if (IC.Alloca.Value) {
    auto *ArgTy = PtrTy;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(tryToCast(IRB, &I, ArgTy, DL));
    RTArgNames.push_back("Value");
  }

  if (IC.Alloca.AllocationSize) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    Value *SizeValue = nullptr;
    TypeSize TypeSize = DL.getTypeAllocSize(I.getAllocatedType());
    if (TypeSize.isFixed())
      SizeValue = getCI(ArgTy, TypeSize.getFixedValue());
    if (!SizeValue) {
      SizeValue = IRB.CreateSub(
          IRB.CreatePtrToInt(
              IRB.CreateGEP(I.getAllocatedType(), &I, {getCI(Int32Ty, 1)}),
              ArgTy),
          IRB.CreatePtrToInt(&I, ArgTy));
    }
    if (I.isArrayAllocation())
      SizeValue = IRB.CreateMul(
          SizeValue, IRB.CreateZExtOrBitCast(I.getArraySize(), ArgTy));
    RTArgs.push_back(SizeValue);
    RTArgNames.push_back("AllocationSize");
  }

  if (IC.Alloca.Alignment) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.getAlign().value()));
    RTArgNames.push_back("Alignment");
  }

  Type *RetTy = IC.Alloca.ReplaceValue ? PtrTy : nullptr;
  FunctionCallee FC = getCallee("alloca", RTArgTypes, RTArgNames, /*After=*/true,
                                /*Indirection=*/false, RetTy);
  auto *CI = IRB.CreateCall(FC, RTArgs);
  if (IC.Alloca.ReplaceValue) {
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
  auto &TLI = GetTLI(*I.getFunction());
  auto ACI = getAllocationCallInfo(&I, &TLI);

  if (ACI && IC.AllocationCall.Instrument)
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
      if (IC.MemoryIntrinsic.Instrument)
        return instrumentMemoryIntrinsic(*II);
      break;
    case Intrinsic::trap:
    case Intrinsic::debugtrap:
    case Intrinsic::ubsantrap:
      if (IC.GeneralIntrinsic.Instrument)
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

  SmallVector <Use *> InstrUses;

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

  SmallVector<Type *> RTArgTypes;
  SmallVector<std::string> RTArgNames;

  if (IC.CallArg.Value) {
    auto *ArgTy = PtrTy;
    RTArgTypes.push_back(ArgTy);
    RTArgNames.push_back("Value");
  }

  Type *RetTy = IC.CallArg.ReplaceValue ? PtrTy : nullptr;

  for (Use *U : InstrUses) {
    SmallVector<Value *> RTArgs;
    if (IC.CallArg.Value)
      RTArgs.push_back(U->get());

    FunctionCallee FC =
        getCallee("call_arg", RTArgTypes, RTArgNames, /*After=*/false,
                  /*Indirection=*/false, RetTy);

    Value *Ret = IRB.CreateCall(FC, RTArgs);
    if (RetTy)
      U->set(Ret);
  }

  return true;
}

bool InstrumentorImpl::instrumentAllocationCall(CallBase &I,
                                                const AllocationCallInfo &ACI) {
  if (IC.AllocationCall.CB && !IC.AllocationCall.CB(I))
    return false;

  Instruction *IP = I.getNextNonDebugInstruction();
  IRB.SetInsertPoint(IP);

  SmallVector<Type *> RTArgTypes;
  SmallVector<Value *> RTArgs;
  SmallVector<std::string> RTArgNames;

  if (IC.AllocationCall.MemoryPointer) {
    auto *ArgTy = PtrTy;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(tryToCast(IRB, &I, ArgTy, DL));
    RTArgNames.push_back("MemoryPointer");
  }

  if (IC.AllocationCall.MemorySize) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    auto *LHS = tryToCast(IRB, ACI.SizeLHS, ArgTy, DL);
    auto *RHS = tryToCast(IRB, ACI.SizeRHS, ArgTy, DL);
    bool LHSIsUnknown =
        isa<ConstantInt>(LHS) && cast<ConstantInt>(LHS)->isAllOnesValue();
    bool RHSIsUnknown =
        isa<ConstantInt>(RHS) && cast<ConstantInt>(RHS)->isAllOnesValue();
    Value *Size = nullptr;
    if (!LHSIsUnknown && !RHSIsUnknown)
      Size = IRB.CreateMul(LHS, RHS);
    else if (LHSIsUnknown)
      Size = RHS;
    else
      Size = LHS;
    RTArgs.push_back(Size);
    RTArgNames.push_back("MemorySize");
  }

  if (IC.AllocationCall.Alignment) {
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(tryToCast(IRB, ACI.Alignment, ArgTy, DL));
    RTArgNames.push_back("Alignment");
  }

  if (IC.AllocationCall.Family) {
    auto *ArgTy = PtrTy;
    RTArgTypes.push_back(ArgTy);
    StringRef Family = ACI.Family.value_or("");
    if (Family.empty())
      RTArgs.push_back(Constant::getNullValue(ArgTy));
    else
      RTArgs.push_back(getGlobalString(Family));
    RTArgNames.push_back("Family");
  }

  if (IC.AllocationCall.InitialValue) {
    auto *ArgTy = Int8Ty;
    RTArgTypes.push_back(ArgTy);
    if (!ACI.InitialValue)
      RTArgs.push_back(getCI(ArgTy, -1));
    else if (isa<UndefValue>(ACI.InitialValue))
      RTArgs.push_back(getCI(ArgTy, 1));
    else if (ACI.InitialValue->isZeroValue())
      RTArgs.push_back(getCI(ArgTy, 0));
    else
      RTArgs.push_back(getCI(ArgTy, -1));
    RTArgNames.push_back("InitialValue");
  }

  Type *RetTy = IC.AllocationCall.ReplaceValue ? PtrTy : nullptr;
  FunctionCallee FC =
      getCallee("allocation_call", RTArgTypes, RTArgNames, /*After=*/true,
                /*Indirection=*/false, RetTy);
  auto *CI = IRB.CreateCall(FC, RTArgs);
  if (IC.AllocationCall.ReplaceValue) {
    IRB.SetInsertPoint(CI->getNextNonDebugInstruction());
    I.replaceUsesWithIf(tryToCast(IRB, CI, I.getType(), DL), [&](Use &U) {
      return NewInsts.lookup(cast<Instruction>(U.getUser())) != Epoche;
    });
  }

  return true;
}

bool InstrumentorImpl::instrumentMemoryIntrinsic(IntrinsicInst &I) {
  if (IC.MemoryIntrinsic.CB && !IC.MemoryIntrinsic.CB(I))
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

  AllocaInst *IndirectionAI = nullptr;
  SmallVector<Type *> RTArgTypes;
  SmallVector<Value *> RTArgs;
  SmallVector<std::string> RTArgNames;

  if (IC.MemoryIntrinsic.KindId) {
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, KindId));
    RTArgNames.push_back("KindId");
  }

  if (IC.MemoryIntrinsic.DestinationPointer) {
    assert(DestPtr && "Expected destination pointer");
    auto *ArgTy = PtrTy;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(DestPtr);
    RTArgNames.push_back("DestinationPointer");
  }

  if (IC.MemoryIntrinsic.DestinationPointerAddressSpace) {
    assert(DestPtr && "Expected destination pointer");
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(
        getCI(ArgTy, DestPtr->getType()->getPointerAddressSpace()));
    RTArgNames.push_back("DestinationPointerAddressSpace");
  }

  if (!SrcPtr)
    SrcPtr = Constant::getAllOnesValue(PtrTy);

  if (IC.MemoryIntrinsic.SourcePointer) {
    assert(SrcPtr && "Expected source pointer");
    auto *ArgTy = PtrTy;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(SrcPtr);
    RTArgNames.push_back("SourcePointer");
  }

  if (IC.MemoryIntrinsic.SourcePointerAddressSpace) {
    assert(SrcPtr && "Expected source pointer");
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, SrcPtr->getType()->getPointerAddressSpace()));
    RTArgNames.push_back("SourcePointerAddressSpace");
  }

  if (!MemsetValue)
    MemsetValue = Constant::getAllOnesValue(Int64Ty);
  if (IC.MemoryIntrinsic.MemsetValue) {
    assert(SrcPtr && " ");
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, SrcPtr->getType()->getPointerAddressSpace()));
    RTArgNames.push_back("MemsetValue");
  }

  if (IC.MemoryIntrinsic.Length) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(Length);
    RTArgNames.push_back("Length");
  }

  if (IC.MemoryIntrinsic.IsVolatile) {
    auto *ArgTy = Int8Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.isVolatile()));
    RTArgNames.push_back("IsVolatile");
  }

  FunctionCallee FC = getCallee("memory_intrinsic", RTArgTypes, RTArgNames,
                                /*After=*/false, /*Indirection=*/false);
  IRB.CreateCall(FC, RTArgs);

  return true;
}

bool InstrumentorImpl::instrumentGeneralIntrinsic(IntrinsicInst &I) {
  if (IC.GeneralIntrinsic.CB && !IC.GeneralIntrinsic.CB(I))
    return false;

  IRB.SetInsertPoint(&I);

  SmallVector<Type *> RTArgTypes;
  SmallVector<Value *> RTArgs;
  SmallVector<std::string> RTArgNames;

  if (IC.MemoryIntrinsic.KindId) {
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.getIntrinsicID()));
    RTArgNames.push_back("KindId");
  }

  FunctionCallee FC = getCallee("general_intrinsic", RTArgTypes, RTArgNames,
                                /*After=*/false, /*Indirection=*/false);
  IRB.CreateCall(FC, RTArgs);

  return true;
}

template <typename MemoryInstTy>
bool InstrumentorImpl::analyzeAccess(MemoryInstTy &I)
{
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
  if (IC.Store.CB && !IC.Store.CB(I))
    return false;
  if (hasAnnotation(&I, SkipAnnotation))
    return false;
  if (hasAnnotation(&I, COAnnotation))
    return false;

  IRB.SetInsertPoint(&I);

  AllocaInst *IndirectionAI = nullptr;
  SmallVector<Type *> RTArgTypes;
  SmallVector<Value *> RTArgs;
  SmallVector<std::string> RTArgNames;

  if (IC.Store.PointerOperand) {
    auto *ArgTy = PtrTy;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(
        IRB.CreatePointerBitCastOrAddrSpaceCast(I.getPointerOperand(), ArgTy));
    RTArgNames.push_back("PointerOperand");
  }

  if (IC.Store.PointerOperandAddressSpace) {
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.getPointerAddressSpace()));
    RTArgNames.push_back("PointerOperandAddressSpace");
  }

  if (IC.Store.ValueOperand) {
    Type *ArgTy = Int64Ty;
    if (DL.getTypeSizeInBits(I.getValueOperand()->getType()) > 64) {
      IndirectionAI =
          getAlloca(I.getFunction(), I.getValueOperand()->getType());
      IRB.CreateStore(I.getValueOperand(), IndirectionAI);
      ArgTy = PtrTy;
    }

    RTArgTypes.push_back(ArgTy);
    if (IndirectionAI)
      RTArgs.push_back(IndirectionAI);
    else
      RTArgs.push_back(tryToCast(IRB, I.getValueOperand(), ArgTy, DL));
    RTArgNames.push_back("ValueOperand");
  }

  if (IC.Store.ValueOperandSize) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(
        getCI(ArgTy, DL.getTypeStoreSize(I.getValueOperand()->getType())));
    RTArgNames.push_back("ValueOperandSize");
  }

  if (IC.Store.ValueOperandTypeId) {
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.getValueOperand()->getType()->getTypeID()));
    RTArgNames.push_back("ValueOperandTypeId");
  };

  if (IC.Store.Alignment) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.getAlign().value()));
    RTArgNames.push_back("Alignment");
  }

  if (IC.Store.AtomicityOrdering) {
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, uint64_t(I.getOrdering())));
    RTArgNames.push_back("AtomicityOrdering");
  }

  if (IC.Store.SyncScopeId) {
    auto *ArgTy = Int8Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, uint64_t(I.getSyncScopeID())));
    RTArgNames.push_back("SyncScopeId");
  }

  if (IC.Store.IsVolatile) {
    auto *ArgTy = Int8Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.isVolatile()));
    RTArgNames.push_back("IsVolatile");
  }

  if (IC.Store.BasePointerInfo) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    Value *BPI = findBasePointer(I.getPointerOperand());
    if (!BPI)
      BPI = getCI(ArgTy, 0);
    RTArgs.push_back(BPI);
    RTArgNames.push_back("BasePointerInfo");
  }

  Type *RetTy = IC.Store.ReplacePointerOperand ? PtrTy : nullptr;

  FunctionCallee FC =
      getCallee("store", RTArgTypes, RTArgNames, /*After=*/false, IndirectionAI, RetTy);
  Value *Ret = IRB.CreateCall(FC, RTArgs);
  if (RetTy)
    I.setOperand(I.getPointerOperandIndex(), Ret);

  return true;
}

bool InstrumentorImpl::instrumentLoad(LoadInst &I, bool After) {
  if (IC.Load.CB && !IC.Load.CB(I))
    return false;
  if (hasAnnotation(&I, SkipAnnotation))
    return false;
  if (hasAnnotation(&I, COAnnotation))
    return false;

  AllocaInst *IndirectionAI = nullptr;
  if (After)
    IRB.SetInsertPoint(I.getNextNonDebugInstruction());
  else
    IRB.SetInsertPoint(&I);

  SmallVector<Type *> RTArgTypes;
  SmallVector<Value *> RTArgs;
  SmallVector<std::string> RTArgNames;

  if (IC.Load.PointerOperand) {
    auto *ArgTy = PtrTy;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(
        IRB.CreatePointerBitCastOrAddrSpaceCast(I.getPointerOperand(), ArgTy));
    RTArgNames.push_back("PointerOperand");
  }

  if (IC.Load.PointerOperandAddressSpace) {
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.getPointerAddressSpace()));
    RTArgNames.push_back("PointerOperandAddressSpace");
  }

  if (IC.Load.Value && After) {
    Type *ArgTy = Int64Ty;
    if (DL.getTypeSizeInBits(I.getType()) > 64) {
      IndirectionAI = getAlloca(I.getFunction(), I.getType());
      IRB.CreateStore(&I, IndirectionAI);
      ArgTy = PtrTy;
    }

    RTArgTypes.push_back(ArgTy);
    if (IndirectionAI)
      RTArgs.push_back(IndirectionAI);
    else
      RTArgs.push_back(tryToCast(IRB, &I, ArgTy, DL));
    RTArgNames.push_back(IndirectionAI ? "ValueStorage" : "Value");
  }

  if (IC.Load.ValueSize) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, DL.getTypeStoreSize(I.getType())));
    RTArgNames.push_back("ValueSize");
  }

  if (IC.Load.ValueTypeId) {
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.getType()->getTypeID()));
    RTArgNames.push_back("ValueTypeId");
  };

  if (IC.Load.Alignment) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.getAlign().value()));
    RTArgNames.push_back("Alignment");
  }

  if (IC.Load.AtomicityOrdering) {
    auto *ArgTy = Int32Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, uint64_t(I.getOrdering())));
    RTArgNames.push_back("AtomicityOrdering");
  }

  if (IC.Load.SyncScopeId) {
    auto *ArgTy = Int8Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, uint64_t(I.getSyncScopeID())));
    RTArgNames.push_back("SyncScopeId");
  }

  if (IC.Load.IsVolatile) {
    auto *ArgTy = Int8Ty;
    RTArgTypes.push_back(ArgTy);
    RTArgs.push_back(getCI(ArgTy, I.isVolatile()));
    RTArgNames.push_back("IsVolatile");
  }

  if (IC.Load.BasePointerInfo) {
    auto *ArgTy = Int64Ty;
    RTArgTypes.push_back(ArgTy);
    Value *BPI = findBasePointer(I.getPointerOperand());
    if (!BPI)
      BPI = getCI(ArgTy, 0);
    RTArgs.push_back(BPI);
    RTArgNames.push_back("BasePointerInfo");
  }

  bool ReplaceValue = IC.Load.ReplaceValue && After;
  bool ReplacePointerOperand = IC.Load.ReplacePointerOperand && !After;

  Type *RetTy = nullptr;
  if (ReplaceValue && !IndirectionAI)
    RetTy = Int64Ty;
  else if (ReplacePointerOperand)
    RetTy = PtrTy;

  FunctionCallee FC =
      getCallee("load", RTArgTypes, RTArgNames, After, IndirectionAI, RetTy);
  auto *CI = IRB.CreateCall(FC, RTArgs);
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
  if (IC.Unreachable.CB && !IC.Unreachable.CB(I))
    return false;

  IRB.SetInsertPoint(&I);

  SmallVector<Type *> RTArgTypes;
  SmallVector<Value *> RTArgs;
  SmallVector<std::string> RTArgNames;

  FunctionCallee FC = getCallee("unreachable", RTArgTypes, RTArgNames, /*After=*/false, /*Indirection=*/false);
  IRB.CreateCall(FC, RTArgs);

  return true;
}

bool InstrumentorImpl::instrumentFunction(Function &Fn) {
  bool Changed = false;
  if (!shouldInstrumentFunction(&Fn))
    return Changed;

  if (IC.Load.SkipSafeAccess || IC.Load.SkipSafeAccess) {
    // TODO: Merge this into the main loop with RPOT
    for (auto &BB : Fn) {
      for (auto &I : BB) {
        switch (I.getOpcode()) {
        case Instruction::Load:
          if (IC.Load.SkipSafeAccess)
            Changed |= analyzeAccess(cast<LoadInst>(I));
          break;
        case Instruction::Store:
          if (IC.Store.SkipSafeAccess)
            Changed |= analyzeAccess(cast<StoreInst>(I));
          break;
        default:
          break;
        }
      }
    }
  }

  if (IC.BasePointer.Instrument)
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
        if (IC.BasePointer.Instrument)
          if (I.getType() == PtrTy)
            Changed |= instrumentBasePointer(I);
        break;
      default:
        break;
      }

      switch (I.getOpcode()) {
      case Instruction::Alloca:
        if (IC.Alloca.Instrument)
          Changed |= instrumentAlloca(cast<AllocaInst>(I));
        break;
      case Instruction::Call:
        if (IC.AllocationCall.Instrument || IC.MemoryIntrinsic.Instrument
            || IC.GeneralIntrinsic.Instrument)
          Changed |= instrumentCall(cast<CallBase>(I));
        break;
      case Instruction::Load:
        if (IC.Load.InstrumentBefore)
          Changed |= instrumentLoad(cast<LoadInst>(I), /*After=*/false);
        if (IC.Load.InstrumentAfter)
          Changed |= instrumentLoad(cast<LoadInst>(I), /*After=*/true);
        break;
      case Instruction::Store:
        if (IC.Store.Instrument)
          Changed |= instrumentStore(cast<StoreInst>(I));
        break;
      case Instruction::Unreachable:
        if (IC.Unreachable.Instrument)
          Changed |= instrumentUnreachable(cast<UnreachableInst>(I));
        break;
      default:
        break;
      }
    }
  }

  // TODO: Can be merged into the previous loop
  if (IC.CallArg.Instrument) {
    for (auto &It : RPOT)
      for (auto &I : *It)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (!NewInsts.contains(CI))
            Changed |= instrumentCallArgs(*CI);
  }

  return Changed;
}

bool InstrumentorImpl::instrumentMainFunction(Function &MainFn)
{
  if (!shouldInstrumentFunction(&MainFn))
    return false;
  if (IC.MainFunction.CB && !IC.MainFunction.CB(MainFn))
    return false;

  std::string MainFnName = getRTName("", "main");
  MainFn.setName(MainFnName);

  Function *InstMainFn = Function::Create(MainFn.getFunctionType(),
                                          GlobalValue::ExternalLinkage,
                                          "main", M);

  auto *EntryBB = BasicBlock::Create(Ctx, "entry", InstMainFn);
  IRB.SetInsertPoint(EntryBB, EntryBB->getFirstNonPHIOrDbgOrAlloca());

  SmallVector<Value *> Args;
  SmallVector<Value *> PtrArgs;
  SmallVector<Type *> RTArgTypes;
  SmallVector<std::string> RTArgNames;

  for (Argument &Arg : InstMainFn->args())
    Args.push_back(&Arg);

  if (IC.MainFunction.Args) {
    if (Args.empty())
      PtrArgs = { NullPtrVal, NullPtrVal };
    else if (IC.MainFunction.Args)
      for (size_t A = 0; A < Args.size(); ++A)
        PtrArgs.push_back(IRB.CreateAlloca(Args[A]->getType()));

    RTArgTypes = {PtrTy, PtrTy};
    RTArgNames = {"ArgcPtr", "ArgvPtr"};
  }

  for (size_t A = 0; A < PtrArgs.size(); ++A)
    if (PtrArgs[A] != NullPtrVal)
      IRB.CreateStore(Args[A], PtrArgs[A]);

  if (IC.MainFunction.InstrumentBefore) {
    FunctionCallee FC = getCallee("main", RTArgTypes, RTArgNames, /*After=*/false,
                                  /*Indirection=*/false);
    IRB.CreateCall(FC, PtrArgs);
  }

  for (size_t A = 0; A < PtrArgs.size(); ++A)
    if (PtrArgs[A] != NullPtrVal)
      Args[A] = IRB.CreateLoad(Args[A]->getType(), PtrArgs[A]);

  FunctionCallee FnCallee = M.getOrInsertFunction(MainFnName, MainFn.getFunctionType());
  Value *Ret = IRB.CreateCall(FnCallee, Args);

  if (IC.MainFunction.InstrumentAfter) {
    if (IC.MainFunction.Value) {
      RTArgTypes.push_back(Int32Ty);
      RTArgNames.push_back("ret");
      PtrArgs.push_back(Ret);
    }

    Type *RetTy = IC.MainFunction.ReplaceValue ? Int32Ty : nullptr;
    FunctionCallee FC = getCallee("main", RTArgTypes, RTArgNames, /*After=*/true,
                                  /*Indirection=*/false, RetTy);
    Value *Replacement = IRB.CreateCall(FC, PtrArgs);
    if (RetTy)
      Ret = Replacement;
  }

  IRB.CreateRet(Ret);

  return true;
}

bool InstrumentorImpl::instrumentModule(bool After) {
  Function *YtorFn = After ? DtorFn : CtorFn;
  assert(YtorFn);

  IRB.SetInsertPointPastAllocas(YtorFn);

  SmallVector<Type *> RTArgTypes;
  SmallVector<Value *> RTArgs;
  SmallVector<std::string> RTArgNames;

  if (IC.Module.Name) {
    RTArgTypes.push_back(PtrTy);
    RTArgNames.push_back("Name");
    RTArgs.push_back(getGlobalString(M.getName()));
  }

  if (IC.Module.TargetTriple) {
    RTArgTypes.push_back(PtrTy);
    RTArgNames.push_back("TargetTriple");
    RTArgs.push_back(getGlobalString(M.getTargetTriple()));
  }

  FunctionCallee FC = getCallee("module", RTArgTypes, RTArgNames, /*After=*/After,
                                /*Indirection=*/false);

  IRB.CreateCall(FC, RTArgs);

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

  SmallVector<Type *> RTArgTypes;
  SmallVector<std::string> RTArgNames;

  if (IC.GlobalVariable.Value) {
    RTArgTypes.push_back(PtrTy);
    RTArgNames.push_back("Value");
  }
  if (IC.GlobalVariable.Size) {
    RTArgTypes.push_back(Int64Ty);
    RTArgNames.push_back("Size");
  }
  if (IC.GlobalVariable.Alignment) {
    RTArgTypes.push_back(Int64Ty);
    RTArgNames.push_back("Alignment");
  }
  if (IC.GlobalVariable.Constant) {
    RTArgTypes.push_back(Int32Ty);
    RTArgNames.push_back("Constant");
  }
  if (IC.GlobalVariable.UnnamedAddress) {
    RTArgTypes.push_back(Int32Ty);
    RTArgNames.push_back("UnnamedAddr");
  }
  if (IC.GlobalVariable.Name) {
    RTArgTypes.push_back(PtrTy);
    RTArgNames.push_back("Name");
  }

  Type *RetTy = IC.GlobalVariable.ReplaceValue ? PtrTy : nullptr;
  FunctionCallee FC =
      getCallee("global", RTArgTypes, RTArgNames, /*After=*/true,
                /*Indirection=*/false, RetTy);

  SmallVector<GlobalVariable *> InstrGlobals;
  for (GlobalVariable *GV : Globals) {
    GlobalVariable *InstrGV =
        new GlobalVariable(M, PtrTy, false, GlobalValue::PrivateLinkage,
                           NullPtrVal, getRTName("", GV->getName()));

    SmallVector<Value *> RTArgs;
    if (IC.GlobalVariable.Value)
      RTArgs.push_back(GV);
    if (IC.GlobalVariable.Size)
      RTArgs.push_back(getCI(Int64Ty, DL.getTypeAllocSize(GV->getValueType())));
    if (IC.GlobalVariable.Alignment)
      RTArgs.push_back(getCI(Int64Ty, GV->getAlign().valueOrOne().value()));
    if (IC.GlobalVariable.Constant)
      RTArgs.push_back(getCI(Int32Ty, GV->isConstant()));
    if (IC.GlobalVariable.UnnamedAddress)
      RTArgs.push_back(getCI(Int32Ty, int(GV->getUnnamedAddr())));
    if (IC.GlobalVariable.Name)
      RTArgs.push_back(getGlobalString(GV->getName()));

    Value *Ret = IRB.CreateCall(FC, RTArgs);
    if (RetTy)
      IRB.CreateStore(Ret, InstrGV);

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
  if (auto *Arg = dyn_cast<Argument>(&ArgOrInst)) {
    IRB.SetInsertPointPastAllocas(Arg->getParent());
  } else if (auto *I = dyn_cast<Instruction>(&ArgOrInst)) {
    do {
      I = I->getNextNonDebugInstruction();
    } while (isa<AllocaInst>(I));
    IRB.SetInsertPoint(I);
  }

  SmallVector<Type *> RTArgTypes;
  SmallVector<Value *> RTArgs;
  SmallVector<std::string> RTArgNames;

  if (IC.BasePointer.Value) {
    RTArgTypes.push_back(PtrTy);
    RTArgs.push_back(&ArgOrInst);
    RTArgNames.push_back("Value");
  }

  Type *RetTy = IC.BasePointer.ReturnPointerInfo ? Int64Ty : nullptr;
  FunctionCallee FC = getCallee("baseptr", RTArgTypes, RTArgNames, /*After=*/true,
                                /*Indirection=*/false, RetTy);

  auto *CI = IRB.CreateCall(FC, RTArgs);
  if (RetTy)
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

Value * InstrumentorImpl::findBasePointer(Value *V) {
  if (!IC.BasePointer.Instrument || !IC.BasePointer.ReturnPointerInfo)
    return nullptr;

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
    return nullptr;
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
  Function *MainFn = nullptr;

  if (IC.Module.InstrumentBefore || IC.GlobalVariable.Instrument)
    addCtorOrDtor(/*Ctor=*/true);
  if (IC.Module.InstrumentAfter)
    addCtorOrDtor(/*Ctor=*/false);

  if (IC.Module.InstrumentBefore)
    Changed |= instrumentModule(/*After=*/false);
  if (IC.Module.InstrumentAfter)
    Changed |= instrumentModule(/*After=*/true);

  if (IC.GlobalVariable.Instrument)
    Changed |= prepareGlobalVariables();

  for (Function &Fn : M) {
    Changed |= instrumentFunction(Fn);

    if (Fn.getName() == "main")
      MainFn = &Fn;
  }

  if (IC.GlobalVariable.Instrument)
    Changed |= instrumentGlobalVariables();

  if (MainFn && (IC.MainFunction.InstrumentBefore || IC.MainFunction.InstrumentAfter))
    Changed |= instrumentMainFunction(*MainFn);

  if (IC.BasePointer.SkipUnused)
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

  assert(!verifyModule(M, &errs()));
  return PreservedAnalyses::none();
}
