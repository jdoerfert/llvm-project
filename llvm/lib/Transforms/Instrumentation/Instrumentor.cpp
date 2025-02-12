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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
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

#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>

using namespace llvm;
using namespace llvm::instrumentor;

#define DEBUG_TYPE "instrumentor"

static cl::opt<std::string> WriteJSONConfig(
    "instrumentor-write-config-file",
    cl::desc(
        "Write the instrumentor configuration into the specified JSON file"),
    cl::init(""));
static cl::opt<std::string> ReadJSONConfig(
    "instrumentor-read-config-file",
    cl::desc(
        "Read the instrumentor configuration from the specified JSON file"),
    cl::init(""));

namespace {

void writeInstrumentorConfig(InstrumentationConfig &IConf) {
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

  J.attributeBegin("configuration");
  J.objectBegin();
  for (auto *BaseCO : IConf.BaseConfigurationOpportunities) {
    switch (BaseCO->Kind) {
    case BaseConfigurationOpportunity::STRING:
      J.attribute(BaseCO->Name, BaseCO->getString());
      break;
    case BaseConfigurationOpportunity::BOOLEAN:
      J.attribute(BaseCO->Name, BaseCO->getBool());
      break;
    }
    if (!BaseCO->Description.empty())
      J.attribute(std::string(BaseCO->Name) + ".description",
                  BaseCO->Description);
  }
  J.objectEnd();
  J.attributeEnd();

  for (unsigned KindVal = 0; KindVal != InstrumentationLocation::Last;
       ++KindVal) {
    auto Kind = InstrumentationLocation::KindTy(KindVal);

    auto &KindChoices = IConf.IChoices[Kind];
    if (KindChoices.empty())
      continue;

    J.attributeBegin(InstrumentationLocation::getKindStr(Kind));
    J.objectBegin();
    for (auto &ChoiceIt : KindChoices) {
      J.attributeBegin(ChoiceIt.getKey());
      J.objectBegin();
      J.attribute("enabled", ChoiceIt.second->Enabled);
      for (auto &ArgIt : ChoiceIt.second->IRTArgs) {
        J.attribute(ArgIt.Name, ArgIt.Enabled);
        if ((ArgIt.Flags & IRTArg::REPLACABLE) ||
            (ArgIt.Flags & IRTArg::REPLACABLE_CUSTOM))
          J.attribute(std::string(ArgIt.Name) + ".replace", true);
        if (!ArgIt.Description.empty())
          J.attribute(std::string(ArgIt.Name) + ".description",
                      ArgIt.Description);
      }
      J.objectEnd();
      J.attributeEnd();
    }
    J.objectEnd();
    J.attributeEnd();
  }

  J.objectEnd();
}

bool readInstrumentorConfigFromJSON(InstrumentationConfig &IConf) {
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

  StringMap<BaseConfigurationOpportunity *> BCOMap;
  for (auto *BO : IConf.BaseConfigurationOpportunities)
    BCOMap[BO->Name] = BO;

  SmallPtrSet<InstrumentationOpportunity *, 32> SeenIOs;
  for (auto &It : *Config) {
    auto *Obj = It.second.getAsObject();
    if (!Obj) {
      errs() << "WARNING: malformed JSON configuration, expected an object.\n";
      continue;
    }
    if (It.first == "configuration") {
      for (auto &ObjIt : *Obj) {
        if (auto *BO = BCOMap.lookup(ObjIt.first)) {
          switch (BO->Kind) {
          case BaseConfigurationOpportunity::STRING:
            if (auto V = ObjIt.second.getAsString()) {
              BO->setString(IConf.SS.save(*V));
            } else
              errs() << "WARNING: configuration key '" << ObjIt.first
                     << "' expects a string, value ignored\n";
            break;
          case BaseConfigurationOpportunity::BOOLEAN:
            if (auto V = ObjIt.second.getAsBoolean())
              BO->setBool(*V);
            else
              errs() << "WARNING: configuration key '" << ObjIt.first
                     << "' expects a boolean, value ignored\n";
            break;
          }
        } else if (!StringRef(ObjIt.first).ends_with(".description")) {
          errs() << "WARNING: configuration key not found and ignored: "
                 << ObjIt.first << "\n";
        }
      }
      continue;
    }

    auto &IChoiceMap =
        IConf.IChoices[InstrumentationLocation::getKindFromStr(It.first)];
    for (auto &ObjIt : *Obj) {
      auto *InnerObj = ObjIt.second.getAsObject();
      if (!InnerObj) {
        errs()
            << "WARNING: malformed JSON configuration, expected an object.\n";
        continue;
      }
      auto *IO = IChoiceMap.lookup(ObjIt.first);
      if (!IO) {
        errs() << "WARNING: malformed JSON configuration, expected an object "
                  "matching an instrumentor choice.\n";
        continue;
      }
      SeenIOs.insert(IO);
      StringMap<bool> ValueMap, ReplaceMap;
      for (auto &InnerObjIt : *InnerObj) {
        auto Name = StringRef(InnerObjIt.first);
        if (Name.consume_back(".replace"))
          ReplaceMap[Name] = InnerObjIt.second.getAsBoolean().value_or(false);
        else
          ValueMap[Name] = InnerObjIt.second.getAsBoolean().value_or(false);
      }
      IO->Enabled = ValueMap["enabled"];
      for (auto &IRArg : IO->IRTArgs) {
        IRArg.Enabled = ValueMap[IRArg.Name];
        if (!ReplaceMap.lookup(IRArg.Name)) {
          IRArg.Flags &= ~IRTArg::REPLACABLE;
          IRArg.Flags &= ~IRTArg::REPLACABLE_CUSTOM;
        }
      }
    }
  }

  for (auto &IChoiceMap : IConf.IChoices)
    for (auto &It : IChoiceMap)
      if (!SeenIOs.count(It.second))
        It.second->Enabled = false;

  return true;
}

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

template <typename Ty> Constant *getCI(Type *IT, Ty Val) {
  return ConstantInt::get(IT, Val);
}

class InstrumentorImpl final {
public:
  InstrumentorImpl(InstrumentationConfig &IConf, Module &M,
                   ModuleAnalysisManager &MAM)
      : IConf(IConf), M(M),
        FAM(MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager()),
        IIRB(M, [this](Function &F) -> TargetLibraryInfo & {
          return FAM.getResult<TargetLibraryAnalysis>(F);
        }) {}

  void printRuntimeSignatures() {
    auto *OutPtr = getStubRuntimeOut();
    if (!OutPtr)
      return;
    auto &Out = *OutPtr;

    for (auto &ChoiceMap : IConf.IChoices) {
      for (auto &[_, IO] : ChoiceMap) {
        if (!IO->Enabled)
          continue;
        IRTCallDescription IRTCallDesc(*IO, IO->getRetTy(M.getContext()));
        const auto &Signatures = IRTCallDesc.createCSignature(IConf, IIRB.DL);
        const auto &Bodies = IRTCallDesc.createCBodies(IConf, IIRB.DL);
        if (!Signatures.first.empty()) {
          Out << Signatures.first << " {\n";
          Out << "  " << Bodies.first << "}\n\n";
        }
        if (!Signatures.second.empty()) {
          Out << Signatures.second << " {\n";
          Out << "  " << Bodies.second << "}\n\n";
        }
      }
    }
  }

  ~InstrumentorImpl() {
    if (StubRuntimeOut)
      delete StubRuntimeOut;
  }

  /// Instrument the module, public entry point.
  bool instrument();

private:
  bool shouldInstrumentFunction(Function &Fn);
  bool shouldInstrumentGlobalVariable(GlobalVariable &GV);

  bool instrumentFunction(Function &Fn);
  bool instrumentModule();

  template <typename MemoryInstTy> bool analyzeAccess(MemoryInstTy &I);

  DenseMap<unsigned, InstrumentationOpportunity *> InstChoicesPRE,
      InstChoicesPOST;

  raw_fd_ostream *StubRuntimeOut = nullptr;

  raw_fd_ostream *getStubRuntimeOut() {
    if (!IConf.RuntimeStubsFile->getString().empty()) {
      std::error_code EC;
      StubRuntimeOut =
          new raw_fd_ostream(IConf.RuntimeStubsFile->getString(), EC);
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

  /// The instrumentor configuration.
  InstrumentationConfig &IConf;

  /// The underlying module.
  Module &M;

  FunctionAnalysisManager &FAM;

protected:
  /// A special IR builder that keeps track of the inserted instructions.
  InstrumentorIRBuilderTy IIRB;
};

} // end anonymous namespace

bool InstrumentorImpl::shouldInstrumentFunction(Function &Fn) {
  if (Fn.isDeclaration())
    return false;
  return !Fn.getName().starts_with(IConf.getRTName()) || Fn.hasFnAttribute("instrument");
}

bool InstrumentorImpl::shouldInstrumentGlobalVariable(GlobalVariable &GV) {
  return !GV.getName().starts_with("llvm.") &&
         !GV.getName().starts_with(IConf.getRTName());
}

bool InstrumentorImpl::instrumentFunction(Function &Fn) {
  bool Changed = false;
  if (!shouldInstrumentFunction(Fn)) 
    return Changed;

  ReversePostOrderTraversal<Function *> RPOT(&Fn);
  for (auto &It : RPOT) {
    for (auto &I : *It) {
      // Skip instrumentation instructions.
      if (IIRB.NewInsts.contains(&I))
        continue;

      // Count epochs eagerly.
      ++IIRB.Epoche;

      Value *IPtr = &I;
      if (auto *IO = InstChoicesPRE.lookup(I.getOpcode())) {
        IIRB.IRB.SetInsertPoint(&I);
        Changed |= bool(IO->instrument(IPtr, IConf, IIRB));
      }

      if (auto *IO = InstChoicesPOST.lookup(I.getOpcode())) {
        IIRB.IRB.SetInsertPoint(I.getNextNonDebugInstruction());
        Changed |= bool(IO->instrument(IPtr, IConf, IIRB));
      }
    }
  }

  Value *FPtr = &Fn;
  for (auto &ChoiceIt : IConf.IChoices[InstrumentationLocation::FUNCTION_PRE]) {
    if (!ChoiceIt.second->Enabled)
      continue;
    // Count epochs eagerly.
    ++IIRB.Epoche;

    IIRB.IRB.SetInsertPointPastAllocas(cast<Function>(FPtr));
    ChoiceIt.second->instrument(FPtr, IConf, IIRB);
  }

  return Changed;
}

bool InstrumentorImpl::instrumentModule() {
  SmallVector<GlobalVariable *> Globals(make_pointer_range(M.globals()));

  auto CreateYtor = [&](bool Ctor) {
    Function *YtorFn = Function::Create(
        FunctionType::get(IIRB.VoidTy, false), GlobalValue::PrivateLinkage,
        IConf.getRTName(Ctor ? "ctor" : "dtor", ""), M);

    auto *EntryBB = BasicBlock::Create(IIRB.Ctx, "entry", YtorFn);
    IIRB.IRB.SetInsertPoint(EntryBB, EntryBB->begin());
    IIRB.IRB.CreateRetVoid();

    if (Ctor)
      appendToGlobalCtors(M, YtorFn, 0);
    else
      appendToGlobalDtors(M, YtorFn, 0);
    return YtorFn;
  };

  Function *CtorFn = nullptr, *DtorFn = nullptr;
  bool Changed = false;
  for (auto Loc : {InstrumentationLocation::MODULE_PRE,
                   InstrumentationLocation::MODULE_POST}) {
    bool IsPRE = InstrumentationLocation::isPRE(Loc);
    Function *&YtorFn = IsPRE ? CtorFn : DtorFn;
    for (auto &ChoiceIt : IConf.IChoices[Loc]) {
      auto *IO = ChoiceIt.second;
      if (!IO->Enabled)
        continue;
      if (!YtorFn)
        YtorFn = CreateYtor(IsPRE);
      IIRB.IRB.SetInsertPointPastAllocas(YtorFn);
      Value *YtorPtr = YtorFn;

      // Count epochs eagerly.
      ++IIRB.Epoche;

      Changed |= bool(IO->instrument(YtorPtr, IConf, IIRB));
    }
  }

  for (auto Loc : {InstrumentationLocation::GLOBAL_PRE,
                   InstrumentationLocation::GLOBAL_POST}) {
    bool IsPRE = InstrumentationLocation::isPRE(Loc);
    Function *&YtorFn = IsPRE ? CtorFn : DtorFn;
    for (auto &ChoiceIt : IConf.IChoices[Loc]) {
      auto *IO = ChoiceIt.second;
      if (!IO->Enabled)
        continue;
      if (!YtorFn)
        YtorFn = CreateYtor(IsPRE);
      for (GlobalVariable *GV : Globals) {
        if (!shouldInstrumentGlobalVariable(*GV))
          continue;
        if (IsPRE)
          IIRB.IRB.SetInsertPoint(YtorFn->getEntryBlock().getTerminator());
        else
          IIRB.IRB.SetInsertPointPastAllocas(YtorFn);
        Value *GVPtr = GV;

        // Count epochs eagerly.
        ++IIRB.Epoche;

        Changed |= bool(IO->instrument(GVPtr, IConf, IIRB));
      }
    }
  }

  return Changed;
}

bool InstrumentorImpl::instrument() {
  bool Changed = false;

  for (auto &ChoiceIt :
       IConf.IChoices[InstrumentationLocation::INSTRUCTION_PRE])
    if (ChoiceIt.second->Enabled)
      InstChoicesPRE[ChoiceIt.second->getOpcode()] = ChoiceIt.second;
  for (auto &ChoiceIt :
       IConf.IChoices[InstrumentationLocation::INSTRUCTION_POST])
    if (ChoiceIt.second->Enabled)
      InstChoicesPOST[ChoiceIt.second->getOpcode()] = ChoiceIt.second;

  for (Function &Fn : M)
    Changed |= instrumentFunction(Fn);

  Changed |= instrumentModule();

  return Changed;
}

PreservedAnalyses InstrumentorPass::run(Module &M, ModuleAnalysisManager &MAM) {
  InstrumentationConfig &IConf =
      UserIConf ? *UserIConf : *new InstrumentationConfig();
  IConf.populate(M.getContext());

  InstrumentorImpl Impl(IConf, M, MAM);
  if (!readInstrumentorConfigFromJSON(IConf))
    return PreservedAnalyses::all();
  writeInstrumentorConfig(IConf);

  Impl.printRuntimeSignatures();

  bool Changed = Impl.instrument();
  if (!Changed)
    return PreservedAnalyses::all();

  if (verifyModule(M))
    M.dump();
  assert(!verifyModule(M, &errs()));

  return PreservedAnalyses::none();
}

BaseConfigurationOpportunity *
BaseConfigurationOpportunity::getBoolOption(InstrumentationConfig &IConf,
                                            StringRef Name,
                                            StringRef Description, bool Value) {
  auto *BCO = new BaseConfigurationOpportunity();
  BCO->Name = Name;
  BCO->Description = Description;
  BCO->Kind = BOOLEAN;
  BCO->V.B = Value;
  IConf.addBaseChoice(BCO);
  return BCO;
}
BaseConfigurationOpportunity *BaseConfigurationOpportunity::getStringOption(
    InstrumentationConfig &IConf, StringRef Name, StringRef Description,
    StringRef Value) {
  auto *BCO = new BaseConfigurationOpportunity();
  BCO->Name = Name;
  BCO->Description = Description;
  BCO->Kind = STRING;
  BCO->V.S = Value;
  IConf.addBaseChoice(BCO);
  return BCO;
}

void InstrumentationConfig::populate(LLVMContext &Ctx) {
  /// List of all instrumentation opportunities.
  UnreachableIO::populate(*this, Ctx);
  BasePointerIO::populate(*this, Ctx);
  FunctionIO::populate(*this, Ctx);
  PtrToIntIO::populate(*this, Ctx);
  ModuleIO::populate(*this, Ctx);
  GlobalIO::populate(*this, Ctx);
  AllocaIO::populate(*this, Ctx);
  StoreIO::populate(*this, Ctx);
  LoadIO::populate(*this, Ctx);
  CallIO::populate(*this, Ctx);
  ICmpIO::populate(*this, Ctx);
}

void InstrumentationConfig::addChoice(InstrumentationOpportunity &IO) {
  auto *&ICPtr = IChoices[IO.getLocationKind()][IO.getName()];
  if (ICPtr && IO.getLocationKind() != InstrumentationLocation::SPECIAL_VALUE) {
    errs() << "WARNING: registered two instrumentation opportunities for the "
              "same location ("
           << ICPtr->getName() << " vs " << IO.getName() << ")!\n";
  }
  ICPtr = &IO;
}

void InstrumentationConfig::addCache(InstrumentationOpportunity &IO, InstrumentationCache *Cache) {
  auto *&ICPtr = ICaches[IO.getLocationKind()][IO.getName()];
  if (ICPtr) {
    errs() << "WARNING: registered two instrumentation caches for the "
              "same location!\n";
  }
  ICPtr = Cache;
}

Value *
InstrumentationConfig::getBasePointerInfo(Value &V,
                                          InstrumentorIRBuilderTy &IIRB) {
  Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();

  Value *VPtr = &V;
  VPtr = const_cast<Value *>(getUnderlyingObjectAggressive(VPtr));
  Value *&BPI = BasePointerInfoMap[{VPtr, Fn}];
  if (!BPI) {
    auto *BPIO =
        IChoices[InstrumentationLocation::SPECIAL_VALUE]["base_pointer_info"];
    if (!BPIO->Enabled) {
      errs() << "WARNING: Base pointer info disabled but required, passing "
                "nullptr.\n";
      return BPI = Constant::getNullValue(IIRB.IRB.getVoidTy());
    }
    IRBuilderBase::InsertPointGuard IP(IIRB.IRB);
    if (auto *BasePtrPHI = dyn_cast<PHINode>(VPtr))
      IIRB.IRB.SetInsertPoint(BasePtrPHI->getParent()->getFirstNonPHIOrDbg());
    else if (auto *BasePtrI = dyn_cast<Instruction>(VPtr))
      IIRB.IRB.SetInsertPoint(BasePtrI->getNextNode());
    else if (isa<GlobalValue>(VPtr) || isa<Argument>(VPtr))
      IIRB.IRB.SetInsertPointPastAllocas(
          IIRB.IRB.GetInsertBlock()->getParent());
    else {
      VPtr->dump();
      llvm_unreachable("Unexpected base pointer!");
    }
    BPI = BPIO->instrument(VPtr, *this, IIRB);
  }
  return BPI;
}

Value *InstrumentationOpportunity::forceCast(Value &V, Type &Ty,
                                             InstrumentorIRBuilderTy &IIRB) {
  if (V.getType()->isVoidTy())
    return Ty.isVoidTy() ? &V : Constant::getNullValue(&Ty);
  return tryToCast(IIRB.IRB, &V, &Ty,
                   IIRB.IRB.GetInsertBlock()->getDataLayout());
}

Value *InstrumentationOpportunity::replaceValue(Value &V, Value &NewV,
                                                InstrumentationConfig &IConf,
                                                InstrumentorIRBuilderTy &IIRB) {
  if (V.getType()->isVoidTy())
    return &V;

  auto *NewVCasted = &NewV;
  if (auto *I = dyn_cast<Instruction>(&NewV)) {
    IRBuilderBase::InsertPointGuard IPG(IIRB.IRB);
    IIRB.IRB.SetInsertPoint(I->getNextNode());
    NewVCasted = tryToCast(IIRB.IRB, &NewV, V.getType(), IIRB.DL,
                           /*AllowTruncate=*/true);
  }
  V.replaceUsesWithIf(NewVCasted, [&](Use &U) {
    if (IIRB.NewInsts.lookup(cast<Instruction>(U.getUser())) == IIRB.Epoche)
      return false;
    return true;
  });

  return &V;
}

IRTCallDescription::IRTCallDescription(InstrumentationOpportunity &IO,
                                       Type *RetTy)
    : IO(IO), RetTy(RetTy) {
  for (auto &It : IO.IRTArgs) {
    if (!It.Enabled)
      continue;
    NumReplaceableArgs += bool(It.Flags & IRTArg::REPLACABLE);
    MightRequireIndirection |= It.Flags & IRTArg::POTENTIALLY_INDIRECT;
  }
  if (NumReplaceableArgs > 1)
    MightRequireIndirection = RequiresIndirection = true;
}

static std::pair<std::string, std::string> getAsCType(Type *Ty,
                                                      unsigned Flags) {
  if (Ty->isIntegerTy()) {
    auto BW = Ty->getIntegerBitWidth();
    if (BW == 1)
      return {"bool ", "bool *"};
    auto S = "int" + std::to_string(BW) + "_t ";
    return {S, S + "*"};
  }
  if (Ty->isPointerTy())
    return {Flags & IRTArg::STRING ? "char *" : "void *", "void **"};
  if (Ty->isFloatTy())
    return {"float ", "float *"};
  if (Ty->isDoubleTy())
    return {"double ", "double *"};
  return {"<>", "<>"};
}

static std::string getPrintfFormatString(Type *Ty, unsigned Flags) {

  if (Ty->isIntegerTy()) {
    if (Ty->getIntegerBitWidth() > 32) {
      assert(Ty->getIntegerBitWidth() == 64);
      return "%lli";
    }
    return "%i";
  }
  if (Ty->isPointerTy())
    return Flags & IRTArg::STRING ? "%s" : "%p";
  if (Ty->isFloatTy())
    return "%f";
  if (Ty->isDoubleTy())
    return "%lf";
  return "<>";
}

std::pair<std::string, std::string>
IRTCallDescription::createCBodies(InstrumentationConfig &IConf,
                                  const DataLayout &DL) {
  std::string DirectFormat = "printf(\"" + IO.getName().str() +
                             (IO.IP.isPRE() ? " pre" : " post") + " -- ";
  std::string IndirectFormat = DirectFormat;
  std::string DirectArg, IndirectArg, DirectReturnValue, IndirectReturnValue;

  auto AddToFormats = [&](Twine S) {
    DirectFormat += S.str();
    IndirectFormat += S.str();
  };
  auto AddToArgs = [&](Twine S) {
    DirectArg += S.str();
    IndirectArg += S.str();
  };
  bool First = true;
  for (auto &IRArg : IO.IRTArgs) {
    if (!IRArg.Enabled)
      continue;
    if (!First)
      AddToFormats(", ");
    First = false;
    AddToArgs(", " + IRArg.Name);
    AddToFormats(IRArg.Name + ": ");
    if (NumReplaceableArgs == 1 && (IRArg.Flags & IRTArg::REPLACABLE)) {
      DirectReturnValue = IRArg.Name;
      if (!isPotentiallyIndirect(IRArg))
        IndirectReturnValue = IRArg.Name;
    }
    if (!isPotentiallyIndirect(IRArg)) {
      AddToFormats(getPrintfFormatString(IRArg.Ty, IRArg.Flags));
    } else {
      DirectFormat += getPrintfFormatString(IRArg.Ty, IRArg.Flags);
      IndirectFormat += "%p";
      IndirectArg += "_ptr";
      // Add the indirect argument size
      if (!(IRArg.Flags & IRTArg::INDIRECT_HAS_SIZE)) {
        IndirectFormat += ", " + IRArg.Name.str() + "_size: %i";
        IndirectArg += ", " + IRArg.Name.str() + "_size";
      }
    }
  }

  std::string DirectBody = DirectFormat + "\\n\"" + DirectArg + ");\n";
  std::string IndirectBody = IndirectFormat + "\\n\"" + IndirectArg + ");\n";
  if (RetTy) {
    assert(DirectReturnValue.empty() && IndirectReturnValue.empty() &&
           "Explicit return type but also implicit one!");
    IndirectReturnValue = DirectReturnValue = "0";
  }
  if (!DirectReturnValue.empty())
    DirectBody += "  return " + DirectReturnValue + ";\n";
  if (!IndirectReturnValue.empty())
    IndirectBody += "  return " + IndirectReturnValue + ";\n";
  return {DirectBody, IndirectBody};
}

std::pair<std::string, std::string>
IRTCallDescription::createCSignature(InstrumentationConfig &IConf,
                                     const DataLayout &DL) {
  SmallVector<std::string> DirectArgs, IndirectArgs;
  std::string DirectRetTy = "void ", IndirectRetTy = "void ";
  for (auto &IRArg : IO.IRTArgs) {
    if (!IRArg.Enabled)
      continue;
    const auto &[DirectArgTy, IndirectArgTy] =
        getAsCType(IRArg.Ty, IRArg.Flags);
    std::string DirectArg = DirectArgTy + IRArg.Name.str();
    std::string IndirectArg = IndirectArgTy + IRArg.Name.str() + "_ptr";
    std::string IndirectArgSize = "int32_t " + IRArg.Name.str() + "_size";
    DirectArgs.push_back(DirectArg);
    if (NumReplaceableArgs == 1 && (IRArg.Flags & IRTArg::REPLACABLE)) {
      DirectRetTy = DirectArgTy;
      if (!isPotentiallyIndirect(IRArg))
        IndirectRetTy = DirectArgTy;
    }
    if (!isPotentiallyIndirect(IRArg)) {
      IndirectArgs.push_back(DirectArg);
    } else {
      IndirectArgs.push_back(IndirectArg);
      if (!(IRArg.Flags & IRTArg::INDIRECT_HAS_SIZE))
        IndirectArgs.push_back(IndirectArgSize);
    }
  }

  auto DirectName =
      IConf.getRTName(IO.IP.isPRE() ? "pre_" : "post_", IO.getName(), "");
  auto IndirectName =
      IConf.getRTName(IO.IP.isPRE() ? "pre_" : "post_", IO.getName(), "_ind");
  auto MakeSignature = [&](std::string &RetTy, std::string &Name,
                           SmallVectorImpl<std::string> &Args) {
    return RetTy + Name + "(" + join(Args, ", ") + ")";
  };

  if (RetTy) {
    assert(DirectRetTy == "void " && IndirectRetTy == "void " &&
           "Explicit return type but also implicit one!");
    IndirectRetTy = DirectRetTy = getAsCType(RetTy, 0).first;
  }
  if (RequiresIndirection)
    return {"", MakeSignature(IndirectRetTy, IndirectName, IndirectArgs)};
  if (!MightRequireIndirection)
    return {MakeSignature(DirectRetTy, DirectName, DirectArgs), ""};
  return {MakeSignature(DirectRetTy, DirectName, DirectArgs),
          MakeSignature(IndirectRetTy, IndirectName, IndirectArgs)};
}

FunctionType *
IRTCallDescription::createLLVMSignature(InstrumentationConfig &IConf,
                                        LLVMContext &Ctx, const DataLayout &DL,
                                        bool ForceIndirection) {
  assert(((ForceIndirection && MightRequireIndirection) ||
          (!ForceIndirection && !RequiresIndirection)) &&
         "Wrong indirection setting!");

  SmallVector<Type *> ParamTypes;
  for (auto &It : IO.IRTArgs) {
    if (!It.Enabled)
      continue;
    if (!ForceIndirection || !isPotentiallyIndirect(It)) {
      ParamTypes.push_back(It.Ty);
      if (NumReplaceableArgs == 1 && (It.Flags & IRTArg::REPLACABLE))
        RetTy = It.Ty;
      continue;
    }

    // The indirection pointer and the size of the value.
    ParamTypes.push_back(PointerType::get(Ctx, 0));
    if (!(It.Flags & IRTArg::INDIRECT_HAS_SIZE))
      ParamTypes.push_back(IntegerType::getInt32Ty(Ctx));
  }
  if (!RetTy)
    RetTy = Type::getVoidTy(Ctx);

  return FunctionType::get(RetTy, ParamTypes, /*isVarArg=*/false);
}

CallInst *IRTCallDescription::createLLVMCall(Value *&V,
                                             InstrumentationConfig &IConf,
                                             InstrumentorIRBuilderTy &IIRB,
                                             const DataLayout &DL) {
  SmallVector<Value *> CallParams;

  bool ForceIndirection = RequiresIndirection;
  for (auto &It : IO.IRTArgs) {
    if (!It.Enabled)
      continue;
    auto *&Param = IConf.DirectArgumentCache[{IIRB.Epoche, It.Name}];
    if (!Param)
      Param = It.GetterCB(*V, *It.Ty, IConf, IIRB);
    if (Param->getType()->isVoidTy()) {
      Param = Constant::getNullValue(It.Ty);
    } else if (Param->getType()->isAggregateType() ||
               DL.getTypeSizeInBits(Param->getType()) >
                   DL.getTypeSizeInBits(It.Ty)) {
      if (!isPotentiallyIndirect(It)) {
        errs() << "WARNING: Indirection needed for " << *V << " in "
               << IO.getName()
               << " call, but not indicated; instrumentation is skipped";
        return nullptr;
      }
      ForceIndirection = true;
    } else {
      Param = tryToCast(IIRB.IRB, Param, It.Ty, DL);
    }
    CallParams.push_back(Param);
  }

  SmallVector<AllocaInst *> IndirectionAllocas;
  if (ForceIndirection) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
    for (unsigned I = 0, Offset = 0, E = IO.IRTArgs.size(); I < E; ++I) {
      if (!IO.IRTArgs[I].Enabled)
        continue;
      if (!isPotentiallyIndirect(IO.IRTArgs[I]))
        continue;
      auto *&CallParam = CallParams[I + Offset];
      if (!(IO.IRTArgs[I].Flags & IRTArg::INDIRECT_HAS_SIZE)) {
        CallParams.insert(&CallParam + 1, IIRB.IRB.getInt32(DL.getTypeStoreSize(
                                              CallParam->getType())));
        Offset += 1;
      }
      auto *&CachedParam =
          IConf.IndirectArgumentCache[{IIRB.Epoche, IO.IRTArgs[I].Name}];
      if (CachedParam) {
        CallParam = CachedParam;
        continue;
      }
      auto *AI = IIRB.getAlloca(Fn, CallParam->getType());
      IndirectionAllocas.push_back(AI);
      IIRB.IRB.CreateStore(CallParam, AI);
      CallParam = CachedParam = AI;
    }
  }

  auto *FnTy =
      createLLVMSignature(IConf, V->getContext(), DL, ForceIndirection);
  auto CompleteName =
      IConf.getRTName(IO.IP.isPRE() ? "pre_" : "post_", IO.getName(),
                      ForceIndirection ? "_ind" : "");
  auto FC = IIRB.IRB.GetInsertBlock()->getModule()->getOrInsertFunction(
      CompleteName, FnTy);
  auto *CI = IIRB.IRB.CreateCall(FC, CallParams);

  for (unsigned I = 0, E = IO.IRTArgs.size(); I < E; ++I) {
    if (!IO.IRTArgs[I].Enabled)
      continue;
    if (!isReplacable(IO.IRTArgs[I]))
      continue;
    bool IsCustomReplaceable = IO.IRTArgs[I].Flags & IRTArg::REPLACABLE_CUSTOM;
    Value *NewValue =
        FnTy->isVoidTy() || IsCustomReplaceable
            ? IConf.DirectArgumentCache[{IIRB.Epoche, IO.IRTArgs[I].Name}]
            : CI;
    assert(NewValue);
    if (ForceIndirection && !IsCustomReplaceable &&
        isPotentiallyIndirect(IO.IRTArgs[I])) {
      auto *Q = IConf.IndirectArgumentCache[{IIRB.Epoche, IO.IRTArgs[I].Name}];
      NewValue = IIRB.IRB.CreateLoad(IO.IRTArgs[I].Ty, Q);
    }
    V = IO.IRTArgs[I].SetterCB(*V, *NewValue, IConf, IIRB);
  }

  IIRB.returnAllocas(std::move(IndirectionAllocas));
  return CI;
}

template <typename Range>
static Value *createValuePack(const Range &R, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
  auto *Fn = IIRB.IRB.GetInsertBlock()->getParent();
  auto &DL = Fn->getDataLayout();
  auto *I32Ty = IIRB.IRB.getInt32Ty();
  SmallVector<Value *> Values;
  SmallVector<Type *> Types;
  bool AllConstant = true;
  for (auto &V : R) {
    Values.push_back(getCI(I32Ty, DL.getTypeStoreSize(V->getType())));
    Types.push_back(I32Ty);
    Values.push_back(getCI(I32Ty, V->getType()->getTypeID()));
    Types.push_back(I32Ty);
    Values.push_back(V);
    Types.push_back(V->getType());
    AllConstant &= isa<Constant>(V);
  }
  StructType *STy = StructType::get(Fn->getContext(), Types);
  if (AllConstant) {
    Constant *Initializer = ConstantStruct::get(
        STy, SmallVector<Constant *>(llvm::map_range(
                 Values, [](Value *V) { return cast<Constant>(V); })));

    GlobalVariable *&GV = IConf.ConstantGlobalsCache[Initializer];
    if (!GV)
      GV = new GlobalVariable(*Fn->getParent(), STy, true,
                              GlobalValue::InternalLinkage, Initializer,
                              IConf.getRTName().str() + "value_pack");
    return GV;
  }

  auto *AI = IIRB.getAlloca(Fn, STy);
  //unsigned Offset = 0;
  for (auto [Idx, Param] : enumerate(Values)) {
    auto *Ptr = IIRB.IRB.CreateStructGEP(STy, AI, Idx);
    //auto *Ptr = IIRB.IRB.CreateConstInBoundsGEP1_32(IIRB.Int8Ty, AI, Offset);
    IIRB.IRB.CreateStore(Param, Ptr);
    //Offset += DL.getTypeAllocSize(Param->getType());
  }
  IIRB.returnAllocas({AI});
  return AI;
}

template <typename Range>
static void readValuePack(const Range &R, Value &Pack,
                          InstrumentorIRBuilderTy &IIRB,
                          function_ref<void(int, Value *)> SetterCB) {
  auto *Fn = IIRB.IRB.GetInsertBlock()->getParent();
  auto &DL = Fn->getDataLayout();
  SmallVector<Value *> ParameterValues;
  unsigned Offset = 0;
  for (const auto &[Idx, Param] : enumerate(R)) {
    Offset += 8;
    auto *Ptr = IIRB.IRB.CreateConstInBoundsGEP1_32(IIRB.Int8Ty, &Pack, Offset);
    auto *NewV = IIRB.IRB.CreateLoad(Param->getType(), Ptr);
    SetterCB(Idx, NewV);
    Offset += DL.getTypeAllocSize(Param->getType());
  }
}

Value *AllocaIO::getSize(Value &V, Type &Ty, InstrumentationConfig &IO,
                         InstrumentorIRBuilderTy &IIRB) {
  auto &AI = cast<AllocaInst>(V);
  const DataLayout &DL = AI.getDataLayout();
  Value *SizeValue = nullptr;
  TypeSize TypeSize = DL.getTypeAllocSize(AI.getAllocatedType());
  if (TypeSize.isFixed()) {
    SizeValue = getCI(&Ty, TypeSize.getFixedValue());
  } else {
    auto *NullPtr = ConstantPointerNull::get(AI.getType());
    SizeValue = IIRB.IRB.CreatePtrToInt(
        IIRB.IRB.CreateGEP(AI.getAllocatedType(), NullPtr,
                           {IIRB.IRB.getInt32(1)}),
        &Ty);
  }
  if (AI.isArrayAllocation())
    SizeValue = IIRB.IRB.CreateMul(
        SizeValue, IIRB.IRB.CreateZExtOrBitCast(AI.getArraySize(), &Ty));
  return SizeValue;
}

Value *AllocaIO::setSize(Value &V, Value &NewV, InstrumentationConfig &IO,
                         InstrumentorIRBuilderTy &IIRB) {
  auto &AI = cast<AllocaInst>(V);
  const DataLayout &DL = AI.getDataLayout();
  auto *NewAI = IIRB.IRB.CreateAlloca(IIRB.IRB.getInt8Ty(),
                                      DL.getAllocaAddrSpace(), &NewV);
  NewAI->setAlignment(AI.getAlign());
  AI.replaceAllUsesWith(NewAI);
  IIRB.eraseLater(&AI);
  return NewAI;
}

Value *AllocaIO::getAlignment(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
  return getCI(&Ty, cast<AllocaInst>(V).getAlign().value());
}

Value *StoreIO::getPointer(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return SI.getPointerOperand();
}
Value *StoreIO::setPointer(Value &V, Value &NewV, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  SI.setOperand(SI.getPointerOperandIndex(), &NewV);
  return &SI;
}
Value *StoreIO::getPointerAS(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return getCI(&Ty, SI.getPointerAddressSpace());
}
Value *StoreIO::getBasePointerInfo(Value &V, Type &Ty,
                                   InstrumentationConfig &IConf,
                                   InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return IConf.getBasePointerInfo(*SI.getPointerOperand(), IIRB);
}
Value *StoreIO::getValue(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return SI.getValueOperand();
}
Value *StoreIO::getValueSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  auto &DL = SI.getDataLayout();
  return getCI(&Ty, DL.getTypeStoreSize(SI.getValueOperand()->getType()));
}
Value *StoreIO::getAlignment(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return getCI(&Ty, SI.getAlign().value());
}
Value *StoreIO::getValueTypeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return getCI(&Ty, SI.getValueOperand()->getType()->getTypeID());
}
Value *StoreIO::getAtomicityOrdering(Value &V, Type &Ty,
                                     InstrumentationConfig &IConf,
                                     InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return getCI(&Ty, uint64_t(SI.getOrdering()));
}
Value *StoreIO::getSyncScopeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return getCI(&Ty, uint64_t(SI.getSyncScopeID()));
}
Value *StoreIO::isVolatile(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return getCI(&Ty, SI.isVolatile());
}

Value *LoadIO::getPointer(Value &V, Type &Ty, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return LI.getPointerOperand();
}
Value *LoadIO::setPointer(Value &V, Value &NewV, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  LI.setOperand(LI.getPointerOperandIndex(), &NewV);
  return &LI;
}
Value *LoadIO::getPointerAS(Value &V, Type &Ty, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return getCI(&Ty, LI.getPointerAddressSpace());
}
Value *LoadIO::getBasePointerInfo(Value &V, Type &Ty,
                                  InstrumentationConfig &IConf,
                                  InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return IConf.getBasePointerInfo(*LI.getPointerOperand(), IIRB);
}
Value *LoadIO::getValue(Value &V, Type &Ty, InstrumentationConfig &IConf,
                        InstrumentorIRBuilderTy &IIRB) {
  return &V;
}
Value *LoadIO::getValueSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  auto &DL = LI.getDataLayout();
  return getCI(&Ty, DL.getTypeStoreSize(LI.getType()));
}
Value *LoadIO::getAlignment(Value &V, Type &Ty, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return getCI(&Ty, LI.getAlign().value());
}
Value *LoadIO::getValueTypeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return getCI(&Ty, LI.getType()->getTypeID());
}
Value *LoadIO::getAtomicityOrdering(Value &V, Type &Ty,
                                    InstrumentationConfig &IConf,
                                    InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return getCI(&Ty, uint64_t(LI.getOrdering()));
}
Value *LoadIO::getSyncScopeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return getCI(&Ty, uint64_t(LI.getSyncScopeID()));
}
Value *LoadIO::isVolatile(Value &V, Type &Ty, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return getCI(&Ty, LI.isVolatile());
}

Value *CallIO::getCallee(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB) {
  auto &CI = cast<CallInst>(V);
  if (CI.getIntrinsicID() != Intrinsic::not_intrinsic)
    return Constant::getNullValue(&Ty);
  return CI.getCalledOperand();
}
Value *CallIO::getCalleeName(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB) {
  auto &CI = cast<CallInst>(V);
  if (auto *Fn = CI.getCalledFunction())
    return IConf.getGlobalString(IConf.DemangleFunctionNames->getBool()
                                     ? demangle(Fn->getName())
                                     : Fn->getName(),
                                 IIRB);
  return Constant::getNullValue(&Ty);
}
Value *CallIO::getIntrinsicId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
  auto &CI = cast<CallInst>(V);
  return getCI(&Ty, CI.getIntrinsicID());
}
Value *CallIO::getAllocationInfo(Value &V, Type &Ty,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
  auto &CI = cast<CallInst>(V);
  auto &TLI = IIRB.TLIGetter(*CI.getFunction());
  auto ACI = getAllocationCallInfo(&CI, &TLI);
  if (!ACI)
    return Constant::getNullValue(&Ty);

  auto &Ctx = CI.getContext();

  StructType *STy =
      StructType::get(Ctx, {IIRB.PtrTy, IIRB.Int32Ty, IIRB.Int32Ty,
                            IIRB.Int32Ty, IIRB.Int8Ty, IIRB.Int32Ty});
  SmallVector<Constant *> Values;

  if (ACI->Family)
    Values.push_back(IConf.getGlobalString(*ACI->Family, IIRB));
  else
    Values.push_back(Constant::getNullValue(IIRB.PtrTy));

  Values.push_back(getCI(IIRB.Int32Ty, ACI->SizeLHSArgNo));
  Values.push_back(getCI(IIRB.Int32Ty, ACI->SizeRHSArgNo));
  Values.push_back(getCI(IIRB.Int32Ty, ACI->AlignmentArgNo));

  if (auto *InitialCI = dyn_cast_if_present<ConstantInt>(ACI->InitialValue)) {
    Values.push_back(getCI(IIRB.Int8Ty, 1));
    Values.push_back(getCI(IIRB.Int32Ty, InitialCI->getZExtValue()));
  } else if (isa_and_present<UndefValue>(ACI->InitialValue)) {
    Values.push_back(getCI(IIRB.Int8Ty, 2));
    Values.push_back(getCI(IIRB.Int32Ty, 0));
  } else {
    Values.push_back(getCI(IIRB.Int8Ty, 0));
    Values.push_back(getCI(IIRB.Int32Ty, 0));
  }

  Constant *Initializer = ConstantStruct::get(STy, Values);
  GlobalVariable *&GV = IConf.ConstantGlobalsCache[Initializer];
  if (!GV)
    GV = new GlobalVariable(*CI.getModule(), STy, true,
                            GlobalValue::InternalLinkage, Initializer,
                            IConf.getRTName().str() + "allocation_call_info");
  return GV;
}
Value *CallIO::getValueSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB) {
  auto &CI = cast<CallInst>(V);
  if (CI.getType()->isVoidTy())
    return getCI(&Ty, 0);
  auto &DL = CI.getDataLayout();
  return getCI(&Ty, DL.getTypeStoreSize(CI.getType()));
}
Value *CallIO::getNumCallParameters(Value &V, Type &Ty,
                                    InstrumentationConfig &IConf,
                                    InstrumentorIRBuilderTy &IIRB) {
  auto &CI = cast<CallInst>(V);
  return getCI(&Ty, CI.arg_size());
}
Value *CallIO::getCallParameters(Value &V, Type &Ty,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
  auto &CI = cast<CallInst>(V);
  return createValuePack(CI.args(), IConf, IIRB);
}
Value *CallIO::setCallParameters(Value &V, Value &NewV,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
  auto &CI = cast<CallInst>(V);
  readValuePack(CI.args(), NewV, IIRB, [&](int Idx, Value *ReplV) {
    // Do not replace `immarg` operands with a non-immediate.
    if (CI.getParamAttr(Idx, Attribute::ImmArg).isValid())
      return;
    CI.setArgOperand(Idx, ReplV);
  });
  return &CI;
}
Value *CallIO::isDefinition(Value &V, Type &Ty, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB) {
  auto &CI = cast<CallInst>(V);
  return getCI(&Ty, !CI.getCalledFunction()->isDeclaration());
}

Value *ICmpIO::getCmpPredicate(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB) {
  auto &II = cast<ICmpInst>(V);
  return getCI(&Ty, II.getCmpPredicate());
}
Value *ICmpIO::isPtrCmp(Value &V, Type &Ty, InstrumentationConfig &IConf,
                        InstrumentorIRBuilderTy &IIRB) {
  auto &II = cast<ICmpInst>(V);
  return getCI(&Ty, II.getOperand(0)->getType()->isPointerTy());
}
Value *ICmpIO::getLHS(Value &V, Type &Ty, InstrumentationConfig &IConf,
                      InstrumentorIRBuilderTy &IIRB) {
  auto &II = cast<ICmpInst>(V);
  return tryToCast(IIRB.IRB, II.getOperand(0), &Ty, IIRB.DL);
}
Value *ICmpIO::getRHS(Value &V, Type &Ty, InstrumentationConfig &IConf,
                      InstrumentorIRBuilderTy &IIRB) {
  auto &II = cast<ICmpInst>(V);
  return tryToCast(IIRB.IRB, II.getOperand(1), &Ty, IIRB.DL);
}

Value *PtrToIntIO::getPtr(Value &V, Type &Ty, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB) {
  auto &PI = cast<PtrToIntInst>(V);
  return PI.getPointerOperand();
}

Value *BasePointerIO::getPointerKind(Value &V, Type &Ty,
                                     InstrumentationConfig &IConf,
                                     InstrumentorIRBuilderTy &IIRB) {
  if (isa<Argument>(V))
    return getCI(&Ty, 0);
  if (isa<GlobalValue>(V))
    return getCI(&Ty, 1);
  if (isa<Instruction>(V))
    return getCI(&Ty, 2);
  return getCI(&Ty, 3);
}

Value *FunctionIO::getFunctionAddress(Value &V, Type &Ty,
                                      InstrumentationConfig &IConf,
                                      InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  if (Fn.isIntrinsic())
    return Constant::getNullValue(&Ty);
  return &V;
}
Value *FunctionIO::getFunctionName(Value &V, Type &Ty,
                                   InstrumentationConfig &IConf,
                                   InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  return IConf.getGlobalString(IConf.DemangleFunctionNames->getBool()
                                   ? demangle(Fn.getName())
                                   : Fn.getName(),
                               IIRB);
}
Value *FunctionIO::getNumArguments(Value &V, Type &Ty,
                                   InstrumentationConfig &IConf,
                                   InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  return getCI(&Ty, Fn.arg_size());
}
Value *FunctionIO::getArguments(Value &V, Type &Ty,
                                InstrumentationConfig &IConf,
                                InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  if (Fn.arg_empty())
    return Constant::getNullValue(&Ty);
  return createValuePack(make_pointer_range(Fn.args()), IConf, IIRB);
}
Value *FunctionIO::setArguments(Value &V, Value &NewV,
                                InstrumentationConfig &IConf,
                                InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  if (Fn.arg_empty())
    return &V;
  readValuePack(
      make_pointer_range(Fn.args()), NewV, IIRB, [&](int Idx, Value *ReplV) {
        Fn.getArg(Idx)->replaceUsesWithIf(ReplV, [&](Use &U) {
          return IIRB.NewInsts.lookup(cast<Instruction>(U.getUser())) !=
                 IIRB.Epoche;
        });
      });
  return &V;
}

Value *ModuleIO::getModuleName(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  return IConf.getGlobalString(Fn.getParent()->getName(), IIRB);
}
Value *ModuleIO::getTargetTriple(Value &V, Type &Ty,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  return IConf.getGlobalString(Fn.getParent()->getTargetTriple(), IIRB);
}

Value *GlobalIO::getAddress(Value &V, Type &Ty, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB) {
  GlobalVariable &GV = cast<GlobalVariable>(V);
  if (GV.getAddressSpace())
    return ConstantExpr::getAddrSpaceCast(&GV, IIRB.PtrTy);
  return &GV;
}
Value *GlobalIO::setAddress(Value &V, Value &NewV, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB) {
  GlobalVariable &GV = cast<GlobalVariable>(V);
  SmallPtrSet<Use *, 8> Visited;
  SmallVector<Use *> Worklist(make_pointer_range(GV.uses()));

  GlobalVariable *ShadowGV = nullptr;
  auto ShadowName = IConf.getRTName("shadow.", GV.getName());
  auto &DL = GV.getDataLayout();
  if (GV.isDeclaration()) {
    ShadowGV = new GlobalVariable(*GV.getParent(), GV.getType(), false,
                                  GlobalVariable::WeakODRLinkage, &GV,
                                  ShadowName, &GV, GV.getThreadLocalMode(),
                                  DL.getDefaultGlobalsAddressSpace());
  } else {
    ShadowGV =
        new GlobalVariable(*GV.getParent(), NewV.getType(), false,
                           (GV.hasLocalLinkage() || GV.hasWeakLinkage())
                               ? GV.getLinkage()
                               : GlobalVariable::ExternalLinkage,
                           PoisonValue::get(NewV.getType()), ShadowName, &GV);
    IIRB.IRB.CreateStore(&NewV, ShadowGV);
  }

  DenseMap<Function *, Value *> FunctionReloadMap;
  while (!Worklist.empty()) {
    Use *U = Worklist.pop_back_val();
    if (!Visited.insert(U).second)
      continue;
    auto *I = dyn_cast<Instruction>(U->getUser());
    if (!I) {
      for (auto &UU : (*U)->uses())
        Worklist.push_back(&UU);
      continue;
    }
    if (IIRB.NewInsts.lookup(I) == IIRB.Epoche)
      continue;
    IIRB.IRB.SetInsertPointPastAllocas(I->getFunction());
    auto *&Reload = FunctionReloadMap[I->getFunction()];
    if (!Reload)
      Reload = IIRB.IRB.CreateLoad(NewV.getType(), ShadowGV);
    U->set(Reload);
  }
  return &V;
}
Value *GlobalIO::getSymbolName(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB) {
  GlobalVariable &GV = cast<GlobalVariable>(V);
  return IConf.getGlobalString(GV.getName(), IIRB);
}
Value *GlobalIO::getInitialValue(Value &V, Type &Ty,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
  GlobalVariable &GV = cast<GlobalVariable>(V);
  return GV.hasInitializer() ? GV.getInitializer()
                             : Constant::getNullValue(&Ty);
}
Value *GlobalIO::getInitialValueSize(Value &V, Type &Ty,
                                     InstrumentationConfig &IConf,
                                     InstrumentorIRBuilderTy &IIRB) {
  GlobalVariable &GV = cast<GlobalVariable>(V);
  auto &DL = GV.getDataLayout();
  return GV.hasInitializer()
             ? getCI(&Ty, DL.getTypeAllocSize(GV.getValueType()))
             : Constant::getNullValue(&Ty);
}
Value *GlobalIO::isConstant(Value &V, Type &Ty, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB) {
  GlobalVariable &GV = cast<GlobalVariable>(V);
  return getCI(&Ty, GV.isConstant());
}
