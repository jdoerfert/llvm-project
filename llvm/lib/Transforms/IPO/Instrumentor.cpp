//===-- Instrumentor.cpp - Highly configurable instrumentation pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Instrumentor.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
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
                  "matching an instrumentor choice, got "
               << ObjIt.first << ".\n";
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
  InstrumentorImpl(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
                   Module &M, FunctionAnalysisManager &FAM)
      : IConf(IConf), M(M), FAM(FAM), IIRB(IIRB) {
    IConf.populate(IIRB);
  }

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
  void linkRuntime();

  bool shouldInstrumentTarget();
  bool shouldInstrumentFunction(Function &Fn);
  bool shouldInstrumentGlobalVariable(GlobalVariable &GV);

  bool instrumentFunction(Function &Fn);
  bool instrumentModule();
  bool prepareModule();

  bool preprocessLoops(Function &Fn) {
    bool Changed = false;
    auto *LVRIO = IConf.IChoices[InstrumentationLocation::SPECIAL_VALUE]
                                ["loop_value_range"];
    if (!LVRIO || !LVRIO->Enabled)
      return Changed;

    auto &LI = FAM.getResult<LoopAnalysis>(Fn);
    auto &DT = FAM.getResult<DominatorTreeAnalysis>(Fn);

    SmallVector<Loop *> Worklist;
    Worklist.append(LI.begin(), LI.end());

    while (!Worklist.empty()) {
      auto *L = Worklist.pop_back_val();
      if (!L->getLoopPreheader())
        if (InsertPreheaderForLoop(L, &DT, &LI, nullptr, true))
          Changed = true;
      Worklist.append(L->begin(), L->end());
    }
    return Changed;
  }

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
  InstrumentorIRBuilderTy &IIRB;
};

} // end anonymous namespace
//
Value *instrumentor::getUnderlyingObjectRecursive(Value *Ptr) {
  auto *NewVPtr = const_cast<Value *>(getUnderlyingObjectAggressive(Ptr));
  while (NewVPtr != Ptr) {
    Ptr = NewVPtr;
    NewVPtr = const_cast<Value *>(getUnderlyingObjectAggressive(Ptr));
  }
  return Ptr;
}

bool InstrumentorImpl::shouldInstrumentTarget() {
  const auto &TripleStr = M.getTargetTriple();
  Triple T(TripleStr);
  const bool IsGPU = T.isAMDGPU() || T.isNVPTX();

  bool RegexMatches = true;
  const auto TargetRegexStr = IConf.TargetRegex->getString();
  if (!TargetRegexStr.empty()) {
    llvm::Regex TargetRegex(TargetRegexStr);
    std::string ErrMsg;
    if (!TargetRegex.isValid(ErrMsg)) {
      errs() << "WARNING: failed to parse target regex: " << ErrMsg << "\n";
      return false;
    }
    RegexMatches = TargetRegex.match(TripleStr);
  }

  return ((IsGPU && IConf.GPUEnabled->getBool()) ||
          (!IsGPU && IConf.HostEnabled->getBool())) &&
         RegexMatches;
}

bool InstrumentorImpl::shouldInstrumentFunction(Function &Fn) {
  if (Fn.isDeclaration())
    return false;
  return !Fn.getName().starts_with(IConf.getRTName()) ||
         Fn.hasFnAttribute("instrument");
}

bool InstrumentorImpl::shouldInstrumentGlobalVariable(GlobalVariable &GV) {
  return !GV.getName().starts_with("llvm.") &&
         !GV.getName().starts_with(IConf.getRTName());
}

bool InstrumentorImpl::instrumentFunction(Function &Fn) {
  bool Changed = false;
  if (!shouldInstrumentFunction(Fn))
    return Changed;

  IConf.startFunction();

  // Ensure there is at least one alloca to make the insertion point stable
  IIRB.getAlloca(&Fn, IIRB.PtrTy);

  Changed |= preprocessLoops(Fn);

  InstrumentationCaches ICaches;

  auto InstrumentInst = [&](Instruction &I) {
    // Skip instrumentation instructions.
    if (IIRB.NewInsts.contains(&I))
      return;

    // Count epochs eagerly.
    ++IIRB.Epoche;

    Value *IPtr = &I;
    if (auto *IO = InstChoicesPRE.lookup(I.getOpcode())) {
      IIRB.IRB.SetInsertPoint(&I);
      ensureDbgLoc(IIRB.IRB);
      Changed |= bool(IO->instrument(IPtr, IConf, IIRB, ICaches));
    }

    if (auto *IO = InstChoicesPOST.lookup(I.getOpcode())) {
      IIRB.IRB.SetInsertPoint(I.getNextNonDebugInstruction());
      ensureDbgLoc(IIRB.IRB);
      Changed |= bool(IO->instrument(IPtr, IConf, IIRB, ICaches));
    }
    IIRB.returnAllocas();
  };

  SmallVector<Instruction *> FinalTIs;
  ReversePostOrderTraversal<Function *> RPOT(&Fn);

  if (IConf.IChoices[InstrumentationLocation::SPECIAL_VALUE]
                    ["loop_value_range"]) {
    for (auto &It : RPOT) {
      for (auto &I : *It)
        if (auto *Ptr = AA::getPointerOperand(&I, /*AllowVolatile*/ true))
          IIRB.computeLoopRangeValues(*Ptr, 0);
    }
  }

  for (auto &It : RPOT) {
    for (auto &I : *It)
      InstrumentInst(I);

    auto *TI = It->getTerminator();
    if (!TI->getNumSuccessors())
      FinalTIs.push_back(TI);
  }

  //  for (auto *GenI : IIRB.GeneratedInsts)
  //    InstrumentInst(*GenI);
  //  IIRB.GeneratedInsts.clear();

  Value *FPtr = &Fn;
  for (auto &ChoiceIt : IConf.IChoices[InstrumentationLocation::FUNCTION_PRE]) {
    if (!ChoiceIt.second->Enabled)
      continue;
    // Count epochs eagerly.
    ++IIRB.Epoche;

    IIRB.IRB.SetInsertPointPastAllocas(cast<Function>(FPtr));
    ensureDbgLoc(IIRB.IRB);
    ChoiceIt.second->instrument(FPtr, IConf, IIRB, ICaches);
    IIRB.returnAllocas();
  }

  for (auto &ChoiceIt :
       IConf.IChoices[InstrumentationLocation::FUNCTION_POST]) {
    if (!ChoiceIt.second->Enabled)
      continue;
    // Count epochs eagerly.
    ++IIRB.Epoche;

    for (auto *FinalTI : FinalTIs) {
      IIRB.IRB.SetInsertPoint(FinalTI);
      ensureDbgLoc(IIRB.IRB);
      ChoiceIt.second->instrument(FPtr, IConf, IIRB, ICaches);
      IIRB.returnAllocas();
    }
  }

  return Changed;
}

bool InstrumentorImpl::instrumentModule() {
  SmallVector<GlobalVariable *> Globals;
  Globals.reserve(M.global_size());
  for (GlobalVariable &GV : M.globals()) {
    // llvm.metadata contains globals such as llvm.used
    if (GV.getSection() == "llvm.metadata" ||
        GV.getName() == "llvm.global_dtors" ||
        GV.getName() == "llvm.global_ctors")
      continue;
    Globals.push_back(&GV);
  }

  auto CreateYtor = [&](bool Ctor) {
    auto Name = IConf.getRTName(Ctor ? "ctor" : "dtor", "");
    Function *YtorFn = Function::Create(FunctionType::get(IIRB.VoidTy, false),
                                        GlobalValue::PrivateLinkage, Name, M);
    auto *GV = new GlobalVariable(M, YtorFn->getType(), true,
                                  GlobalValue::ExternalLinkage, YtorFn, "");
    GV->setSection(Name);

    auto *EntryBB = BasicBlock::Create(IIRB.Ctx, "entry", YtorFn);
    IIRB.IRB.SetInsertPoint(EntryBB, EntryBB->begin());
    ensureDbgLoc(IIRB.IRB);
    IIRB.IRB.CreateRetVoid();

    //    if (Ctor)
    //      appendToGlobalCtors(M, YtorFn, 1000);
    //    else
    //      appendToGlobalDtors(M, YtorFn, 1000);
    return YtorFn;
  };

  InstrumentationCaches ICaches;

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
      ensureDbgLoc(IIRB.IRB);
      Value *YtorPtr = YtorFn;

      // Count epochs eagerly.
      ++IIRB.Epoche;

      Changed |= bool(IO->instrument(YtorPtr, IConf, IIRB, ICaches));
      IIRB.returnAllocas();
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
        ensureDbgLoc(IIRB.IRB);
        Value *GVPtr = GV;

        // Count epochs eagerly.
        ++IIRB.Epoche;

        Changed |= bool(IO->instrument(GVPtr, IConf, IIRB, ICaches));
        IIRB.returnAllocas();
      }
    }
  }

  return Changed;
}

void InstrumentorImpl::linkRuntime() {
  const auto RuntimeBitcode = IConf.RuntimeBitcode->getString();
  if (RuntimeBitcode.empty())
    return;

  SMDiagnostic Err;
  auto RTM = parseIRFile(RuntimeBitcode, Err, M.getContext());
  if (!RTM) {
    errs() << "ERROR: failed to parse runtime bitcode file '" << RuntimeBitcode
           << "':\n";
    Err.print(M.getName().data(), errs());
    return;
  }

  auto InternalizeCallback = [&](Module &M, const StringSet<> &GVS) {
    internalizeModule(M, [&GVS](const GlobalValue &GV) {
      return !GV.hasName() || !GVS.count(GV.getName());
    });
  };

  if (Linker::linkModules(M, std::move(RTM), 0, InternalizeCallback))
    llvm_unreachable("failed to link in runtime bitcode");

  if (!IConf.InlineRuntimeEagerly->getBool())
    return;

  for (auto [I, _] : IIRB.NewInsts) {
    auto *CI = dyn_cast<CallInst>(I);
    if (!CI || isa<IntrinsicInst>(CI))
      continue;

    InlineFunctionInfo IFI;
    auto InlineResult = InlineFunction(*CI, IFI);
    if (!InlineResult.isSuccess()) {
      errs() << "WARNING: inlining of runtime call failed: "
             << CI->getCalledFunction()->getName() << "\n";
      errs() << "Reason: " << InlineResult.getFailureReason() << "\n";
      errs() << "Signatures: " << *CI->getFunctionType() << " vs "
             << *CI->getCalledFunction()->getFunctionType() << "\n";
    }
  }

  for (auto It : IIRB.AllocaMap) {
    auto *Fn = It.first.first;
    DominatorTree DT(*Fn);
    auto &Allocas = *It.second;
    erase_if(Allocas,
             [](const AllocaInst *AI) { return !isAllocaPromotable(AI); });
    PromoteMemToReg(Allocas, DT);
    delete It.second;
  }
  IIRB.AllocaMap.clear();
}

bool InstrumentorImpl::prepareModule() {
  // We need to make sure each invoke instruction first makes a jump to a basic
  // block without any PHI nodes, so that there is a place where we can insert
  // post-value instrumentation that could later be used in a PHI node in the
  // original normal destination block.
  //
  // Example case:
  //
  // invoke_bb:
  //   %invoke_res = invoke ... to label %normal ...
  // normal:
  //   %p = phi [%invoke_res, %invoke_bb]
  //   ...
  //
  // This allow us to instrument the result of the invoke:
  //
  // invoke_bb:
  //   %invoke_res = invoke ... to label %normal ...
  // new_bb:
  //   %post_invoke_res = __rt_post_invoke(%invoke_res)
  //   br %normal
  // normal:
  //   %p = phi [%post_invoke_res, %new_bb]
  //   %p2 = phi [%invoke_res, %new_bb]
  //   ...
  //
  bool Changed = false;
  for (Function &F : M) {
    SmallVector<InvokeInst *> Invokes;
    for (BasicBlock &BB : F)
      if (InvokeInst *Invoke = dyn_cast<InvokeInst>(BB.getTerminator()))
        Invokes.push_back(Invoke);
    if (Invokes.empty())
      continue;
    auto &LI = IIRB.analysisGetter<LoopAnalysis>(F);
    auto &DT = IIRB.analysisGetter<DominatorTreeAnalysis>(F);
    DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);
    for (InvokeInst *Invoke : Invokes) {
      BasicBlock *InvokeBB = Invoke->getParent();
      BasicBlock *Normal = Invoke->getNormalDest();
      BasicBlock *BeforeNormal = splitBlockBefore(Normal, Normal->begin(), &DTU,
                                                  &LI, /*MSSAU*/ nullptr);
      BeforeNormal->setName("invoke.split");
      Invoke->setNormalDest(BeforeNormal);
      Normal->replacePhiUsesWith(InvokeBB, BeforeNormal);

      Changed = true;
    }
  }
  return Changed;
}

bool InstrumentorImpl::instrument() {
  bool Changed = false;
  if (!shouldInstrumentTarget())
    return Changed;

  for (auto &ChoiceIt :
       IConf.IChoices[InstrumentationLocation::INSTRUCTION_PRE])
    if (ChoiceIt.second->Enabled)
      InstChoicesPRE[ChoiceIt.second->getOpcode()] = ChoiceIt.second;
  for (auto &ChoiceIt :
       IConf.IChoices[InstrumentationLocation::INSTRUCTION_POST])
    if (ChoiceIt.second->Enabled)
      InstChoicesPOST[ChoiceIt.second->getOpcode()] = ChoiceIt.second;

  Changed |= prepareModule();
  Changed |= instrumentModule();

  for (Function &Fn : M)
    Changed |= instrumentFunction(Fn);

  linkRuntime();

  return Changed;
}

BasicBlock::iterator
InstrumentorIRBuilderTy::getBestHoistPoint(BasicBlock::iterator IP,
                                           HoistKindTy HoistKind) {
  switch (HoistKind) {
  case DO_NOT_HOIST:
    return IP;
  case HOIST_IN_BLOCK:
    return IP.getNodeParent()->getFirstNonPHIOrDbgOrAlloca();
  case HOIST_OUT_OF_LOOPS: {
    auto BlockIP = IP.getNodeParent()->getFirstNonPHIOrDbgOrAlloca();
    auto &LI = analysisGetter<LoopAnalysis>(*BlockIP->getFunction());
    auto *L = LI.getLoopFor(BlockIP->getParent());
    while (L) {
      if (L->getLoopPreheader())
        IP = L->getLoopPreheader()->getFirstNonPHIOrDbgOrAlloca();
      L = L->getParentLoop();
    }
    return IP;
  }
  case HOIST_MAXIMALLY:
    return IP->getFunction()->getEntryBlock().getFirstNonPHIOrDbgOrAlloca();
  }
  llvm_unreachable("Unknown kind!");
}

BasicBlock::iterator InstrumentorIRBuilderTy::hoistInstructionsAndAdjustIP(
    Instruction &InitialI, BasicBlock::iterator IP, DominatorTree &DT,
    bool ForceInitial) {
  if (DT.dominates(&InitialI, IP))
    return IP;

  SmallVector<Instruction *> Stack;
  SmallPtrSet<Instruction *, 8> UnmovableInst;
  SmallVector<Instruction *> Worklist;
  Worklist.push_back(&InitialI);

  while (!Worklist.empty()) {
    auto *I = Worklist.pop_back_val();
    if (isa<PHINode>(I)) {
      UnmovableInst.insert(I);
      continue;
    }
    if ((!ForceInitial || I != &InitialI) &&
        (I->mayHaveSideEffects() || I->mayReadFromMemory())) {
      UnmovableInst.insert(I);
      continue;
    }

    Stack.push_back(I);
    for (auto *Op : I->operand_values())
      if (auto *OpI = dyn_cast<Instruction>(Op))
        Worklist.push_back(OpI);
  }

  for (auto *I : UnmovableInst) {
    if (!DT.dominates(I, IP))
      IP = *I->getInsertionPointAfterDef();
  }

  UnreachableInst *UI = new UnreachableInst(InitialI.getContext(), IP);
  IP = UI->getIterator();

  SmallPtrSet<Instruction *, 8> Seen;
  for (auto *I : reverse(Stack)) {
    if (!Seen.insert(I).second)
      continue;
    if (UnmovableInst.count(I))
      continue;
    if (DT.dominates(I, IP))
      continue;
    I->moveBefore(IP);
  }

  IP = std::next(IP);
  UI->eraseFromParent();

  return IP;
}

std::pair<BasicBlock::iterator, bool>
InstrumentorIRBuilderTy::computeLoopRangeValues(Value &V,
                                                uint32_t AdditionalSize) {
  LoopRangeInfo &LRI = LoopRangeInfoMap[&V];
  if (LRI.Min) {
    LRI.AdditionalSize = std::max(LRI.AdditionalSize, AdditionalSize);
    // TODO: This is not manifested in IR
    return {BasicBlock::iterator(), false};
  }
  auto *BB = IRB.GetInsertBlock();
  auto *Fn = BB->getParent();
  auto &SE = analysisGetter<ScalarEvolutionAnalysis>(*Fn);
  auto &LI = analysisGetter<LoopAnalysis>(*Fn);
  auto *BBLoop = LI.getLoopFor(BB);
  if (!BBLoop) {
    LLVM_DEBUG(errs() << " - value not in a loop " << V << "\n");
    return {BasicBlock::iterator(), false};
  }
  // TODO: This is a hack since SCEV somehow remembers values that we replaced.
  //  SE.forgetAllLoops();
  auto *VSCEV = SE.getSCEVAtScope(&V, BBLoop);
  if (isa<SCEVCouldNotCompute>(VSCEV)) {
    LLVM_DEBUG(errs() << " - loop evaluation not computable for " << V << "\n");
    return {BasicBlock::iterator(), false};
  }

  SmallPtrSet<const Loop *, 8> Loops;
  SE.getUsedLoops(VSCEV, Loops);
  if (Loops.empty()) {
    LLVM_DEBUG(errs() << " - no loops in value " << *VSCEV << "\n");
    return {BasicBlock::iterator(), false};
  }

  SmallVector<const Loop *, 8> LoopsOrdered;
  LoopsOrdered.resize(BBLoop->getLoopDepth() + 1);
  for (auto *L : Loops) {
    if (!L->contains(BBLoop))
      continue;
    auto Depth = L->getLoopDepth();
    assert(!LoopsOrdered[Depth]);
    LoopsOrdered[Depth] = L;
  }

  auto *PointerSCEV = SE.getPointerBase(VSCEV);
  if (PointerSCEV != VSCEV)
    VSCEV = SE.removePointerBase(VSCEV);
  auto *FirstSCEV = VSCEV;
  auto *LastSCEV = VSCEV;
  auto *Ty = VSCEV->getType();
  int32_t I = LoopsOrdered.size() - 1;

  auto &DT = analysisGetter<DominatorTreeAnalysis>(*Fn);
  auto IP = IRB.GetInsertPoint();

  for (; I >= 0; --I) {
    auto *L = LoopsOrdered[I];
    if (!L)
      continue;
    auto *ExitingBB = L->getExitingBlock();
    if (!ExitingBB)
      continue;
    if (!SE.hasComputableLoopEvolution(FirstSCEV, L) ||
        !SE.hasComputableLoopEvolution(LastSCEV, L) ||
        !SE.hasLoopInvariantBackedgeTakenCount(L) ||
        isa<SCEVCouldNotCompute>(SE.getBackedgeTakenCount(L))) {
      LLVM_DEBUG(errs() << " -- loop at depth " << I
                        << " not computable:" << *FirstSCEV << " : "
                        << *LastSCEV << " in " << L->getName() << "\n");
      break;
    }
    if (IP->getParent() != ExitingBB &&
        !DT.dominates(IP->getParent(), ExitingBB))
      break;

    LoopToScevMapT FirstL2SMap, LastL2SMap;
    FirstL2SMap[L] = SE.getZero(Ty);
    LastL2SMap[L] = SE.getBackedgeTakenCount(L);
    SmallPtrSet<const Loop *, 8> BTCLoops;
    SE.getUsedLoops(LastL2SMap[L], BTCLoops);
    for (auto *L : BTCLoops) {
      auto Depth = L->getLoopDepth();
      if (Depth < LoopsOrdered.size() && LoopsOrdered[Depth] == L)
        LoopsOrdered[Depth] = nullptr;
    }
    LLVM_DEBUG(errs() << "L -> " << *FirstL2SMap[L] << " : " << *LastL2SMap[L]
                      << "\n");
    FirstSCEV = SCEVLoopAddRecRewriter::rewrite(FirstSCEV, FirstL2SMap, SE);
    LastSCEV = SCEVLoopAddRecRewriter::rewrite(LastSCEV, LastL2SMap, SE);
  }
  if (LastSCEV == VSCEV) {
    LLVM_DEBUG(errs() << " - outermost loop not computable " << *VSCEV << "\n");
    return {BasicBlock::iterator(), false};
  }

  SCEVExpander Expander(SE, DL, ".vrange");
  assert(isa<LoadInst>(IP) || isa<StoreInst>(IP));
  Expander.setInsertPoint(IP);

  auto &TTI = analysisGetter<TargetIRAnalysis>(*Fn);
  auto *ExpansionLoop = I >= 0 ? const_cast<Loop *>(LoopsOrdered[I]) : nullptr;
  if (ExpansionLoop)
    if (Expander.isHighCostExpansion({FirstSCEV, LastSCEV}, ExpansionLoop, 7,
                                     &TTI, &*IP))
      return {BasicBlock::iterator(), false};

  Value *FirstVal = Expander.expandCodeFor(FirstSCEV, Ty);
  Value *LastVal = Expander.expandCodeFor(LastSCEV, Ty);
  auto *TmpVal = FirstVal;
  FirstVal = IRB.CreateIntrinsic(Intrinsic::umin, {Ty}, {TmpVal, LastVal});
  LastVal = IRB.CreateIntrinsic(Intrinsic::umax, {Ty}, {TmpVal, LastVal});

  if (PointerSCEV != VSCEV) {
    auto *Ptr = Expander.expandCodeFor(PointerSCEV, PointerSCEV->getType());
    FirstVal = IRB.CreatePtrAdd(Ptr, FirstVal);
    LastVal = IRB.CreatePtrAdd(Ptr, LastVal);
  }

  IP = getBestHoistPoint(IP, HOIST_OUT_OF_LOOPS);
  if (auto *FirstValI = dyn_cast<Instruction>(FirstVal))
    IP = hoistInstructionsAndAdjustIP(*FirstValI, IP, DT);
  if (auto *LastValI = dyn_cast<Instruction>(LastVal))
    IP = hoistInstructionsAndAdjustIP(*LastValI, IP, DT);

  //  append_range(GeneratedInsts, Expander.getAllInsertedInstructions());

  LRI = {FirstVal, LastVal, AdditionalSize};
  return {IP, true};
}

bool InstrumentorIRBuilderTy::isKnownDereferenceableAccess(
    Instruction &I, Value &Ptr, uint32_t AccessSize) {
  auto &TLI = analysisGetter<TargetLibraryAnalysis>(*I.getFunction());
  uint64_t Size;
  if (!getObjectSize(&Ptr, Size, DL, &TLI))
    return false;
  return Size >= AccessSize;
}

PreservedAnalyses InstrumentorPass::run(Module &M, ModuleAnalysisManager &MAM) {
  InstrumentationConfig &IConf =
      UserIConf ? *UserIConf : *new InstrumentationConfig();
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  InstrumentorIRBuilderTy &IIRB =
      UserIIRB ? *UserIIRB : *new InstrumentorIRBuilderTy(M, FAM);
  InstrumentorImpl Impl(IConf, IIRB, M, FAM);
  if (IConf.ReadConfig && !readInstrumentorConfigFromJSON(IConf))
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

void InstrumentationConfig::populate(InstrumentorIRBuilderTy &IIRB) {
  /// List of all instrumentation opportunities.
  LoopValueRangeIO::populate(*this, IIRB);
  UnreachableIO::populate(*this, IIRB.Ctx);
  BasePointerIO::populate(*this, IIRB.Ctx);
  FunctionIO::populate(*this, IIRB.Ctx);
  PtrToIntIO::populate(*this, IIRB.Ctx);
  ModuleIO::populate(*this, IIRB.Ctx);
  GlobalIO::populate(*this, IIRB.Ctx);
  AllocaIO::populate(*this, IIRB.Ctx);
  BranchIO::populate(*this, IIRB.Ctx);
  StoreIO::populate(*this, IIRB);
  LoadIO::populate(*this, IIRB);
  CallIO::populate(*this, IIRB.Ctx);
  ICmpIO::populate(*this, IIRB.Ctx);
  VAArgIO::populate(*this, IIRB.Ctx);
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

Value *
InstrumentationConfig::getBasePointerInfo(Value &V,
                                          InstrumentorIRBuilderTy &IIRB) {
  Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();

  Value *VPtr;
  {
    Value *&UnderlyingVPtr = UnderlyingObjsMap[&V];
    if (!UnderlyingVPtr)
      UnderlyingVPtr = getUnderlyingObjectRecursive(&V);
    VPtr = UnderlyingVPtr;
  }

  Value *&BPI = BasePointerInfoMap[{VPtr, Fn}];
  if (!BPI) {
    auto *BPIO =
        IChoices[InstrumentationLocation::SPECIAL_VALUE]["base_pointer_info"];
    if (!BPIO || !BPIO->Enabled) {
      errs() << "WARNING: Base pointer info disabled but required, passing "
                "nullptr.\n";
      return BPI = Constant::getNullValue(IIRB.PtrTy);
    }
    IRBuilderBase::InsertPointGuard IP(IIRB.IRB);
    if (auto *BasePtrI = dyn_cast<Instruction>(VPtr))
      IIRB.IRB.SetInsertPoint(*BasePtrI->getInsertionPointAfterDef());
    else if (isa<Constant>(VPtr) || isa<Argument>(VPtr))
      IIRB.IRB.SetInsertPointPastAllocas(
          IIRB.IRB.GetInsertBlock()->getParent());
    else {
      VPtr->dump();
      llvm_unreachable("Unexpected base pointer!");
    }
    ensureDbgLoc(IIRB.IRB);

    // Use fresh caches for safety, as this function may be called from
    // another instrumentation opportunity.
    InstrumentationCaches ICaches;
    BPI = BPIO->instrument(VPtr, *this, IIRB, ICaches);
    IIRB.returnAllocas();
    if (!BPI)
      return BPI = Constant::getNullValue(BPIO->getRetTy(IIRB.Ctx));
  }
  return BPI;
}

Value *InstrumentationConfig::getLoopValueRange(Value &V,
                                                InstrumentorIRBuilderTy &IIRB,
                                                uint32_t AdditionalSize) {
  Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();
  auto &SE = IIRB.analysisGetter<ScalarEvolutionAnalysis>(*Fn);

  Value *VPtr = &V;
  Value *&LVR = LoopValueRangeMap[{SE.getSCEV(VPtr), Fn}];
  if (!LVR) {
    auto *LVRIO =
        IChoices[InstrumentationLocation::SPECIAL_VALUE]["loop_value_range"];
    if (!LVRIO || !LVRIO->Enabled) {
      errs() << "WARNING: Loop value range disabled but required, passing "
                "nullptr.\n";
      return LVR = Constant::getNullValue(IIRB.PtrTy);
    }

    // Use fresh caches for safety, as this function may be called from
    // another instrumentation opportunity.
    InstrumentationCaches ICaches;
    static_cast<LoopValueRangeIO *>(LVRIO)->setAdditionalSize(AdditionalSize);
    LVR = LVRIO->instrument(VPtr, *this, IIRB, ICaches);
    IIRB.returnAllocas();
    if (!LVR)
      return LVR = Constant::getNullValue(LVRIO->getRetTy(IIRB.Ctx));
  }
  return LVR;
}

// Converts instruction opportunity epoch to sequential ID
static int32_t epocheToId(uint32_t Epoche) {
  static DenseMap<uint32_t, int32_t> EpocheIdMap;
  static int32_t ID = 0;
  int32_t &EpochId = EpocheIdMap[Epoche];
  if (EpochId == 0)
    EpochId = ++ID;
  return EpochId;
}

Value *InstrumentationOpportunity::getIdPre(Value &V, Type &Ty,
                                            InstrumentationConfig &IConf,
                                            InstrumentorIRBuilderTy &IIRB) {
  return getCI(&Ty, epocheToId(IIRB.Epoche));
}

Value *InstrumentationOpportunity::getIdPost(Value &V, Type &Ty,
                                             InstrumentationConfig &IConf,
                                             InstrumentorIRBuilderTy &IIRB) {
  return getCI(&Ty, -epocheToId(IIRB.Epoche));
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
    ensureDbgLoc(IIRB.IRB);
    NewVCasted = tryToCast(IIRB.IRB, &NewV, V.getType(), IIRB.DL,
                           /*AllowTruncate=*/true);
  }
  V.replaceUsesWithIf(NewVCasted, [&](Use &U) {
    if (IIRB.NewInsts.lookup(cast<Instruction>(U.getUser())) == IIRB.Epoche)
      return false;
    if (isa<LifetimeIntrinsic>(U.getUser()) || U.getUser()->isDroppable())
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
  if (RetTy)
    IndirectReturnValue = DirectReturnValue = "0";
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
    auto UserRetTy = getAsCType(RetTy, 0).first;
    assert((DirectRetTy == UserRetTy || DirectRetTy == "void ") &&
           (IndirectRetTy == UserRetTy || IndirectRetTy == "void ") &&
           "Explicit return type but also implicit one!");
    IndirectRetTy = DirectRetTy = UserRetTy;
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
      if (!RetTy && NumReplaceableArgs == 1 && (It.Flags & IRTArg::REPLACABLE))
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
                                             const DataLayout &DL,
                                             InstrumentationCaches &ICaches) {
  SmallVector<Value *> CallParams;

  IRBuilderBase::InsertPointGuard IRP(IIRB.IRB);
  auto IP = IIRB.getBestHoistPoint(IIRB.IRB.GetInsertPoint(), IO.HoistKind);

  bool ForceIndirection = RequiresIndirection;
  for (auto &It : IO.IRTArgs) {
    if (!It.Enabled)
      continue;
    auto *&Param = ICaches.DirectArgCache[{IIRB.Epoche, IO.getName(), It.Name}];
    if (!Param || It.NoCache)
      // Avoid passing the caches to the getter.
      Param = It.GetterCB(*V, *It.Ty, IConf, IIRB);
    if (!Param)
      errs() << IO.getName() << " : " << It.Name << "\n";
    assert(Param);

    if (Param->getType()->isVoidTy()) {
      Param = Constant::getNullValue(It.Ty);
    } else if (Param->getType()->isAggregateType() ||
               DL.getTypeSizeInBits(Param->getType()) >
                   DL.getTypeSizeInBits(It.Ty)) {
      if (!isPotentiallyIndirect(It)) {
        errs() << "WARNING: Indirection needed for " << It.Name << " of " << *V
               << " in " << IO.getName() << ", but not indicated\n. Got "
               << *Param << " expected " << *It.Ty
               << "; instrumentation is skipped";
        return nullptr;
      }
      ForceIndirection = true;
    } else {
      Param = tryToCast(IIRB.IRB, Param, It.Ty, DL);
    }
    if (IO.HoistKind != DO_NOT_HOIST)
      if (auto *ParamI = dyn_cast<Instruction>(Param)) {
        auto &DT =
            IIRB.analysisGetter<DominatorTreeAnalysis>(*ParamI->getFunction());
        IP = IIRB.hoistInstructionsAndAdjustIP(*ParamI, IP, DT);
      }
    CallParams.push_back(Param);
  }

  if (ForceIndirection) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();

    unsigned Offset = 0;
    for (auto &It : IO.IRTArgs) {
      if (!It.Enabled)
        continue;

      if (!isPotentiallyIndirect(It)) {
        ++Offset;
        continue;
      }
      auto *&CallParam = CallParams[Offset++];
      if (!(It.Flags & IRTArg::INDIRECT_HAS_SIZE)) {
        CallParams.insert(&CallParam + 1, IIRB.IRB.getInt32(DL.getTypeStoreSize(
                                              CallParam->getType())));
        Offset += 1;
      }

      auto *&CachedParam =
          ICaches.IndirectArgCache[{IIRB.Epoche, IO.getName(), It.Name}];
      if (CachedParam) {
        CallParam = CachedParam;
        continue;
      }

      auto *AI = IIRB.getAlloca(Fn, CallParam->getType());
      IIRB.IRB.CreateStore(CallParam, AI);
      CallParam = CachedParam = AI;
    }
  }

  if (!ForceIndirection)
    IIRB.IRB.SetInsertPoint(IP);
  ensureDbgLoc(IIRB.IRB);

  auto *FnTy =
      createLLVMSignature(IConf, V->getContext(), DL, ForceIndirection);
  auto CompleteName =
      IConf.getRTName(IO.IP.isPRE() ? "pre_" : "post_", IO.getName(),
                      ForceIndirection ? "_ind" : "");
  auto FC = IIRB.IRB.GetInsertBlock()->getModule()->getOrInsertFunction(
      CompleteName, FnTy);
  auto *CI = IIRB.IRB.CreateCall(FC, CallParams);
  CI->addFnAttr(Attribute::get(IIRB.Ctx, Attribute::WillReturn));

  for (unsigned I = 0, E = IO.IRTArgs.size(); I < E; ++I) {
    if (!IO.IRTArgs[I].Enabled)
      continue;
    if (!isReplacable(IO.IRTArgs[I]))
      continue;
    bool IsCustomReplaceable = IO.IRTArgs[I].Flags & IRTArg::REPLACABLE_CUSTOM;
    Value *NewValue = FnTy->isVoidTy() || IsCustomReplaceable
                          ? ICaches.DirectArgCache[{IIRB.Epoche, IO.getName(),
                                                    IO.IRTArgs[I].Name}]
                          : CI;
    assert(NewValue);
    if (ForceIndirection && !IsCustomReplaceable &&
        isPotentiallyIndirect(IO.IRTArgs[I])) {
      auto *Q = ICaches.IndirectArgCache[{IIRB.Epoche, IO.getName(),
                                          IO.IRTArgs[I].Name}];
      NewValue = IIRB.IRB.CreateLoad(V->getType(), Q);
    }
    V = IO.IRTArgs[I].SetterCB(*V, *NewValue, IConf, IIRB);
  }
  return CI;
}

template <typename Ty> constexpr Value *getValue(Ty &ValueOrUse) {
  if constexpr (std::is_same<Ty, Use>::value)
    return ValueOrUse.get();
  else
    return static_cast<Value *>(&ValueOrUse);
}

template <typename Range>
static Value *createValuePack(const Range &R, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
  auto *Fn = IIRB.IRB.GetInsertBlock()->getParent();
  auto *I32Ty = IIRB.IRB.getInt32Ty();
  SmallVector<Constant *> ConstantValues;
  SmallVector<std::pair<Value *, uint32_t>> Values;
  SmallVector<Type *> Types;
  for (auto &RE : R) {
    Value *V = getValue(RE);
    if (!V->getType()->isSized())
      continue;
    auto VSize = IIRB.DL.getTypeAllocSize(V->getType());
    ConstantValues.push_back(getCI(I32Ty, VSize));
    Types.push_back(I32Ty);
    ConstantValues.push_back(getCI(I32Ty, V->getType()->getTypeID()));
    Types.push_back(I32Ty);
    if (uint32_t MisAlign = VSize % 8) {
      Types.push_back(ArrayType::get(IIRB.Int8Ty, 8 - MisAlign));
      ConstantValues.push_back(ConstantArray::getNullValue(Types.back()));
    }
    Types.push_back(V->getType());
    if (auto *C = dyn_cast<Constant>(V)) {
      ConstantValues.push_back(C);
      continue;
    }
    Values.push_back({V, ConstantValues.size()});
    ConstantValues.push_back(Constant::getNullValue(V->getType()));
  }
  if (Types.empty())
    return ConstantPointerNull::get(PointerType::getUnqual(IIRB.Ctx));

  StructType *STy = StructType::get(Fn->getContext(), Types, /*isPacked=*/true);
  Constant *Initializer = ConstantStruct::get(STy, ConstantValues);

  GlobalVariable *&GV = IConf.ConstantGlobalsCache[Initializer];
  if (!GV)
    GV = new GlobalVariable(*Fn->getParent(), STy, false,
                            GlobalValue::InternalLinkage, Initializer,
                            IConf.getRTName("", "value_pack"));

  auto *AI = IIRB.getAlloca(Fn, STy);
  IIRB.IRB.CreateMemCpy(AI, AI->getAlign(), GV, MaybeAlign(GV->getAlignment()),
                        IIRB.DL.getTypeAllocSize(STy));
  for (auto [Param, Idx] : Values) {
    auto *Ptr = IIRB.IRB.CreateStructGEP(STy, AI, Idx);
    IIRB.IRB.CreateStore(Param, Ptr);
  }
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
  for (const auto &[Idx, RE] : enumerate(R)) {
    Value *V = getValue(RE);
    if (!V->getType()->isSized())
      continue;
    Offset += 8;
    auto VSize = DL.getTypeAllocSize(V->getType());
    auto Padding = alignTo(VSize, 8) - VSize;
    Offset += Padding;
    auto *Ptr = IIRB.IRB.CreateConstInBoundsGEP1_32(IIRB.Int8Ty, &Pack, Offset);
    auto *NewV = IIRB.IRB.CreateLoad(V->getType(), Ptr);
    SetterCB(Idx, NewV);
    Offset += VSize;
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
Value *StoreIO::getLoopValueRangeInfo(Value &V, Type &Ty,
                                      InstrumentationConfig &IConf,
                                      InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return IConf.getLoopValueRange(
      *SI.getPointerOperand(), IIRB,
      IIRB.DL.getTypeStoreSize(SI.getValueOperand()->getType()));
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
Value *LoadIO::getLoopValueRangeInfo(Value &V, Type &Ty,
                                     InstrumentationConfig &IConf,
                                     InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return IConf.getLoopValueRange(*LI.getPointerOperand(), IIRB,
                                 IIRB.DL.getTypeStoreSize(LI.getType()));
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
  auto &TLI = IIRB.analysisGetter<TargetLibraryAnalysis>(*CI.getFunction());
  auto ACI = getAllocationCallInfo(&CI, &TLI);
  if (!ACI)
    return Constant::getNullValue(&Ty);

  auto &Ctx = CI.getContext();

  StructType *STy = StructType::get(Ctx,
                                    {IIRB.PtrTy, IIRB.Int32Ty, IIRB.Int32Ty,
                                     IIRB.Int32Ty, IIRB.Int8Ty, IIRB.Int32Ty},
                                    /*isPacked=*/true);
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
  if (!Config.ArgFilter)
    return getCI(&Ty, CI.arg_size());
  auto FRange = make_filter_range(CI.args(), Config.ArgFilter);
  return getCI(&Ty, std::distance(FRange.begin(), FRange.end()));
}
Value *CallIO::getCallParameters(Value &V, Type &Ty,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
  auto &CI = cast<CallInst>(V);
  if (!Config.ArgFilter)
    return createValuePack(CI.args(), IConf, IIRB);
  return createValuePack(make_filter_range(CI.args(), Config.ArgFilter), IConf,
                         IIRB);
}
Value *CallIO::setCallParameters(Value &V, Value &NewV,
                                 InstrumentationConfig &IConf,
                                 InstrumentorIRBuilderTy &IIRB) {
  auto &CI = cast<CallInst>(V);
  auto *CIt = CI.arg_begin();
  auto CB = [&](int Idx, Value *ReplV) {
    while (Config.ArgFilter && !Config.ArgFilter(*CIt))
      ++CIt;
    // Do not replace `immarg` operands with a non-immediate.
    if (CI.getParamAttr(CIt->getOperandNo(), Attribute::ImmArg).isValid())
      return;
    CIt->set(ReplV);
    ++CIt;
  };
  if (!Config.ArgFilter)
    readValuePack(CI.args(), NewV, IIRB, CB);
  else
    readValuePack(make_filter_range(CI.args(), Config.ArgFilter), NewV, IIRB,
                  CB);
  return &CI;
}
Value *CallIO::isDefinition(Value &V, Type &Ty, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB) {
  auto &CI = cast<CallInst>(V);
  if (auto *Fn = CI.getCalledFunction())
    return getCI(&Ty, !Fn->isDeclaration());
  return getCI(&Ty, 0);
}

Value *BranchIO::isConditional(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB) {
  auto &BI = cast<BranchInst>(V);
  return getCI(&Ty, BI.isConditional());
}

Value *BranchIO::getValue(Value &V, Type &Ty, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB) {
  auto &BI = cast<BranchInst>(V);
  if (BI.isUnconditional())
    return getCI(&Ty, 1);
  return BI.getCondition();
}

Value *BranchIO::setValue(Value &V, Value &NewV, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB) {
  auto *BI = cast<BranchInst>(&V);
  if (BI->isConditional()) {
    auto *Cast = tryToCast(IIRB.IRB, &NewV, BI->getCondition()->getType(),
                           IIRB.DL, true);
    BI->setCondition(Cast);
  }
  return BI;
}

Value *BranchIO::getNumSuccessors(Value &V, Type &Ty,
                                  InstrumentationConfig &IConf,
                                  InstrumentorIRBuilderTy &IIRB) {
  auto &BI = cast<BranchInst>(V);
  return getCI(&Ty, BI.getNumSuccessors());
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

Value *VAArgIO::getPointer(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB) {
  auto &VI = cast<VAArgInst>(V);
  return VI.getPointerOperand();
}
Value *VAArgIO::setPointer(Value &V, Value &NewV, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB) {
  auto &VI = cast<VAArgInst>(V);
  VI.setOperand(VI.getPointerOperandIndex(), &NewV);
  return &VI;
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

Value *LoopValueRangeIO::getInitialLoopValue(Value &V, Type &Ty,
                                             InstrumentationConfig &IConf,
                                             InstrumentorIRBuilderTy &IIRB) {
  if (auto *IV = IIRB.getInitialLoopValue(V))
    return tryToCast(IIRB.IRB, IV, &Ty, IIRB.DL);
  return Constant::getNullValue(&Ty);
}

Value *LoopValueRangeIO::getFinalLoopValue(Value &V, Type &Ty,
                                           InstrumentationConfig &IConf,
                                           InstrumentorIRBuilderTy &IIRB) {
  if (auto *FV = IIRB.getFinalLoopValue(V))
    return tryToCast(IIRB.IRB, FV, &Ty, IIRB.DL);
  return Constant::getNullValue(&Ty);
}

Value *LoopValueRangeIO::getMaxOffset(Value &V, Type &Ty,
                                      InstrumentationConfig &IConf,
                                      InstrumentorIRBuilderTy &IIRB) {
  if (auto *FV = IIRB.getMaxOffset(V, Ty))
    return tryToCast(IIRB.IRB, FV, &Ty, IIRB.DL);
  return Constant::getNullValue(&Ty);
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
  if (!Config.ArgFilter)
    return getCI(&Ty, Fn.arg_size());
  auto FRange = make_filter_range(Fn.args(), Config.ArgFilter);
  return getCI(&Ty, std::distance(FRange.begin(), FRange.end()));
}
Value *FunctionIO::getArguments(Value &V, Type &Ty,
                                InstrumentationConfig &IConf,
                                InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  if (!Config.ArgFilter)
    return createValuePack(Fn.args(), IConf, IIRB);
  return createValuePack(make_filter_range(Fn.args(), Config.ArgFilter), IConf,
                         IIRB);
}
Value *FunctionIO::setArguments(Value &V, Value &NewV,
                                InstrumentationConfig &IConf,
                                InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  auto *AIt = Fn.arg_begin();
  auto CB = [&](int Idx, Value *ReplV) {
    while (Config.ArgFilter && !Config.ArgFilter(*AIt))
      ++AIt;
    Fn.getArg(Idx)->replaceUsesWithIf(ReplV, [&](Use &U) {
      return IIRB.NewInsts.lookup(cast<Instruction>(U.getUser())) !=
             IIRB.Epoche;
    });
    ++AIt;
  };
  if (!Config.ArgFilter)
    readValuePack(Fn.args(), NewV, IIRB, CB);
  else
    readValuePack(make_filter_range(Fn.args(), Config.ArgFilter), NewV, IIRB,
                  CB);
  return &Fn;
}
Value *FunctionIO::isMainFunction(Value &V, Type &Ty,
                                  InstrumentationConfig &IConf,
                                  InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  return getCI(&Ty, Fn.getName() == "main");
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

  GlobalVariable *ShadowGV = nullptr;
  auto ShadowName = IConf.getRTName("shadow.", GV.getName());
  auto &DL = GV.getDataLayout();
  if (GV.isDeclaration()) {
    ShadowGV = new GlobalVariable(*GV.getParent(), GV.getType(), false,
                                  GlobalVariable::WeakODRLinkage, &GV,
                                  ShadowName, &GV, GV.getThreadLocalMode(),
                                  DL.getDefaultGlobalsAddressSpace());
  } else {
    ShadowGV = new GlobalVariable(
        *GV.getParent(), NewV.getType(), false, GV.getLinkage(),
        PoisonValue::get(NewV.getType()), ShadowName, &GV);
    IIRB.IRB.CreateStore(&NewV, ShadowGV);
  }

  SmallVector<Use *> Worklist(make_pointer_range(GV.uses()));
  SmallPtrSet<Use *, 32> Done;
  DenseMap<std::pair<Value *, Function *>, Instruction *> VMap;
  DenseMap<Value *, Instruction *> ConstToInstMap;
  DenseMap<Function *, Instruction *> ReloadMap;

  auto MakeInstForConst = [&](Use &U) {
    Instruction *&I = ConstToInstMap[U];
    if (I)
      return;
    if (U == &GV) {
    } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
      I = CE->getAsInstruction();
    }
  };

  auto InsertConsts = [&](Instruction *UserI, Use &UserU) {
    SmallVector<std::pair<Instruction *, Use *>> Worklist;
    auto *&Reload = ReloadMap[UserI->getFunction()];
    if (!Reload) {
      Reload = new LoadInst(
          GV.getType(), ShadowGV, GV.getName() + ".shadow_load",
          UserI->getFunction()->getEntryBlock().getFirstNonPHIOrDbgOrAlloca());
      IIRB.NewInsts.insert({Reload, IIRB.Epoche});
    }
    Worklist.push_back({UserI, &UserU});
    while (!Worklist.empty()) {
      auto [I, U] = Worklist.pop_back_val();
      if (*U == &GV) {
        U->set(ReloadMap[I->getFunction()]);
        continue;
      }
      if (auto *CI = ConstToInstMap[*U]) {
        auto *CIClone = CI->clone();
        IIRB.NewInsts.insert({CIClone, IIRB.Epoche});
        if (auto *PHI = dyn_cast<PHINode>(I)) {
          auto *BB = PHI->getIncomingBlock(U->getOperandNo());
          CIClone->insertBefore(BB->getTerminator()->getIterator());
        } else {
          CIClone->insertBefore(I->getIterator());
        }
        U->set(CIClone);
        for (auto &CICUse : CIClone->operands()) {
          Worklist.push_back({CIClone, &CICUse});
        }
      }
    }
  };

  SmallPtrSet<Use *, 8> Visited;
  while (!Worklist.empty()) {
    Use *U = Worklist.pop_back_val();
    if (!Done.insert(U).second)
      continue;
    MakeInstForConst(*U);
    auto *I = dyn_cast<Instruction>(U->getUser());
    if (!I) {
      append_range(Worklist, make_pointer_range(U->getUser()->uses()));
      continue;
    }
    if (IIRB.NewInsts.lookup(I) == IIRB.Epoche)
      continue;
    if (isa<LandingPadInst>(I))
      continue;
    if (auto *II = dyn_cast<IntrinsicInst>(I))
      if (II->getIntrinsicID() == Intrinsic::eh_typeid_for)
        continue;
    if (I->getParent())
      InsertConsts(I, *U);
  }

  for (auto &It : ConstToInstMap)
    if (It.second)
      It.second->deleteValue();

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
Value *GlobalIO::isDefinition(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
  GlobalVariable &GV = cast<GlobalVariable>(V);
  return getCI(&Ty, !GV.isDeclaration());
}
