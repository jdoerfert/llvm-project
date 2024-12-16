//===- llvm-extract.cpp - LLVM function extraction utility ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility changes the input module to only contain a single function,
// which is primarily used for debugging transformations.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/BlockExtractor.h"
#include "llvm/Transforms/IPO/ExtractGV.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/StripSymbols.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/BreakCriticalEdges.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <memory>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "llvm-extract-loops"

cl::OptionCategory ExtractLoopsCat("llvm-extract-loops Options");

// InputFilename - The filename to read from.
static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input bitcode file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> OutputFilenamePrefix(
    "output-prefix", cl::desc("Specify output filename prefix"),
    cl::value_desc("filename prefix"), cl::init("llvm_extracted_loop."),
    cl::cat(ExtractLoopsCat));

static cl::opt<std::string> OutputFilenameSuffix(
    "output-suffix", cl::desc("Specify output filename suffix"),
    cl::value_desc("filename suffix"), cl::init(""), cl::cat(ExtractLoopsCat));

static cl::opt<bool> Force("f", cl::desc("Enable binary output on terminals"),
                           cl::cat(ExtractLoopsCat));

static cl::opt<bool> KeepConstInit("keep-const-init",
                                   cl::desc("Keep initializers of constants"),
                                   cl::init(true), cl::cat(ExtractLoopsCat));

static cl::opt<bool>
    Recursive("recursive", cl::desc("Recursively extract all called functions"),
              cl::init(true), cl::cat(ExtractLoopsCat));

static cl::opt<bool> OutputAssembly("S",
                                    cl::desc("Write output as LLVM assembly"),
                                    cl::Hidden, cl::cat(ExtractLoopsCat));

static cl::opt<bool> PreserveBitcodeUseListOrder(
    "preserve-bc-uselistorder",
    cl::desc("Preserve use-list order when writing LLVM bitcode."),
    cl::init(true), cl::Hidden, cl::cat(ExtractLoopsCat));

static cl::opt<bool> PreserveAssemblyUseListOrder(
    "preserve-ll-uselistorder",
    cl::desc("Preserve use-list order when writing LLVM assembly."),
    cl::init(false), cl::Hidden, cl::cat(ExtractLoopsCat));

static cl::opt<bool>
    PrettyPrintJson("pretty-print-json",
                    cl::desc("Pretty print the loop metadata json"),
                    cl::init(false), cl::cat(ExtractLoopsCat));

static void writeLoopMetadata(LoopInfo &LI, ScalarEvolution &SE, Loop &L,
                              DenseMap<Loop *, unsigned> &LoopIDs,
                              std::string Filename) {
  std::string TripCount = "unknown";
  if (SE.getSmallConstantTripCount(&L)) {
    TripCount = "constant";
  } else {
    auto LoopBounds = L.getBounds(SE);
    if (LoopBounds && LoopBounds->getStepValue())
      TripCount = "dynamic";
  }

  int ParentID = -1;
  Loop *ParentL = L.getParentLoop();
  if (ParentL)
    ParentID = LoopIDs[ParentL];
  SmallVector<Loop *> InnerLoops;
  L.getInnerLoopsInPreorder(L, InnerLoops);
  unsigned NumInnerLoops = InnerLoops.size();

  json::Object Json({{"loop_depth", L.getLoopDepth()},
                     {"loop_trip_count", TripCount},
                     {"parent_function", L.getHeader()->getParent()->getName()},
                     {"loop_id", LoopIDs[&L]},
                     {"parent_loop_id", ParentID},
                     {"num_inner_loops", NumInnerLoops}});

  std::error_code EC;
  ToolOutputFile Out(Filename, EC, sys::fs::OF_None);
  json::OStream JOS(Out.os(), PrettyPrintJson ? 2 : 0);
  JOS.value(std::move(Json));
  Out.os() << '\n';
  Out.keep();
}

static void writeExtractedModuleInPlace(Module &M, Function &F,
                                        std::string Filename) {
  F.setName("__llvm_extracted_loop");

  SetVector<GlobalValue *> GVs;
  GVs.insert(&F);

  if (Recursive) {
    std::vector<llvm::Function *> Workqueue;
    for (GlobalValue *GV : GVs) {
      if (auto *F = dyn_cast<Function>(GV)) {
        Workqueue.push_back(F);
      }
    }
    while (!Workqueue.empty()) {
      Function *F = &*Workqueue.back();
      Workqueue.pop_back();
      for (auto &BB : *F) {
        for (auto &I : BB) {
          CallBase *CB = dyn_cast<CallBase>(&I);
          if (!CB)
            continue;
          Function *CF = CB->getCalledFunction();
          if (!CF)
            continue;
          if (CF->isDeclaration() || !GVs.insert(CF))
            continue;
          Workqueue.push_back(CF);
        }
      }
    }
  }
  std::vector<GlobalValue *> Gvs(GVs.begin(), GVs.end());
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassBuilder PB;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager PM;
  PM.addPass(ExtractGVPass(Gvs, false, KeepConstInit));

  std::error_code EC;
  ToolOutputFile Out(Filename, EC, sys::fs::OF_None);
  if (OutputAssembly)
    PM.addPass(PrintModulePass(Out.os(), "", PreserveAssemblyUseListOrder));
  else if (Force || !CheckBitcodeOutputToConsole(Out.os()))
    PM.addPass(BitcodeWriterPass(Out.os(), PreserveBitcodeUseListOrder));

  PM.run(M, MAM);
  Out.keep();
}

void preprocessModule(Module &M) {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassBuilder PB;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  FunctionPassManager FPM;
  FPM.addPass(SROAPass(SROAOptions::PreserveCFG));
  FPM.addPass(InstCombinePass());
  FPM.addPass(SimplifyCFGPass());
  // Simplify is needed in order to get dedicated exit blocks in the loops,
  // which in turn enables outlining.
  FPM.addPass(LoopSimplifyPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  MPM.run(M, MAM);
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  LLVMContext Context;
  cl::HideUnrelatedOptions(ExtractLoopsCat);
  cl::ParseCommandLineOptions(argc, argv, "llvm loop extractor\n");

  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);

  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }
  if (OutputFilenameSuffix.getNumOccurrences() == 0) {
    if (OutputAssembly)
      OutputFilenameSuffix = ".ll";
    else
      OutputFilenameSuffix = ".bc";
  }

  preprocessModule(*M);
  LLVM_DEBUG(llvm::dbgs() << "After preprocessing: \n" << *M << "\n");

  SmallVector<Function *> ToHandle;
  for (Function &F : *M)
    ToHandle.push_back(&F);

  TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M->getDataLayout());
  unsigned LoopCounter = 0;
  for (Function *F : ToHandle) {
    if (F->isDeclaration())
      continue;

    DominatorTree DT(*F);
    LoopInfo LI(DT);
    AssumptionCache AC(*F);
    ScalarEvolution SE(*F, TLI, AC, DT, LI);

    DenseMap<Loop *, unsigned> LoopIDs;
    unsigned LoopInFunctionCounter = 0;
    for (Loop *L : LI.getLoopsInPreorder()) {
      LLVM_DEBUG(L->dump());
      llvm::ValueToValueMapTy VMap;
      // FIXME preferrably, we want to analyze the module and clone only the GVs
      // that we need. However, we currently clone the entire module so that we
      // can reuse the ExtractGVPass which deletes the unnecessary GVs.
      auto ClonedM = CloneModule(*M, VMap);
      Function *NewF = cast<Function>(VMap[F]);
      SmallVector<BasicBlock *> BBs;
      for (BasicBlock *BB : L->getBlocks())
        BBs.push_back(cast<BasicBlock>(VMap[BB]));
      std::string Suffix = "llvm_extracted_loop." + std::to_string(LoopCounter);
      auto CE =
          CodeExtractor(BBs, /*DT=*/nullptr, /*AggregateArgs=*/false,
                        /*BFI=*/nullptr, /*BPI=*/nullptr, /*AC=*/nullptr,
                        /*AllowVarArgs=*/true, /*AllowAlloca=*/true,
                        /*AllocationBlock=*/nullptr,
                        /*Suffix=*/Suffix, /*ArgsInZeroAddressSpace=*/false);
      CodeExtractorAnalysisCache CEAC(*NewF);
      Function *OutlinedF = CE.extractCodeRegion(CEAC);

      std::string ModuleFilename = OutputFilenamePrefix +
                                   std::to_string(LoopCounter) +
                                   OutputFilenameSuffix;
      writeExtractedModuleInPlace(*ClonedM, *OutlinedF, ModuleFilename);

      LoopIDs[L] = LoopInFunctionCounter;
      std::string JSONFilename = ModuleFilename + ".json";
      writeLoopMetadata(LI, SE, *L, LoopIDs, JSONFilename);

      LoopInFunctionCounter++;
      LoopCounter++;
    }
  }

  return 0;
}
