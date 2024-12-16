#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"

using namespace llvm;

#define TOOL_NAME "inputgen-mbb-pgo-info"

static cl::opt<std::string>
    BCFilePath("bc-path",
               cl::desc("Bitcode file to read from to obtain PGO info for"),
               cl::init(""));

static cl::opt<std::string>
    ProfilePath("profile-path",
                cl::desc("Path to the instrumented profile to use"),
                cl::init(""));

static ExitOnError ExitOnErr(TOOL_NAME " error: ");

class FrequencyProcessorPass : public PassInfoMixin<FrequencyProcessorPass> {
public:
  FrequencyProcessorPass(DenseMap<StringRef, BitVector> &Frequencies);

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);

private:
  DenseMap<StringRef, BitVector> &FunctionFrequencies;
};

FrequencyProcessorPass::FrequencyProcessorPass(
    DenseMap<StringRef, BitVector> &Frequencies)
    : FunctionFrequencies(Frequencies) {}

PreservedAnalyses FrequencyProcessorPass::run(Function &F,
                                              FunctionAnalysisManager &FAM) {
  BitVector BBsCalled;
  const BlockFrequencyInfo &BFI = FAM.getResult<BlockFrequencyAnalysis>(F);
  for (const BasicBlock &Block : F) {
    std::optional<uint64_t> Count = BFI.getBlockProfileCount(&Block);
    if (!Count.has_value() || *Count == 0)
      BBsCalled.push_back(false);
    else
      BBsCalled.push_back(true);
  }
  FunctionFrequencies[F.getName()] = std::move(BBsCalled);
  return PreservedAnalyses::all();
}

int main(int Argc, char **Argv) {
  cl::ParseCommandLineOptions(Argc, Argv, TOOL_NAME);

  LLVMContext Context;

  SMDiagnostic ParseError;
  std::unique_ptr<llvm::Module> IRModule =
      parseIRFile(BCFilePath, ParseError, Context);

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

  MPM.addPass(PGOInstrumentationUse(ProfilePath));

  DenseMap<StringRef, BitVector> FunctionBlockFrequencies;
  MPM.addPass(createModuleToFunctionPassAdaptor(
      FrequencyProcessorPass(FunctionBlockFrequencies)));

  MPM.run(*IRModule, MAM);

  JSONScopedPrinter Printer(outs(), true);

  Printer.arrayBegin("Functions");
  for (const auto &Function : FunctionBlockFrequencies) {
    Printer.objectBegin(Function.first);
    Printer.printNumber("NumBlocks", Function.second.size());
    Printer.printNumber("NumBlocksExecuted", Function.second.count());
    Printer.objectEnd();
  }
  Printer.arrayEnd();

  return 0;
}
