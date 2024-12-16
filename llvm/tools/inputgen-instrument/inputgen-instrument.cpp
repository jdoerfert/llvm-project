
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/InputGenerationImpl.h"
#include "llvm/Transforms/Instrumentation/InstrProfiling.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <cstdio>
#include <memory>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

#define DEBUG_TYPE "inputgen-instrument-tool"

using namespace llvm;

cl::OptionCategory InputGenCategory("input-gen Options");

static cl::opt<std::string> ClOutputPrefix("output-prefix", cl::Required,
                                           cl::cat(InputGenCategory));

static cl::opt<std::string> ClInputFilename(cl::Positional, cl::init("-"),
                                            cl::desc("Input file"),
                                            cl::cat(InputGenCategory));

static cl::opt<std::string> ClFunction("function", cl::cat(InputGenCategory));

static cl::opt<bool> ClStripDebugInfo("strip-debug-info",
                                      cl::cat(InputGenCategory));

static cl::opt<bool> ClStripUnknownOperandBundles("strip-debug-info",
                                                  cl::cat(InputGenCategory));

#define TOOL_NAME "inputgen-instrument"

class InputGenOrchestration {
private:
  const char *Argv0;
  Module &M;

public:
  InputGenOrchestration(const char *Argv0, Module &M) : Argv0(Argv0), M(M) {}
  void genFunctionForAllRuntimes(std::string FunctionName) {}
  void genAllFunctionForAllRuntimes() {}
};

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions(InputGenCategory);
  cl::ParseCommandLineOptions(argc, argv, TOOL_NAME);

  ExitOnError ExitOnErr(TOOL_NAME " error: ");
  LLVMContext Context;

  SMDiagnostic Diag;
  std::unique_ptr<Module> M = parseIRFile(ClInputFilename, Diag, Context);
  if (!M) {
    Diag.print(TOOL_NAME, errs());
    return 1;
  }

  if (ClStripDebugInfo)
    StripDebugInfo(*M);
  // if (ClStripUnknownOperandBundles)
  //   stripUnknownOperandBundles(*M);

  InputGenOrchestration IGO(argv[0], *M);

  if (ClFunction.getNumOccurrences() > 0) {
    IGO.genFunctionForAllRuntimes(ClFunction);
  } else {
    IGO.genAllFunctionForAllRuntimes();
  }

  return 0;
}
