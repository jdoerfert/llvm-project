// LLVM Instrumentor stub runtime

#include <stdio.h>

#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TimeProfiler.h"

extern "C" {

struct __init_ty {
  __init_ty() {
    llvm::timeTraceProfilerInitialize(10, "function profiler", true);
    llvm::timeTraceProfilerBegin("<init>", "");
  }
  ~__init_ty() {
    if (has_main) {
      llvm::timeTraceProfilerEnd();
    }
    llvm::timeTraceProfilerEnd();
    if (auto Err = llvm::timeTraceProfilerWrite("prof.json", "prof.alt.json"))
      printf("Error writing out the time trace: %s\n",
             llvm::toString(std::move(Err)).c_str());
    llvm::timeTraceProfilerCleanup();
  }
  void *callee = nullptr;
  bool callee_found = false;
  bool has_main = false;
} __state;

void __instrumentor_pre_function(void *address, char *name) {
  if (__state.callee == address && !__state.callee_found) {
    llvm::timeTraceProfilerBegin(llvm::demangle(name), "");
    __state.callee_found = true;
  }
  if (!memcmp(name, "main", 4)) {
    __state.has_main = true;
    llvm::timeTraceProfilerBegin("main", "");
  }
}

void __instrumentor_pre_call(void *callee, char *callee_name) {
  llvm::timeTraceProfilerBegin(
      callee_name ? llvm::demangle(callee_name) : "<indirect>", "");
  if (!callee_name)
    __state.callee = callee;
}
void __instrumentor_post_call(void *callee, char *callee_name) {
  if (__state.callee_found) {
    __state.callee = nullptr;
    __state.callee_found = false;
    llvm::timeTraceProfilerEnd();
  }
  llvm::timeTraceProfilerEnd();
}
}
