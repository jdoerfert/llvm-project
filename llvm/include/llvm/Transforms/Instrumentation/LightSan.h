//===- Transforms/Instrumentation/LightSan.h ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass to add sanitization to executables using the Instrumentor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_LIGHTSAN_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_LIGHTSAN_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class LightSanPass : public PassInfoMixin<LightSanPass> {
public:
  LightSanPass(){};
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_LIGHTSAN_H
