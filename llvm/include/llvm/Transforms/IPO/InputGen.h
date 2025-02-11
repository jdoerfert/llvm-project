//===- Transforms/IPO/InputGen.h ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass to create executables that generate inputs for static code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INPUTGEN_H
#define LLVM_TRANSFORMS_IPO_INPUTGEN_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class InputGenPass : public PassInfoMixin<InputGenPass> {

public:
  InputGenPass() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INPUTGEN_H
