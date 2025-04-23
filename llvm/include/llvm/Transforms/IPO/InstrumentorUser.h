//===- Transforms/IPO/InstrumentorUser.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An example pass that uses Instrumentor internally.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INSTRUMENTOR_USER_H
#define LLVM_TRANSFORMS_IPO_INSTRUMENTOR_USER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class InstrumentorUserPass : public PassInfoMixin<InstrumentorUserPass> {
public:
  InstrumentorUserPass() {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INSTRUMENTOR_USER_H
