//===- ImplementsAttrResolver.cpp - Repl. specifications w implemenations -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/ImplementsAttrResolver.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/PassInstrumentation.h"

#define DEBUG_TYPE "implements-attr-resolver"

using namespace llvm;

PreservedAnalyses ImplementsAttrResolverPass::run(Module &M,
                                                  ModuleAnalysisManager &) {
  for (Function &Impl : M) {
    const Attribute &A = Impl.getFnAttribute("implements");
    if (!A.isValid())
      continue;

    const StringRef SpecificationName = A.getValueAsString();
    Function *Specification = M.getFunction(SpecificationName);
    if (!Specification) {
      LLVM_DEBUG(dbgs() << "Found implementation '" << Impl.getName()
                        << "' but no matching specification with name '"
                        << SpecificationName
                        << "', potentially inlined and/or eliminated.\n");
      continue;
    }
    LLVM_DEBUG(dbgs() << "Replace specification '" << Specification->getName()
                      << "' with implementation '" << Impl.getName() << "'\n");
    Specification->replaceAllUsesWith(
        ConstantExpr::getBitCast(&Impl, Specification->getType()));
  }
  return PreservedAnalyses::all();
}
