//===- CallbackEncapsulate.h - Isolate callbacks in own functns -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_CALLBACK_ENCAPSULATE_H
#define LLVM_TRANSFORMS_SCALAR_CALLBACK_ENCAPSULATE_H

#include "llvm/IR/CallSite.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Return true if \p CS is a direct call with a replacing abstract call site
/// that should be used for inter-procedural reasoning instead.
///
/// This function should only be used by abstract call site aware
/// inter-procedural passes. If the return value is true, and the passes will
/// eventually look at all direct and transitive call sites to derive
/// information, they can ignore the direct call site \p CS as there will be an
/// abstract call site that encodes the same call.
bool isDirectCallSiteReplacedByAbstractCallSite(ImmutableCallSite CS);

struct CallbackEncapsulate : PassInfoMixin<CallbackEncapsulate> {
  /// Run the pass over the module.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_CALLBACK_ENCAPSULATE_H
