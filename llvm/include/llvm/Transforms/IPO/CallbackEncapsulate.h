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

/// This method encapsulates the \p Called and the \p Callee function (which can
/// be the same) with new functions that are connected through a callback
/// annotation. The callback annotation uses copies of the arguments and the
/// original ones are still passed. We do this to allow later passes, e.g.,
/// argument promotion, to modify the passed arguments without changing the
/// interface of \p Called and \p Callee. This can be good for two reasons:
///
/// (1) If \p Called is a declaration that has callback behavior and \p Callee
/// is the callback callee we could otherwise not modify the way arguments are
/// passed between them.
///
/// (2) If \p Callee is passed very large structure we want to unpack it to
/// facilitate later analysis but we lack the ability to pack them again to
/// guarantee the same call performance.
///
/// The new abstract call site and the direct one that with the same callee are
/// tied together through metadata as shown in the example below.
///
/// Note that the encapsulation does not change the semantic of the code. While
/// there are more functions and calls involved, there is no semantic change.
/// However, passes aware of abstract call sites and the encoding metadata can
/// use this mechanism to reuse existing logic.
///
/// ------------------------------- Before ------------------------------------
///
///     call Called(p0, p1);
///
///
///   // The definition of Called might not be available. Called can be Callee
///   // or contain call to Callee.
///   Called(arg0, arg1);
///
///   Callee(arg2, arg3) {
///     // Callee code
///   }
///
///
/// ------------------------------- After -------------------------------------
///
///     // call metadata !{!"rpl_cs", !0}
///(A)  call Called_wrapper(p0, p1, Callee_wrapper, p0, p1);
///
///
///   __attribute__((callback(callee_w, arg2_w, arg3_w)))
///   Called_wrapper(arg0, arg1, callee_w, arg2_w, arg3_w) {
///(B)  call Called(arg0, arg1);
///   }
///
///   // The definition of Called might not be available. Called can be Callee
///   // or contain call to Callee.
///   Called(arg0, arg1);
///
///   Callee(arg2, arg3) {
///     // call metadata !{!"rpl_acs", !1}
///(C)  call Callee_wrapper(arg2, arg3);
///   }
///
///   Callee_wrapper(arg2, arg3) {
///(D)  // Callee code
///   }
///
/// !0 = {!1}
/// !1 = {!0}
///
/// In this encoding, the following call edges exist:
///   (1)  (A) -> Called_wrapper  [direct]
///   (2)  (A) -> Callee_wrapper  [transitive]
///   (3)  (B) -> Called          [direct]
///   (4)  (C) -> Callee_wrapper  [direct]
///
/// The shown metadata is used to tie (2) and (4) together such that aware users
/// can ignore (4) in favor of (2). If the metadata is corrupted or dropped, the
/// connection cannot be made and (4) has to be taken into account. This for
/// example the case if (B) was inlined.
bool encapsulateCallSites(Function &Called, Function *Callee = nullptr,
                          int64_t CalleeIdx = -1);

/// Return true if \p CS is a direct call with a replacing abstract call site
/// that should be used for inter-procedural reasoning instead.
///
/// This function should only be used by abstract call site aware
/// inter-procedural passes. If the return value is true, and the passes will
/// eventually look at all direct and transitive call sites to derive
/// information, they can ignore the direct call site \p CS as there will be an
/// abstract call site that encodes the same call.
bool isDirectCallSiteReplacedByAbstractCallSite(ImmutableCallSite CS);

struct CallbackEncapsulatePass : PassInfoMixin<CallbackEncapsulatePass> {
  /// Run the pass over the module.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_CALLBACK_ENCAPSULATE_H
