//===- CallbackEncapsulate.cpp -- Encapsulate callbacks in own functions --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/CallbackEncapsulate.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/ADT/Statistic.h"

using namespace llvm;

#define DEBUG_TYPE "callback-encapsulate"

STATISTIC(NumCallbacksEncapsulated, "Number of callbacks encapsulated");

bool llvm::isDirectCallSiteReplacedByAbstractCallSite(ImmutableCallSite CS) {
  const Instruction *I = CS.getInstruction();
  if (!I)
    return false;
  MDNode *RplACSMD = I->getMetadata("rpl_acs");
  if (!RplACSMD || RplACSMD->getNumOperands() != 1)
    return false;
  MDNode *RplACSOpMD = dyn_cast_or_null<MDNode>(RplACSMD->getOperand(0).get());
  if (!RplACSOpMD || RplACSOpMD->getNumOperands() != 1 || RplACSOpMD->)
}

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
static bool encapsulateCallSites(Function &Called, Function *Callee = nullptr,
                                 int64_t CalleeIdx = -1) {
  assert((Callee || CalleeIdx >= 0) && "Callee or callee index is required!");

  SmallVector<Value *, 16> Args;
  Module &M = *Called.getParent();
  LLVMContext &Ctx = M.getContext();

  MDNode *CalleeWrapperCIMD = nullptr;
  Function *CalledWrapper = nullptr, *CalleeWrapper = nullptr;
  bool Changed = false;

  // Iterate over the call sites of the called function and rewrite calls to
  // target the new wrapper.
  for (User *U : Called.users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;

    // If the encoding will bridge an existing abstract call site, the Callee
    // argument is null and the CalleeIdx is a valid index that identifies the
    // abstract callee.
    Function *CalleeFn = Callee;
    if (!CalleeFn) {
      Value *CalleeOpV = CI->getOperand(CalleeIdx);
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(CalleeOpV))
        if (CE->getNumUses() == 1 && CE->isCast())
          CalleeOpV = CE->getOperand(0);
      CalleeFn = dyn_cast<Function>(CalleeOpV);
      if (!CalleeFn)
        continue;
    }

    // Ignore variadic callees as they don't seem important and are more complex
    // to handle.
    if (CalleeFn->isVarArg())
      continue;

    // Check if we need to create a new wrapper for the callee function. This is
    // true if we have none and we bridge a direct call site or always if we
    // bridge an abstract call site.
    CallInst *CalleeWrapperCI = nullptr;
    if (!Callee || !CalleeWrapper) {
      FunctionType *WrapperTy = CalleeFn->getFunctionType();
      CalleeWrapper = Function::Create(WrapperTy, GlobalValue::InternalLinkage,
                                       CalleeFn->getName() + ".clew", M);
      auto &CalleeWrapperBlockList = CalleeWrapper->getBasicBlockList();
      CalleeWrapperBlockList.splice(CalleeWrapperBlockList.begin(),
                                    CalleeFn->getBasicBlockList());
      BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", CalleeFn);

      Args.clear();
      Args.reserve(CalleeFn->arg_size());
      for (Argument &Arg : CalleeFn->args())
        Args.push_back(&Arg);
      CalleeWrapperCI = CallInst::Create(WrapperTy, CalleeWrapper, Args,
                                         CalleeFn->getName() + ".acs", EntryBB);

      ReturnInst *RI = ReturnInst::Create(
          Ctx, CalleeFn->getReturnType()->isVoidTy() ? nullptr : CalleeWrapperCI,
          EntryBB);
    }

    // Prepare the arguments for the call that is also an abstract call site.
    // Every argument is passed twice and the callee of the abstract call site
    // is passed in the middle.
    Args.clear();
    Args.reserve(CI->getNumArgOperands() * 2 + 1);
    Args.append(CI->arg_begin(), CI->arg_end());
    Args.push_back(CalleeWrapper);
    Args.append(CI->arg_begin(), CI->arg_end());

    // Check if we need to create a new wrapper for the called function. This is
    // true if we have none and we bridge a direct call site or always if we
    // bridge an abstract call site.
    bool NewCalledWrapper = false;
    if (!Callee || !CalledWrapper) {
      SmallVector<Type *, 16> ArgTypes;
      for (Value *V : Args)
        ArgTypes.push_back(V->getType());
      FunctionType *WrapperTy =
          FunctionType::get(Called.getReturnType(), ArgTypes, false);
      CalledWrapper = Function::Create(WrapperTy, GlobalValue::InternalLinkage,
                                       Called.getName() + ".cldw", M);
      NewCalledWrapper = true;
      // TODO callack encoding
    }

    auto *CalledWrapperCI =
        CallInst::Create(CI->getFunctionType(), CalledWrapper, Args,
                         CalleeFn->getName() + ".cs", CI);

    // Create and attach the encoding metadata to the two call site (one
    // abstract, one direct) of the called wrapper function.
    DistinctMDOperandPlaceholder Placeholder(0);
    MDNode *CalledWrapperCIMD =
        MDNode::get(Ctx, {CalleeWrapperCIMD ? cast<Metadata>(CalleeWrapperCIMD)
                                            : &Placeholder});
    CalledWrapperCI->setMetadata("rpl_acs", CalledWrapperCIMD);

    if (!CalleeWrapperCIMD) {
      CalleeWrapperCIMD = MDNode::get(Ctx, {CalledWrapperCIMD});
      CalleeWrapperCI->setMetadata("rpl_cs", CalleeWrapperCIMD);
      CalledWrapperCIMD->replaceOperandWith(0, CalleeWrapperCIMD);
    }

    if (!NewCalledWrapper) {
      CI->eraseFromParent();
    } else {
      BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", CalledWrapper);
      ReturnInst *RI = ReturnInst::Create(
          Ctx, Called.getReturnType()->isVoidTy() ? nullptr : CI, EntryBB);
      CI->moveBefore(RI);

      auto AI = CalledWrapper->arg_begin();
      for (unsigned u = 0, e = CI->getNumArgOperands(); u < e; u++)
        CI->setArgOperand(u, &*(AI++));
    }
  }
  return Changed;
}

static bool encapsulateCallbackCallSites(Function &Called, MDNode &CallbackMD) {
  assert(CallbackMD.getNumOperands() >= 2 && "Incomplete !callback metadata");

  Metadata *OpAsM = CallbackMD.getOperand(/* callback callee idx */0).get();
  auto *OpAsCM = cast<ConstantAsMetadata>(OpAsM);
  assert(OpAsCM->getType()->isIntegerTy(64) &&
          "Malformed !callback metadata");

  int64_t Idx = cast<ConstantInt>(OpAsCM->getValue())->getSExtValue();
  assert(-1 <= Idx && "Out-of-bounds !callback metadata index");
  assert((Idx <= Called.arg_size() || Called.isVarArg()) && "Out-of-bounds !callback metadata index");

  return encapsulateCallSites(Called, nullptr, Idx);
}


///   __attribute__((callback(cb_fn, cb_arg0, cb_arg1)))
///   int F(arg0, cb_fn, cb_arg0, arg1, cb_arg1);
///
///
///
/// The
/// For each call site with non-callback args and callback args (cb_args)
///
///   val = F(arg0, arg1, cb_arg0, arg2, cb_arg1, cb_arg2)
///
/// we
///
///   val = F_wrapper(arg0, arg1, arg2, cb_arg0, cb_arg1, cb_arg2, cb_arg0,
///                  cb_arg1, cb_arg2)
///
///

static bool encapsulateFunctions(Module &M) {
  bool Changed = false;

  SmallVector<Function *, 32> Fns;
  for (Function &F : M)
    Fns.push_back(&F);

  for (Function *F : Fns) {
    // We always encapsulate functions with callback behavior.
    if (MDNode *CallbackMD = F->getMetadata(LLVMContext::MD_callback)) {
      Changed |= encapsulateCallbackCallSites(*F, *CallbackMD);
      continue;
    }

    // Encapsulate function definitions without callback behavior if one of the
    // arguments is a dereferenceable pointer to a "large" struct type.
    // TODO: Make large variable
    // TODO: Allow opaque pointers that are used as struct pointers
    for (Argument &Arg : F->args()) {
      Type *ArgTy = Arg.getType();
      if (!ArgTy->isPointerTy())
        continue;
      auto *ArgStructTy = dyn_cast<StructType>(ArgTy->getPointerElementType());
      if (!ArgStructTy || ArgStructTy->getNumElements() <= 3)
        continue;
      Changed |= encapsulateCallSites(*F, F);
      continue;
    }
  }
  return Changed;
}

namespace {
struct CallbackEncapsulateLegacyPass : public ModulePass {
  static char ID;
  CallbackEncapsulateLegacyPass() : ModulePass(ID) {
    initializeCallbackEncapsulateLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {}

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;
    return encapsulateFunctions(M);
  }
};
} // namespace

char CallbackEncapsulateLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(CallbackEncapsulateLegacyPass, "callback-encapsulate",
                      "Callback encapsulate", false, false)
INITIALIZE_PASS_END(CallbackEncapsulateLegacyPass, "callback-encapsulate",
                    "Callback encapsulate", false, false)
ModulePass *llvm::createCallbackEncapsulatePass() {
  return new CallbackEncapsulateLegacyPass();
}

PreservedAnalyses CallbackEncapsulate::run(Module &M,
                                           ModuleAnalysisManager &MAM) {
  if (!encapsulateFunctions(M))
    return PreservedAnalyses::all();
  //PreservedAnalyses PA;
  //PA.preserve<DominatorTreeAnalysis>();
  //return PA;
  return PreservedAnalyses::none();
}
