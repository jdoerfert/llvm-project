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
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Transforms/IPO.h"

using namespace llvm;

#define DEBUG_TYPE "callback-encapsulate"

STATISTIC(NumCallbacksEncapsulated, "Number of callbacks encapsulated");

/// Helper to extract the "rpl_cs" and "rpl_acs" metadata from a call site.
static std::pair<MDNode *, MDNode *> getMDNodeOperand(ImmutableCallSite ICS,
                                                      StringRef MDString) {
  std::pair<MDNode *, MDNode *> Ret;
  const Instruction *I = ICS.getInstruction();
  if (!I)
    return Ret;

  MDNode *MD = I->getMetadata(MDString);
  if (!MD || MD->getNumOperands() != 1)
    return Ret;

  MDNode *OpMD = dyn_cast_or_null<MDNode>(MD->getOperand(0).get());
  return {MD, OpMD};
}

bool llvm::isDirectCallSiteReplacedByAbstractCallSite(ImmutableCallSite CS) {
  std::pair<MDNode *, MDNode *> RplCSOpMD = getMDNodeOperand(CS, "rpl_cs");
  if (!RplCSOpMD.first || !RplCSOpMD.second)
    return false;

  const Function *Callee = CS.getCalledFunction();
  for (const Use &U : Callee->uses()) {
    AbstractCallSite ACS(&U);
    if (!ACS)
      continue;

    std::pair<MDNode *, MDNode *> RplACSOpMD =
        getMDNodeOperand(ACS.getCallSite(), "rpl_acs");
    if (RplACSOpMD.first == RplCSOpMD.second &&
        RplACSOpMD.second == RplCSOpMD.first)
      return true;
  }

  return false;
}

bool llvm::encapsulateCallSites(Function &Called, Function *Callee,
                                int64_t CalleeIdx) {
  Called.dump();
  assert((Callee || CalleeIdx >= 0) && "Callee or callee index is required!");

  SmallVector<Value *, 16> Args;
  Module &M = *Called.getParent();
  LLVMContext &Ctx = M.getContext();

  MDNode *CalleeWrapperCIMD = nullptr;
  Function *CalledWrapper = nullptr, *CalleeWrapper = nullptr;
  bool Changed = false;

  // Iterate over the call sites of the called function and rewrite calls to
  // target the new wrapper.
  for (const Use &U : Called.uses()) {
    auto *CI = dyn_cast<CallInst>(U.getUser());
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

    CalleeFn->dump();
    // Check if we need to create a new wrapper for the callee function. This is
    // true if we have none and we bridge a direct call site or always if we
    // bridge an abstract call site.
    CallInst *CalleeWrapperCI = nullptr;
    if (!Callee || !CalleeWrapper) {
      FunctionType *WrapperTy = CalleeFn->getFunctionType();
      CalleeWrapper = Function::Create(WrapperTy, GlobalValue::InternalLinkage,
                                       CalleeFn->getName() + ".clew", M);
      CalleeWrapper->setAttributes(CalleeFn->getAttributes());
      CalleeWrapper->dump();
      auto &CalleeWrapperBlockList = CalleeWrapper->getBasicBlockList();
      auto WrapperAI = CalleeWrapper->arg_begin();
      for (Argument &Arg : CalleeFn->args()) {
        Argument *WrapperArg = &*(WrapperAI++);
        Arg.replaceAllUsesWith(WrapperArg);
        WrapperArg->setName(Arg.getName());
      }
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
          Ctx,
          CalleeFn->getReturnType()->isVoidTy() ? nullptr : CalleeWrapperCI,
          EntryBB);
      CalleeWrapper->dump();
    }
    M.dump();
    errs() << "Callee side done\n";

    // Prepare the arguments for the call that is also an abstract call site.
    // Every argument is passed twice and the callee of the abstract call site
    // is passed in the middle.
    Args.clear();
    Args.reserve(CI->getNumArgOperands() * 2 + 1);
    Args.append(CI->arg_begin(), CI->arg_end());

    int CBCalleeIdx = Args.size();
    Args.push_back(CalleeWrapper);

    SmallVector<int, 8> PayloadIndices;

    const Use *ACSUse = CalleeIdx >= 0 ? CI->op_begin() + CalleeIdx : &U;
    AbstractCallSite ACS(ACSUse);
    assert(ACS && "Expected valid abstract call site!");
    for (unsigned u = 0, e = CalleeFn->arg_size(); u < e; u++) {
      int OpIdx = ACS.getCallArgOperandNo(u);
      if (OpIdx < 0)
        continue;
      errs() << "u: " << u << " ==> " << OpIdx << "\n";
      PayloadIndices.push_back(Args.size());
      Args.push_back(CI->getOperand(OpIdx));
    }

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
      WrapperTy->dump();
      CalledWrapper = Function::Create(WrapperTy, GlobalValue::InternalLinkage,
                                       Called.getName() + ".cldw", M);
      CalledWrapper->setAttributes(Called.getAttributes());
      MDBuilder MDB(Ctx);
      CalledWrapper->addMetadata(
          LLVMContext::MD_callback,
          *MDNode::get(
              Ctx, {MDB.createCallbackEncoding(CBCalleeIdx, PayloadIndices,
                                               /* VarArgsArePassed */ false)}));
      CalledWrapper->dump();
      NewCalledWrapper = true;
    }

    errs() << "NewCalledWrapper: " << NewCalledWrapper << "\n";
    errs() << "Args:\n";
      for (auto *A : Args)
        A->dump();
    errs() << "# " << Args.size() << " :: " << CalledWrapper->getFunctionType()->getNumParams() << "\n";
    auto *CalledWrapperCI =
        CallInst::Create(CalledWrapper->getFunctionType(), CalledWrapper, Args,
                         CalleeFn->getName() + ".cs", CI);
    CI->replaceAllUsesWith(CalledWrapperCI);
    CalleeWrapperCI->dump();

    // Create and attach the encoding metadata to the two call site (one
    // abstract, one direct) of the called wrapper function.
    MDNode *CalledWrapperCIMD =
        MDNode::get(Ctx, {CalleeWrapperCIMD ? cast<Metadata>(CalleeWrapperCIMD)
                                            : nullptr});
    CalledWrapperCI->setMetadata("rpl_acs", CalledWrapperCIMD);

    if (!CalleeWrapperCIMD) {
      CalleeWrapperCIMD = MDNode::get(Ctx, {CalledWrapperCIMD});
      CalleeWrapperCI->setMetadata("rpl_cs", CalleeWrapperCIMD);
      M.dump();
      CalledWrapperCIMD->replaceOperandWith(0, CalleeWrapperCIMD);
    }
    M.dump();

    if (!NewCalledWrapper) {
      CI->eraseFromParent();
    } else {
      BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", CalledWrapper);
      ReturnInst *RI = ReturnInst::Create(
          Ctx, Called.getReturnType()->isVoidTy() ? nullptr : CI, EntryBB);
      CI->moveBefore(RI);

      auto CldAI = Called.arg_begin();
      auto CldWrapperAI = CalledWrapper->arg_begin(),
           CldWrapperAE = CalledWrapper->arg_end();
      for (unsigned u = 0, e = CI->getNumArgOperands(); u < e; u++) {
        CldWrapperAI->setName((CldAI++)->getName());
        CI->setArgOperand(u, &*(CldWrapperAI++));
      }

      #if 0
      IRBuilder<> Builder(
          &*CalledWrapper->getEntryBlock().getFirstInsertionPt());
      Function *VarAnnotate = Intrinsic::getDeclaration(&M, Intrinsic::var_annotation);
      Type *I8PtrTy = Type::getInt8PtrTy(Ctx);
      Value *NullPtr = Constant::getNullValue(I8PtrTy);
      Value *NullLine = Constant::getNullValue(Type::getInt32Ty(Ctx));
      do {
        CldWrapperAI->setName(
            CalledWrapperCI->getOperand(CldWrapperAI->getArgNo())->getName());
        Value *ArgPtr = Builder.CreatePointerCast(&*CldWrapperAI, I8PtrTy);
        Builder.CreateCall(VarAnnotate, {ArgPtr, NullPtr, NullPtr, NullLine});
      } while (++CldWrapperAI != CldWrapperAE);
      #endif
    }
      CalledWrapper->dump();
  }

  M.dump();
  return Changed;
}

static bool encapsulateCallbackCallSites(Function &Called, MDNode &CallbackMD) {
  assert(CallbackMD.getNumOperands() >= 2 && "Incomplete !callback metadata");

  Metadata *OpAsM = CallbackMD.getOperand(/* callback callee idx */ 0).get();
  auto *OpAsCM = cast<ConstantAsMetadata>(OpAsM);
  assert(OpAsCM->getType()->isIntegerTy(64) && "Malformed !callback metadata");

  int64_t Idx = cast<ConstantInt>(OpAsCM->getValue())->getSExtValue();
  assert(-1 <= Idx && "Out-of-bounds !callback metadata index");
  assert(((unsigned)Idx <= Called.arg_size() || Called.isVarArg()) &&
         "Out-of-bounds !callback metadata index");
  if (Idx == -1)
    return false;
  return encapsulateCallSites(Called, nullptr, Idx);
}

static bool encapsulateFunctions(Module &M) {
  bool Changed = false;

  SmallVector<Function *, 32> Fns;
  for (Function &F : M)
    Fns.push_back(&F);

  for (Function *F : Fns) {
    // We always encapsulate functions with callback behavior.
    if (MDNode *CallbackMD = F->getMetadata(LLVMContext::MD_callback)) {
      for (const MDOperand &Op : CallbackMD->operands())
        Changed |= encapsulateCallbackCallSites(*F, *cast<MDNode>(Op.get()));
      continue;
    }

    // TODO: We do not internalize functions yet so we limit encapsulation to
    //       local functions for now.
    if (!F->hasLocalLinkage())
      continue;

    // Encapsulate function definitions without callback behavior if one of the
    // arguments is a dereferenceable pointer to a "large" struct type.
    // TODO: Allow opaque pointers that are used as struct pointers
    for (Argument &Arg : F->args()) {
      Type *ArgTy = Arg.getType();
      if (!ArgTy->isPointerTy())
        continue;
      auto *ArgStructTy = dyn_cast<StructType>(ArgTy->getPointerElementType());
      if (!ArgStructTy ||
          ArgStructTy->getNumElements() <= /* Same as in ArgumentPromotion */ 3)
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

PreservedAnalyses CallbackEncapsulatePass::run(Module &M,
                                               ModuleAnalysisManager &MAM) {
  if (!encapsulateFunctions(M))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}
