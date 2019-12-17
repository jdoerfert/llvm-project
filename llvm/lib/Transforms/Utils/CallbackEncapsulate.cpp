//===- CallbackEncapsulate.cpp -- Encapsulate callbacks in own functions --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper methods to deal with callback wrapper around a call sites.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CallbackEncapsulate.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/Error.h"

using namespace llvm;

#define DEBUG_TYPE "callback-encapsulate"

STATISTIC(NumCallbacksEncapsulated, "Number of callbacks encapsulated");

static constexpr char ReplacingACSString[] = "replacing_acs";
static constexpr char ReplacedCSString[] = "replaced_cs";
static constexpr char BeforeWrapperSuffix[] = ".before_wrapper";
static constexpr char AfterWrapperSuffix[] = ".after_wrapper";

/// Helper to extract the "replacing_acs" and "replaced_cs" metadata from a call
/// site.
static std::pair<MDNode *, MDNode *> getMDNodeOperands(const Instruction *I,
                                                       StringRef MDString) {
  std::pair<MDNode *, MDNode *> Ret;
  if (!I)
    return Ret;

  MDNode *MD = I->getMetadata(MDString);
  if (!MD || MD->getNumOperands() != 1)
    return Ret;

  MDNode *OpMD = dyn_cast_or_null<MDNode>(MD->getOperand(0).get());
  return {MD, OpMD};
}

bool llvm::isDirectCallSiteReplacedByAbstractCallSite(const CallBase &CB) {
  // Check if there is "replaced_cs" metadata on the call site and it matches an
  // abstract call site with "replacing_acs" metadata. This means this call site
  // is equivalent to the abstract call site it matches.
  std::pair<MDNode *, MDNode *> RplCSOpMD =
      getMDNodeOperands(&CB, ReplacingACSString);
  if (!RplCSOpMD.first || !RplCSOpMD.second)
    return false;

  const Function *Callee = CB.getCalledFunction();
  for (const Use &U : Callee->uses()) {
    AbstractCallSite ACS(&U);
    if (!ACS || !ACS.isCallbackCall())
      continue;

    std::pair<MDNode *, MDNode *> RplACSOpMD =
        getMDNodeOperands(ACS.getInstruction(), ReplacedCSString);
    if (RplACSOpMD.first == RplCSOpMD.second &&
        RplACSOpMD.second == RplCSOpMD.first)
      return true;
  }

  return false;
}

Function *llvm::createEncapsulateAfterWrapper(Function &Callee) {
  Module &M = *Callee.getParent();
  LLVMContext &Ctx = M.getContext();
  FunctionType *AfterWrapperTy = Callee.getFunctionType();
  Function *AfterWrapper =
      Function::Create(AfterWrapperTy, GlobalValue::InternalLinkage,
                       Callee.getName() + AfterWrapperSuffix, M);

  AfterWrapper->setAttributes(Callee.getAttributes());
  auto &AfterWrapperBlockList = AfterWrapper->getBasicBlockList();
  auto WrapperAI = AfterWrapper->arg_begin();
  for (Argument &Arg : Callee.args()) {
    Argument *WrapperArg = &*(WrapperAI++);
    Arg.replaceAllUsesWith(WrapperArg);
    WrapperArg->setName(Arg.getName());
  }
  AfterWrapperBlockList.splice(AfterWrapperBlockList.begin(),
                               Callee.getBasicBlockList());
  BasicBlock *AfterEntryBB = BasicBlock::Create(Ctx, "entry", &Callee);

  // The after wrapper has the same interface as the transitive callee. The
  // transitive call will just redirect to the after wrapper, thus simply pass
  // all arguments along.
  SmallVector<Value *, 16> Args;
  Args.reserve(Callee.arg_size());
  for (Argument &Arg : Callee.args())
    Args.push_back(&Arg);

  CallInst *AfterWrapperCB =
      CallInst::Create(AfterWrapperTy, AfterWrapper, Args,
                       Callee.getName() + ".acs", AfterEntryBB);
  ReturnInst::Create(
      Ctx, Callee.getReturnType()->isVoidTy() ? nullptr : AfterWrapperCB,
      AfterEntryBB);
  return AfterWrapper;
}

CallBase *llvm::encapsulateAbstractCallSite(AbstractCallSite ACS,
                                            Function *AfterWrapper) {
  assert(ACS && "Expected valid abstract call site!");

  bool IsCallback = ACS.isCallbackCall();
  CallBase *CB = cast<CallBase>(ACS.getInstruction());
  Function &DirectCallee = *CB->getCalledFunction();
  Function &TransitiveCallee = *ACS.getCalledFunction();

  // If we have a direct call, the transitive callee is the same as the direct
  // callee. However, for callback calls this might not be the case.
  assert((IsCallback || (&DirectCallee == &TransitiveCallee)) &&
         "Broken invariant");

  // We do not allow varargs for now.
  if (DirectCallee.isVarArg() || TransitiveCallee.isVarArg())
    return nullptr;

  Module &M = *DirectCallee.getParent();
  LLVMContext &Ctx = M.getContext();

  if (!AfterWrapper)
    AfterWrapper = createEncapsulateAfterWrapper(TransitiveCallee);

  // assert(AfterWrapper->getFunctionType() ==
  // TransitiveCallee.getFunctionType());
  assert(AfterWrapper->getNumUses() == 1);
  assert(isa<CallInst>(AfterWrapper->user_back()));
  CallInst *AfterWrapperCB = cast<CallInst>(AfterWrapper->user_back());
  // TransitiveCallee.dump();
  // AfterWrapperCB->dump();
  // assert(AfterWrapperCB->getCaller() == &TransitiveCallee);

  // Prepare the arguments for the call that is also an abstract call site.
  // Every argument is passed at most twice and the callee of the abstract call
  // site is passed in the middle.
  SmallVector<Value *, 16> Args;
  Args.reserve(CB->getNumArgOperands() * 2 + 1);
  Args.append(CB->arg_begin(), CB->arg_end());

  int CBCalleeIdx = Args.size();
  Args.push_back(AfterWrapper);

  SmallVector<int, 8> PayloadIndices;
  AttributeList CalleeFnAttrs = TransitiveCallee.getAttributes();
  AttributeList ExtCalledAttrs = DirectCallee.getAttributes();

  // Collect the arguments that go into the abstract call. These are all
  // arguments if the call site ACS was direct or the subset that the abstract
  // call site ACS actually used. Given that we might skip arguments we need to
  // track the payload indices for the callback encoding as well. Finally, we
  // keep the attributes of the original arguments we duplicate around.
  for (unsigned u = 0, e = TransitiveCallee.arg_size(); u < e; u++) {
    int OpIdx = ACS.getCallArgOperandNo(u);
    if (OpIdx < 0)
      continue;
    PayloadIndices.push_back(Args.size());
    AttributeSet CalleeFnParamAttrs = CalleeFnAttrs.getParamAttributes(u);
    ExtCalledAttrs = ExtCalledAttrs.addParamAttributes(
        Ctx, Args.size(), AttrBuilder(CalleeFnParamAttrs));
    Args.push_back(CB->getOperand(OpIdx));
  }

  SmallVector<Type *, 16> ArgTypes;
  for (Value *V : Args)
    ArgTypes.push_back(V->getType());

  FunctionType *BeforeWrapperTy =
      FunctionType::get(DirectCallee.getReturnType(), ArgTypes, false);
  Function *BeforeWrapper =
      Function::Create(BeforeWrapperTy, GlobalValue::InternalLinkage,
                       DirectCallee.getName() + BeforeWrapperSuffix, M);
  BeforeWrapper->setAttributes(ExtCalledAttrs);

  MDBuilder MDB(Ctx);
  SmallVector<Metadata *, 4> CBEncodings;
  CBEncodings.push_back(
      MDB.createCallbackEncoding(CBCalleeIdx, PayloadIndices,
                                 /* VarArgsArePassed */ false));

  // If the direct callee already has callback metadata we copy it to the before
  // wrapper which has the same behavior and argument prefix.
  MDNode *ExistingCBMD = DirectCallee.getMetadata(LLVMContext::MD_callback);
  if (ExistingCBMD)
    CBEncodings.append(ExistingCBMD->op_begin(), ExistingCBMD->op_end());
  BeforeWrapper->addMetadata(LLVMContext::MD_callback,
                             *MDNode::get(Ctx, CBEncodings));

  auto *BeforeWrapperCB =
      CallInst::Create(BeforeWrapper->getFunctionType(), BeforeWrapper, Args,
                       TransitiveCallee.getName() + ".cs", CB);
  CB->replaceAllUsesWith(BeforeWrapperCB);

  // Create and attach the encoding metadata to the two call site (one
  // abstract, one direct) of the called wrapper function.
  MDNode *BeforeWrapperCBMD = MDNode::get(Ctx, {nullptr});
  BeforeWrapperCB->setMetadata(ReplacedCSString, BeforeWrapperCBMD);

  MDNode *AfterWrapperCBMD =
      MDNode::get(Ctx, {BeforeWrapperCBMD, CBEncodings[0]});
  AfterWrapperCB->setMetadata(ReplacingACSString, AfterWrapperCBMD);
  BeforeWrapperCBMD->replaceOperandWith(0, AfterWrapperCBMD);

  BasicBlock *BeforeEntryBB = BasicBlock::Create(Ctx, "entry", BeforeWrapper);
  ReturnInst *RI = ReturnInst::Create(
      Ctx, DirectCallee.getReturnType()->isVoidTy() ? nullptr : CB,
      BeforeEntryBB);
  // Reuse the old call in the new wrapper.
  CB->moveBefore(RI);

  // Set the names of and rewire arguments.
  auto CalledAI = DirectCallee.arg_begin();
  auto BeforeWrapperAI = BeforeWrapper->arg_begin(),
       BeforeWrapperAE = BeforeWrapper->arg_end();
  for (unsigned u = 0, e = CB->getNumArgOperands(); u < e; u++) {
    BeforeWrapperAI->setName((CalledAI++)->getName());
    CB->setArgOperand(u, &*(BeforeWrapperAI++));
  }

  CalledAI = DirectCallee.arg_begin();
  for (unsigned u = 0, e = CB->getNumArgOperands(); u < e; u++)
    BeforeWrapperAI->setName((CalledAI++)->getName());

  return BeforeWrapperCB;
}
