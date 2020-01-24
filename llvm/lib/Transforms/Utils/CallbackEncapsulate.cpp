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

static constexpr char ReplicatedCallSiteUseString[] = "rpl_cs_use";
static constexpr char ReplicatedCallSiteString[] = "rpl_cs";
static constexpr char ReplicatedAbstractCallSiteString[] = "rpl_acs";
static constexpr char BeforeWrapperSuffix[] = ".before_wrapper";
static constexpr char AfterWrapperSuffix[] = ".after_wrapper";

static void replaceAlInstUsesWith(Value &Old, Value &New) {
  if (!isa<CallBase>(Old))
    return Old.replaceAllUsesWith(&New);
  SmallVector<Use *, 8> Uses;
  for (Use &U : Old.uses())
    if (isa<Instruction>(U.getUser()))
      Uses.push_back(&U);
  for (Use *U : Uses)
    U->set(&New);
}

/// Helper to extract the "rpl_cs" and "rpl_acs" metadata from a call site.
static std::pair<MDNode *, MDNode *> getMDNodeOperands(ImmutableCallSite ICS,
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

static AbstractCallSite
getAbstractCallSiteReplacingDirectCallSite(ImmutableCallSite ICS) {
  if (!ICS)
    return AbstractCallSite();
  // Check if there is "rpl_acs" metadata on the call site and it matches an
  // abstract call site with "rpl_cs" metadata. This means this call site is
  // equivalent to the abstract call site it matches.
  std::pair<MDNode *, MDNode *> RplCSOpMD =
      getMDNodeOperands(ICS, ReplicatedCallSiteString);
  if (!RplCSOpMD.first || !RplCSOpMD.second)
    return AbstractCallSite();

  const Function *Callee = ICS.getCalledFunction();
  for (const Use &U : Callee->uses()) {
    AbstractCallSite ACS(&U);
    if (!ACS || !ACS.isCallbackCall())
      continue;

    std::pair<MDNode *, MDNode *> RplACSOpMD =
        getMDNodeOperands(ACS.getCallSite(), ReplicatedAbstractCallSiteString);
    if (RplACSOpMD.first == RplCSOpMD.second &&
        RplACSOpMD.second == RplCSOpMD.first)
      return ACS;
  }

  return AbstractCallSite();
}

bool llvm::isDirectCallSiteReplacedByAbstractCallSite(ImmutableCallSite ICS) {
  return !!getAbstractCallSiteReplacingDirectCallSite(ICS);
}

bool llvm::isDirectCallSiteUseReplacedByAbstractCallSiteUse(
    ImmutableCallSite ICS, const Use &U) {
  return false;
#if 0
  assert(ICS && ICS.isArgOperand(&U) &&
         "Expected call site and argument operand use");
  AbstractCallSite ACS = getAbstractCallSiteReplacingDirectCallSite(ICS);
  if (!ACS)
    return false;

  std::pair<MDNode *, MDNode *> RplACSOpMD =
      getMDNodeOperands(ACS.getCallSite(), ReplicatedAbstractCallSiteString);
  const Instruction *I = ICS.getInstruction();
  MDNode *MD = I->getMetadata(ReplicatedCallSiteUseString);
  if (!MD)
    return false;

  unsigned ArgNo = ICS.getArgumentNo(&U);
  assert(MD->getNumOperands() % 2 == 0);
  for (unsigned u = 0, e = MD->getNumOperands(); u < e; u += 2) {
    auto *DirectUseIdxMD = cast<ConstantAsMetadata>(MD->getOperand(u));
    uint64_t DirectUseIdx =
        cast<ConstantInt>(DirectUseIdxMD->getValue())->getZExtValue();
    if (DirectUseIdx != ArgNo)
      continue;

    // TODO: verify ACS
    return true;
  }

  return false;
#endif
}


CallBase *llvm::encapsulateAbstractCallSite(AbstractCallSite ACS) {
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
  if (TransitiveCallee.isVarArg())
    return nullptr;

  Module &M = *DirectCallee.getParent();
  LLVMContext &Ctx = M.getContext();

  FunctionType *AfterWrapperTy = TransitiveCallee.getFunctionType();
  Function *AfterWrapper =
      Function::Create(AfterWrapperTy, GlobalValue::InternalLinkage,
                       TransitiveCallee.getName() + AfterWrapperSuffix, M);
  AfterWrapper->setAttributes(TransitiveCallee.getAttributes());
  auto &AfterWrapperBlockList = AfterWrapper->getBasicBlockList();
  auto WrapperAI = AfterWrapper->arg_begin();
  for (Argument &Arg : TransitiveCallee.args()) {
    Argument *WrapperArg = &*(WrapperAI++);
    Arg.replaceAllUsesWith(WrapperArg);
    WrapperArg->setName(Arg.getName());
  }
  AfterWrapperBlockList.splice(AfterWrapperBlockList.begin(),
                               TransitiveCallee.getBasicBlockList());
  BasicBlock *AfterEntryBB =
      BasicBlock::Create(Ctx, "entry", &TransitiveCallee);

  // The after wrapper has the same interface as the transitive callee. The
  // transitive call will just redirect to the after wrapper, thus simply pass
  // all arguments along.
  SmallVector<Value *, 16> Args;
  Args.reserve(TransitiveCallee.arg_size());
  for (Argument &Arg : TransitiveCallee.args())
    Args.push_back(&Arg);

  std::string CIName;
  // Make sure not to "name" void type values.
  CIName = AfterWrapperTy->getReturnType()->isVoidTy()
               ? ""
               : (TransitiveCallee.getName() + ".acs").str();
  CallInst *AfterWrapperCB = CallInst::Create(AfterWrapperTy, AfterWrapper,
                                              Args, CIName, AfterEntryBB);
  ReturnInst::Create(
      Ctx,
      TransitiveCallee.getReturnType()->isVoidTy() ? nullptr : AfterWrapperCB,
      AfterEntryBB);

  // Prepare the arguments for the call that is also an abstract call site.
  // Every argument is passed at most twice and the callee of the abstract call
  // site is passed in the middle.
  Args.clear();
  Args.reserve(CB->getNumArgOperands() * 2 + 1);
  Args.append(CB->arg_begin(), CB->arg_end());

  int CBCalleeIdx = Args.size();
  Args.push_back(AfterWrapper);

  SmallVector<std::pair<unsigned, unsigned>, 8> ReplUsePairs;
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
    if (OpIdx < 0) {
      PayloadIndices.push_back(-1);
      continue;
    }
    PayloadIndices.push_back(Args.size());
    ReplUsePairs.push_back({OpIdx, Args.size()});
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
  BeforeWrapper->addMetadata(LLVMContext::MD_callback,
                             *MDNode::get(Ctx, CBEncodings));

  // Make sure not to "name" void type values.
  CIName = BeforeWrapperTy->getReturnType()->isVoidTy()
               ? ""
               : (TransitiveCallee.getName() + ".cs").str();
  auto *BeforeWrapperCB = CallInst::Create(BeforeWrapper->getFunctionType(),
                                           BeforeWrapper, Args, CIName, CB);
  replaceAlInstUsesWith(*CB, *BeforeWrapperCB);

  SmallVector<Metadata *, 4> Ops;
  Type *Int64 = Type::getInt64Ty(Ctx);
  for (const auto &ReplUsePair : ReplUsePairs) {
    Ops.push_back(
        MDB.createConstant(ConstantInt::get(Int64, ReplUsePair.first, true)));
    Ops.push_back(
        MDB.createConstant(ConstantInt::get(Int64, ReplUsePair.second, true)));
  }
  MDNode *ReplUseMD = MDNode::get(Ctx, Ops);
  BeforeWrapperCB->setMetadata(ReplicatedCallSiteUseString, ReplUseMD);

  // Create and attach the encoding metadata to the two call site (one
  // abstract, one direct) of the called wrapper function.
  MDNode *BeforeWrapperCBMD = MDNode::get(Ctx, {nullptr});
  BeforeWrapperCB->setMetadata(ReplicatedAbstractCallSiteString,
                               BeforeWrapperCBMD);

  MDNode *AfterWrapperCBMD = MDNode::get(Ctx, {BeforeWrapperCBMD});
  AfterWrapperCB->setMetadata(ReplicatedCallSiteString, AfterWrapperCBMD);
  BeforeWrapperCBMD->replaceOperandWith(0, AfterWrapperCBMD);

  BasicBlock *BeforeEntryBB = BasicBlock::Create(Ctx, "entry", BeforeWrapper);
  ReturnInst *RI = ReturnInst::Create(
      Ctx, DirectCallee.getReturnType()->isVoidTy() ? nullptr : CB,
      BeforeEntryBB);
  // Reuse the old call in the new wrapper.
  CB->moveBefore(RI);

  // Set the names of and rewire arguments.
  unsigned NumDCArgs = DirectCallee.arg_size();
  auto CalledAI = DirectCallee.arg_begin();
  auto BeforeWrapperAI = BeforeWrapper->arg_begin();
  for (unsigned u = 0, e = CB->getNumArgOperands(); u < e;
       ++u, ++BeforeWrapperAI) {
    if (u < NumDCArgs)
      BeforeWrapperAI->setName((CalledAI++)->getName());
    else
      BeforeWrapperAI->setName(Args[u]->getName());
    if (Constant *C = dyn_cast<Constant>(Args[u]))
      CB->setArgOperand(u, C);
    else
      CB->setArgOperand(u, &*BeforeWrapperAI);
  }

  CalledAI = DirectCallee.arg_begin();
  for (unsigned u = 0, e = CB->getNumArgOperands(); u < e; u++) {
    if (u < NumDCArgs)
      BeforeWrapperAI->setName((CalledAI++)->getName());
    else
      BeforeWrapperAI->setName(Args[u]->getName());
  }

  return BeforeWrapperCB;
}
