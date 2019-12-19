//===- AssumptionOutliner.cpp - Extract assumptions into own functions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/AlignmentFromAssumptions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

using namespace llvm;

static cl::opt<bool> DuplicateInst("assumption-outliner-duplicate-inst",
                                   cl::Hidden, cl::init(false));
static cl::opt<bool> Aggressive("assumption-outliner-aggressive", cl::Hidden,
                                cl::init(true));

static void markAsOutlined(Instruction &OutlinedI,
                           SmallVectorImpl<Instruction *> &OutlinedInstsOrdered,
                           DenseMap<Instruction *, int> &NonOutlinedUsesMap) {
  assert(NonOutlinedUsesMap[&OutlinedI] == 0);
  SmallVector<Instruction *, 8> Worklist;
  Worklist.push_back(&OutlinedI);

  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    if (isa<PHINode>(I))
      continue;
    OutlinedInstsOrdered.push_back(I);
    for (Value *Op : I->operands())
      if (auto *OpI = dyn_cast<Instruction>(Op)) {
        int &NonOutlinedUses = NonOutlinedUsesMap[OpI];
        if (NonOutlinedUses == 0)
          NonOutlinedUses += OpI->getNumUses();
        NonOutlinedUses -= 1;
        if (NonOutlinedUses == 0)
          Worklist.push_back(OpI);
      }
  }
}
static Instruction *duplicateInsts(IntrinsicInst &AssumeCall,
                                   Instruction *OutlinedI) {
  SmallPtrSet<Instruction *, 8> Visisted;
  Visisted.insert(&AssumeCall);
  SmallVector<Instruction *, 8> Worklist;
  Worklist.push_back(OutlinedI);

  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    if (isa<PHINode>(I) || I->isTerminator() || isa<AllocaInst>(I))
      continue;
    if (!Visisted.insert(I).second)
      continue;
    for (Value *Op : I->operands())
      if (auto *OpI = dyn_cast<Instruction>(Op))
        Worklist.push_back(OpI);
  }

  SmallPtrSet<Instruction *, 8> Cloned;
  SmallPtrSet<Instruction *, 8> Clones;
  for (Instruction *I : Visisted) {
    Worklist.clear();
    Instruction *Clone = nullptr;
    for (User *Usr : I->users())
      if (Instruction *UserI = dyn_cast<Instruction>(Usr)) {
        if (!Cloned.count(UserI) && Visisted.count(UserI))
          Worklist.push_back(UserI);
        else if (!Clone)
          Clone = I->clone();
        if (Clones.count(UserI))
          Worklist.push_back(UserI);
      }
    if (!Clone)
      continue;
    Clones.insert(Clone);
    Cloned.insert(I);
    Clone->insertAfter(I);
    Clone->setName(I->getName() + ".clone");
    if (OutlinedI == I)
      OutlinedI = Clone;
    while (!Worklist.empty())
      Worklist.pop_back_val()->replaceUsesOfWith(I, Clone);
  }
  return OutlinedI;
}

static bool outline(IntrinsicInst &AssumeCall, AssumptionCache &AC,
                    AlignmentFromAssumptionsPass &AFAP) {
  Instruction *Op = dyn_cast<Instruction>(AssumeCall.getOperand(0));
  if (!Op || (!DuplicateInst && Op->getNumUses() > 1) ||
      (isa<IntrinsicInst>(Op) &&
       cast<IntrinsicInst>(Op)->getIntrinsicID() == Intrinsic::type_test))
    return false;

  SmallVector<OperandBundleDefT<Value *>, 4> OpBundleOps;
  SmallVector<Value *, 8> Operands;

  Value *AAPtr;
  const SCEV *AlignSCEV, *OffSCEV;
  if (AFAP.extractAlignmentInfo(&AssumeCall, AAPtr, AlignSCEV, OffSCEV)) {
    Operands.push_back(AAPtr);
    SCEVExpander Expander(*AFAP.SE, AssumeCall.getModule()->getDataLayout(),
                          "");
    Expander.setInsertPoint(&AssumeCall);
    Operands.push_back(Expander.expandCodeFor(AlignSCEV));
    Operands.push_back(Expander.expandCodeFor(OffSCEV));
    OperandBundleDefT<Value *> AlignOpB("align", Operands);
    OpBundleOps.push_back(AlignOpB);
  } else if (!Aggressive)
    return false;

  AC.unregisterAssumption(&AssumeCall);

  // Find the instructions we want to outline and an order in which we can put
  // them later.
  SmallVector<Instruction *, 32> OutlinedInstsOrdered;
  DenseMap<Instruction *, int> NonOutlinedUsesMap;
  if (DuplicateInst)
    Op = duplicateInsts(AssumeCall, Op);
  markAsOutlined(*Op, OutlinedInstsOrdered, NonOutlinedUsesMap);
  assert(!OutlinedInstsOrdered.empty());

  // Isolate the call into its own basic block.
  SplitBlock(AssumeCall.getParent(), &AssumeCall);
  SplitBlock(AssumeCall.getParent(), AssumeCall.getNextNode());

  // Move the instruction into the new block and order them.
  while (!OutlinedInstsOrdered.empty()) {
    Instruction *I = OutlinedInstsOrdered.pop_back_val();
    I->moveBefore(&AssumeCall);
  }

  CodeExtractorAnalysisCache CEAC(*AssumeCall.getFunction());
  CodeExtractor CE(ArrayRef<BasicBlock *>({AssumeCall.getParent()}));

  Function *OutlinedFn = CE.extractCodeRegion(CEAC);

  assert(OutlinedFn->getNumUses() == 1);
  CallInst *DirectCall = cast<CallInst>(OutlinedFn->user_back());
  OutlinedFn->setName("__assumption_in_" + DirectCall->getCaller()->getName());

  Operands.clear();
  Operands.push_back(OutlinedFn);
  Operands.append(DirectCall->value_op_begin(), --DirectCall->value_op_end());
  OperandBundleDefT<Value *> AssumeFnOpB("assume_fn", Operands);
  OpBundleOps.push_back(AssumeFnOpB);

  LLVMContext &Ctx = OutlinedFn->getContext();
  CallInst *NewAssumeCall = CallInst::Create(AssumeCall.getCalledFunction(),
                                             {ConstantInt::getTrue(Ctx)},
                                             OpBundleOps, "", DirectCall);

  unsigned Merged = 0;
  Merged += MergeBlockIntoPredecessor(
      NewAssumeCall->getParent()->getUniqueSuccessor());
  Merged += MergeBlockIntoPredecessor(NewAssumeCall->getParent());
  Merged +=
      MergeBlockIntoPredecessor(AssumeCall.getParent()->getUniqueSuccessor());
  Merged += MergeBlockIntoPredecessor(AssumeCall.getParent());
  assert(Merged == 4);

  // Remove the direct call.
  DirectCall->eraseFromParent();
  assert(OutlinedFn->getNumUses() == 1);

  AC.registerAssumption(NewAssumeCall);
  assert(NewAssumeCall->hasOperandBundles());

  return true;
}

namespace {
struct AssumptionOutliner : public ModulePass {
  AssumptionOutliner() : ModulePass(ID) {}
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
  }
  static char ID;
};
} // namespace

bool AssumptionOutliner::runOnModule(Module &M) {
  Function *Assume = M.getFunction("llvm.assume");
  if (!Assume)
    return false;
  SmallVector<IntrinsicInst *, 8> Uses;
  for (Use &U : Assume->uses())
    if (auto *II = dyn_cast<IntrinsicInst>(U.getUser()))
      Uses.push_back(II);

  bool Changed = false;
  for (IntrinsicInst *II : Uses) {
    Function *Fn = II->getFunction();
    auto &ACU = getAnalysis<AssumptionCacheTracker>();
    ScalarEvolution *SE = &getAnalysis<ScalarEvolutionWrapperPass>(*Fn).getSE();
    AlignmentFromAssumptionsPass AFAP;
    AFAP.SE = SE;

    auto &AC = ACU.getAssumptionCache(*Fn);
    Changed |= outline(*II, AC, AFAP);
  }

  assert(!verifyModule(M, &errs()));
  return Changed;
}

char AssumptionOutliner::ID = 0;

namespace llvm {
ModulePass *createAssumptionOutlinerPass() { return new AssumptionOutliner(); }
} // namespace llvm

INITIALIZE_PASS_BEGIN(AssumptionOutliner, "assumption-outliner",
                      "Assumption outliner", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_END(AssumptionOutliner, "assumption-outliner",
                    "Assumption outliner", false, false)
