//===- MustExecute.cpp - Printer for isGuaranteedToExecute ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MustExecute.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

const DenseMap<BasicBlock *, ColorVector> &
LoopSafetyInfo::getBlockColors() const {
  return BlockColors;
}

void LoopSafetyInfo::copyColors(BasicBlock *New, BasicBlock *Old) {
  ColorVector &ColorsForNewBlock = BlockColors[New];
  ColorVector &ColorsForOldBlock = BlockColors[Old];
  ColorsForNewBlock = ColorsForOldBlock;
}

bool SimpleLoopSafetyInfo::blockMayThrow(const BasicBlock *BB) const {
  (void)BB;
  return anyBlockMayThrow();
}

bool SimpleLoopSafetyInfo::anyBlockMayThrow() const {
  return MayThrow;
}

void SimpleLoopSafetyInfo::computeLoopSafetyInfo(const Loop *CurLoop) {
  assert(CurLoop != nullptr && "CurLoop can't be null");
  BasicBlock *Header = CurLoop->getHeader();
  // Iterate over header and compute safety info.
  HeaderMayThrow = !isGuaranteedToTransferExecutionToSuccessor(Header);
  MayThrow = HeaderMayThrow;
  // Iterate over loop instructions and compute safety info.
  // Skip header as it has been computed and stored in HeaderMayThrow.
  // The first block in loopinfo.Blocks is guaranteed to be the header.
  assert(Header == *CurLoop->getBlocks().begin() &&
         "First block must be header");
  for (Loop::block_iterator BB = std::next(CurLoop->block_begin()),
                            BBE = CurLoop->block_end();
       (BB != BBE) && !MayThrow; ++BB)
    MayThrow |= !isGuaranteedToTransferExecutionToSuccessor(*BB);

  computeBlockColors(CurLoop);
}

bool ICFLoopSafetyInfo::blockMayThrow(const BasicBlock *BB) const {
  return ICF.hasICF(BB);
}

bool ICFLoopSafetyInfo::anyBlockMayThrow() const {
  return MayThrow;
}

void ICFLoopSafetyInfo::computeLoopSafetyInfo(const Loop *CurLoop) {
  assert(CurLoop != nullptr && "CurLoop can't be null");
  ICF.clear();
  MW.clear();
  MayThrow = false;
  // Figure out the fact that at least one block may throw.
  for (auto &BB : CurLoop->blocks())
    if (ICF.hasICF(&*BB)) {
      MayThrow = true;
      break;
    }
  computeBlockColors(CurLoop);
}

void ICFLoopSafetyInfo::insertInstructionTo(const Instruction *Inst,
                                            const BasicBlock *BB) {
  ICF.insertInstructionTo(Inst, BB);
  MW.insertInstructionTo(Inst, BB);
}

void ICFLoopSafetyInfo::removeInstruction(const Instruction *Inst) {
  ICF.removeInstruction(Inst);
  MW.removeInstruction(Inst);
}

void LoopSafetyInfo::computeBlockColors(const Loop *CurLoop) {
  // Compute funclet colors if we might sink/hoist in a function with a funclet
  // personality routine.
  Function *Fn = CurLoop->getHeader()->getParent();
  if (Fn->hasPersonalityFn())
    if (Constant *PersonalityFn = Fn->getPersonalityFn())
      if (isScopedEHPersonality(classifyEHPersonality(PersonalityFn)))
        BlockColors = colorEHFunclets(*Fn);
}

/// Return true if we can prove that the given ExitBlock is not reached on the
/// first iteration of the given loop.  That is, the backedge of the loop must
/// be executed before the ExitBlock is executed in any dynamic execution trace.
static bool CanProveNotTakenFirstIteration(const BasicBlock *ExitBlock,
                                           const DominatorTree *DT,
                                           const Loop *CurLoop) {
  auto *CondExitBlock = ExitBlock->getSinglePredecessor();
  if (!CondExitBlock)
    // expect unique exits
    return false;
  assert(CurLoop->contains(CondExitBlock) && "meaning of exit block");
  auto *BI = dyn_cast<BranchInst>(CondExitBlock->getTerminator());
  if (!BI || !BI->isConditional())
    return false;
  // If condition is constant and false leads to ExitBlock then we always
  // execute the true branch.
  if (auto *Cond = dyn_cast<ConstantInt>(BI->getCondition()))
    return BI->getSuccessor(Cond->getZExtValue() ? 1 : 0) == ExitBlock;
  auto *Cond = dyn_cast<CmpInst>(BI->getCondition());
  if (!Cond)
    return false;
  // todo: this would be a lot more powerful if we used scev, but all the
  // plumbing is currently missing to pass a pointer in from the pass
  // Check for cmp (phi [x, preheader] ...), y where (pred x, y is known
  auto *LHS = dyn_cast<PHINode>(Cond->getOperand(0));
  auto *RHS = Cond->getOperand(1);
  if (!LHS || LHS->getParent() != CurLoop->getHeader())
    return false;
  auto DL = ExitBlock->getModule()->getDataLayout();
  auto *IVStart = LHS->getIncomingValueForBlock(CurLoop->getLoopPreheader());
  auto *SimpleValOrNull = SimplifyCmpInst(Cond->getPredicate(),
                                          IVStart, RHS,
                                          {DL, /*TLI*/ nullptr,
                                              DT, /*AC*/ nullptr, BI});
  auto *SimpleCst = dyn_cast_or_null<Constant>(SimpleValOrNull);
  if (!SimpleCst)
    return false;
  if (ExitBlock == BI->getSuccessor(0))
    return SimpleCst->isZeroValue();
  assert(ExitBlock == BI->getSuccessor(1) && "implied by above");
  return SimpleCst->isAllOnesValue();
}

/// Collect all blocks from \p CurLoop which lie on all possible paths from
/// the header of \p CurLoop (inclusive) to BB (exclusive) into the set
/// \p Predecessors. If \p BB is the header, \p Predecessors will be empty.
static void collectTransitivePredecessors(
    const Loop *CurLoop, const BasicBlock *BB,
    SmallPtrSetImpl<const BasicBlock *> &Predecessors) {
  assert(Predecessors.empty() && "Garbage in predecessors set?");
  assert(CurLoop->contains(BB) && "Should only be called for loop blocks!");
  if (BB == CurLoop->getHeader())
    return;
  SmallVector<const BasicBlock *, 4> WorkList;
  for (auto *Pred : predecessors(BB)) {
    Predecessors.insert(Pred);
    WorkList.push_back(Pred);
  }
  while (!WorkList.empty()) {
    auto *Pred = WorkList.pop_back_val();
    assert(CurLoop->contains(Pred) && "Should only reach loop blocks!");
    // We are not interested in backedges and we don't want to leave loop.
    if (Pred == CurLoop->getHeader())
      continue;
    // TODO: If BB lies in an inner loop of CurLoop, this will traverse over all
    // blocks of this inner loop, even those that are always executed AFTER the
    // BB. It may make our analysis more conservative than it could be, see test
    // @nested and @nested_no_throw in test/Analysis/MustExecute/loop-header.ll.
    // We can ignore backedge of all loops containing BB to get a sligtly more
    // optimistic result.
    for (auto *PredPred : predecessors(Pred))
      if (Predecessors.insert(PredPred).second)
        WorkList.push_back(PredPred);
  }
}

bool LoopSafetyInfo::allLoopPathsLeadToBlock(const Loop *CurLoop,
                                             const BasicBlock *BB,
                                             const DominatorTree *DT) const {
  assert(CurLoop->contains(BB) && "Should only be called for loop blocks!");

  // Fast path: header is always reached once the loop is entered.
  if (BB == CurLoop->getHeader())
    return true;

  // Collect all transitive predecessors of BB in the same loop. This set will
  // be a subset of the blocks within the loop.
  SmallPtrSet<const BasicBlock *, 4> Predecessors;
  collectTransitivePredecessors(CurLoop, BB, Predecessors);

  // Make sure that all successors of, all predecessors of BB which are not
  // dominated by BB, are either:
  // 1) BB,
  // 2) Also predecessors of BB,
  // 3) Exit blocks which are not taken on 1st iteration.
  // Memoize blocks we've already checked.
  SmallPtrSet<const BasicBlock *, 4> CheckedSuccessors;
  for (auto *Pred : Predecessors) {
    // Predecessor block may throw, so it has a side exit.
    if (blockMayThrow(Pred))
      return false;

    // BB dominates Pred, so if Pred runs, BB must run.
    // This is true when Pred is a loop latch.
    if (DT->dominates(BB, Pred))
      continue;

    for (auto *Succ : successors(Pred))
      if (CheckedSuccessors.insert(Succ).second &&
          Succ != BB && !Predecessors.count(Succ))
        // By discharging conditions that are not executed on the 1st iteration,
        // we guarantee that *at least* on the first iteration all paths from
        // header that *may* execute will lead us to the block of interest. So
        // that if we had virtually peeled one iteration away, in this peeled
        // iteration the set of predecessors would contain only paths from
        // header to BB without any exiting edges that may execute.
        //
        // TODO: We only do it for exiting edges currently. We could use the
        // same function to skip some of the edges within the loop if we know
        // that they will not be taken on the 1st iteration.
        //
        // TODO: If we somehow know the number of iterations in loop, the same
        // check may be done for any arbitrary N-th iteration as long as N is
        // not greater than minimum number of iterations in this loop.
        if (CurLoop->contains(Succ) ||
            !CanProveNotTakenFirstIteration(Succ, DT, CurLoop))
          return false;
  }

  // All predecessors can only lead us to BB.
  return true;
}

/// Returns true if the instruction in a loop is guaranteed to execute at least
/// once.
bool SimpleLoopSafetyInfo::isGuaranteedToExecute(const Instruction &Inst,
                                                 const DominatorTree *DT,
                                                 const Loop *CurLoop) const {
  // If the instruction is in the header block for the loop (which is very
  // common), it is always guaranteed to dominate the exit blocks.  Since this
  // is a common case, and can save some work, check it now.
  if (Inst.getParent() == CurLoop->getHeader())
    // If there's a throw in the header block, we can't guarantee we'll reach
    // Inst unless we can prove that Inst comes before the potential implicit
    // exit.  At the moment, we use a (cheap) hack for the common case where
    // the instruction of interest is the first one in the block.
    return !HeaderMayThrow ||
           Inst.getParent()->getFirstNonPHIOrDbg() == &Inst;

  // If there is a path from header to exit or latch that doesn't lead to our
  // instruction's block, return false.
  return allLoopPathsLeadToBlock(CurLoop, Inst.getParent(), DT);
}

bool ICFLoopSafetyInfo::isGuaranteedToExecute(const Instruction &Inst,
                                              const DominatorTree *DT,
                                              const Loop *CurLoop) const {
  return !ICF.isDominatedByICFIFromSameBlock(&Inst) &&
         allLoopPathsLeadToBlock(CurLoop, Inst.getParent(), DT);
}

bool ICFLoopSafetyInfo::doesNotWriteMemoryBefore(const BasicBlock *BB,
                                                 const Loop *CurLoop) const {
  assert(CurLoop->contains(BB) && "Should only be called for loop blocks!");

  // Fast path: there are no instructions before header.
  if (BB == CurLoop->getHeader())
    return true;

  // Collect all transitive predecessors of BB in the same loop. This set will
  // be a subset of the blocks within the loop.
  SmallPtrSet<const BasicBlock *, 4> Predecessors;
  collectTransitivePredecessors(CurLoop, BB, Predecessors);
  // Find if there any instruction in either predecessor that could write
  // to memory.
  for (auto *Pred : Predecessors)
    if (MW.mayWriteToMemory(Pred))
      return false;
  return true;
}

bool ICFLoopSafetyInfo::doesNotWriteMemoryBefore(const Instruction &I,
                                                 const Loop *CurLoop) const {
  auto *BB = I.getParent();
  assert(CurLoop->contains(BB) && "Should only be called for loop blocks!");
  return !MW.isDominatedByMemoryWriteFromSameBlock(&I) &&
         doesNotWriteMemoryBefore(BB, CurLoop);
}

namespace {
  struct MustExecutePrinter : public FunctionPass {

    static char ID; // Pass identification, replacement for typeid
    MustExecutePrinter() : FunctionPass(ID) {
      initializeMustExecutePrinterPass(*PassRegistry::getPassRegistry());
    }
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
    }
    bool runOnFunction(Function &F) override;
  };
  struct MustBeExecutedContextPrinter : public ModulePass {
    static char ID;

    MustBeExecutedContextPrinter() : ModulePass(ID) {
      initializeMustBeExecutedContextPrinterPass(*PassRegistry::getPassRegistry());
    }
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
      AU.addRequired<PostDominatorTreeWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
    }
    bool runOnModule(Module &M) override;
  };
}

char MustExecutePrinter::ID = 0;
INITIALIZE_PASS_BEGIN(MustExecutePrinter, "print-mustexecute",
                      "Instructions which execute on loop entry", false, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(MustExecutePrinter, "print-mustexecute",
                    "Instructions which execute on loop entry", false, true)

FunctionPass *llvm::createMustExecutePrinter() {
  return new MustExecutePrinter();
}

char MustBeExecutedContextPrinter::ID = 0;
INITIALIZE_PASS_BEGIN(
    MustBeExecutedContextPrinter, "print-must-be-executed-contexts",
    "print the must-be-executed-contexed for all instructions", false, true)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(MustBeExecutedContextPrinter,
                    "print-must-be-executed-contexts",
                    "print the must-be-executed-contexed for all instructions",
                    false, true)

ModulePass *llvm::createMustBeExecutedContextPrinter() {
  return new MustBeExecutedContextPrinter();
}

static bool isMustExecuteIn(const Instruction &I, Loop *L, DominatorTree *DT) {
  // TODO: merge these two routines.  For the moment, we display the best
  // result obtained by *either* implementation.  This is a bit unfair since no
  // caller actually gets the full power at the moment.
  SimpleLoopSafetyInfo LSI;
  LSI.computeLoopSafetyInfo(L);
  return LSI.isGuaranteedToExecute(I, DT, L) ||
    isGuaranteedToExecuteForEveryIteration(&I, L);
}

namespace {
/// An assembly annotator class to print must execute information in
/// comments.
class MustExecuteAnnotatedWriter : public AssemblyAnnotationWriter {
  DenseMap<const Value*, SmallVector<Loop*, 4> > MustExec;

public:
  MustExecuteAnnotatedWriter(const Function &F,
                             DominatorTree &DT, LoopInfo &LI) {
    for (auto &I: instructions(F)) {
      Loop *L = LI.getLoopFor(I.getParent());
      while (L) {
        if (isMustExecuteIn(I, L, &DT)) {
          MustExec[&I].push_back(L);
        }
        L = L->getParentLoop();
      };
    }
  }

  void printInfoComment(const Value &V, formatted_raw_ostream &OS) override {
    if (!MustExec.count(&V))
      return;

    const auto &Loops = MustExec.lookup(&V);
    const auto NumLoops = Loops.size();
    if (NumLoops > 1)
      OS << " ; (mustexec in " << NumLoops << " loops: ";
    else
      OS << " ; (mustexec in: ";

    bool first = true;
    for (const Loop *L : Loops) {
      if (!first)
        OS << ", ";
      first = false;
      OS << L->getHeader()->getName();
    }
    OS << ")";
  }
};

bool MustExecutePrinter::runOnFunction(Function &F) {
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  MustExecuteAnnotatedWriter Writer(F, DT, LI);
  F.print(dbgs(), &Writer);

  return false;
}

bool MustBeExecutedContextPrinter::runOnModule(Module &M) {
  GetterTy<const LoopInfo> LIGetter =
      [&](const Function &F) -> const LoopInfo * {
    return &getAnalysis<LoopInfoWrapperPass>(const_cast<Function &>(F))
        .getLoopInfo();
  };
  GetterTy<const DominatorTree> DTGetter = [&](const Function &F) ->const  DominatorTree * {
    return &getAnalysis<DominatorTreeWrapperPass>(const_cast<Function &>(F))
        .getDomTree();
  };
  GetterTy<const PostDominatorTree> PDTGetter =
      [&](const Function &F) ->const  PostDominatorTree * {
    return &getAnalysis<PostDominatorTreeWrapperPass>(const_cast<Function &>(F))
        .getPostDomTree();
  };

  MustBeExecutedContextExplorer Explorer(true, true, true, true,
                                         LIGetter, DTGetter, PDTGetter);
  for (Function &F : M) {
    for (Instruction &I : instructions(F)) {
      dbgs() << "-- Explore context of: " << I << "\n";
      for (const Instruction *CI : Explorer.range(&I))
        dbgs() << "  [F: " << CI->getFunction()->getName() << "] " << *CI
               << "\n";
    }
  }

  return false;
}
} // namespace

/// Return true if \p L might be an endless loop.
static bool maybeEndlessLoop(const Loop &L) {
  // TODO: Actually try to prove it is not.
  return true;
}

/// Return an instruction that is always executed before the function \p F is
/// left, assuming it is left through a return.
static const Instruction *findFunctionExitJoinPoint(const Function *F,
                                                    const DominatorTree *DT) {
  const BasicBlock *JoinBB = nullptr;
  for (const BasicBlock &BB : *F) {
    // Skip all but return instructions.
    if (!isa<ReturnInst>(BB.getTerminator()))
      continue;

    // The first return instruction found is the initial join point.
    if (!JoinBB) {
      JoinBB = &BB;
      continue;
    }

    // When we find more return instructions the nearest common dominator of all
    // of them is known to be executed prior to all of them.
    if (DT) {
      JoinBB = DT->findNearestCommonDominator(JoinBB, &BB);
      assert(JoinBB && "Assumed a common dominator!");
      continue;
    }

    // If we do no have a dominator tree we still know that *at least* the entry
    // block is executed as we assume the function is left through a return.
    // TODO: Improve this by using getMustBeExecutedNextInstruction() from this
    //       point.
    return F->getEntryBlock().getTerminator();
  }

  // If we traversed all the blocks and found the nearest common dominator we
  // return it. If we did not find a return instruction we return a nullptr but
  // we should indicate a problem instead because the function is never left
  // through a return.
  return JoinBB ? JoinBB->getTerminator() : nullptr;
}

template <typename K, typename V, typename FnTy, typename... ArgsTy>
static V getOrCreateCachedOptional(K Key, DenseMap<K, Optional<V>> &Map,
                                   FnTy &&Fn, ArgsTy... args) {
  Optional<V> &OptVal = Map[Key];
  if (!OptVal.hasValue())
    OptVal = Fn(args...);
  return OptVal.getValue();
}

template <typename DomTreeType, typename AdjRangeTy>
static const BasicBlock *
findJoinPoint(ExplorationDirection Direction, const BasicBlock *InitBB,
              DenseMap<const BasicBlock *, Optional<bool>> &BlockTransferMap,
              const LoopInfo *LI, const DomTreeType *GDT, AdjRangeTy &AdjRange) {
  SmallVector<const BasicBlock *, 8> Worklist;
  for (const BasicBlock *AdjacentBB : AdjRange(InitBB))
    Worklist.push_back(AdjacentBB);
  if (Worklist.empty())
    return nullptr;
  if (Worklist.size() == 1)
    return Worklist[0];

  auto GetSingleAdjBB = [&](const BasicBlock *BB) -> const BasicBlock * {
    const BasicBlock *SingleAdjBB = nullptr;
    for (const BasicBlock *AdjacentBB : AdjRange(BB)) {
      if (SingleAdjBB)
        return nullptr;
      SingleAdjBB = AdjacentBB;
    }
    return SingleAdjBB;
  };

  const BasicBlock *JoinBB = nullptr;
  if (GDT) {
    const auto *InitNode = GDT->getNode(InitBB);
    assert(InitNode && "No (post)dominator tree node found, dead block?");
    const auto *IDomNode = InitNode->getIDom();
    if (IDomNode)
      JoinBB = IDomNode->getBlock();
  } else {
    if (Worklist.size() != 2)
      return nullptr;

    const BasicBlock *Adj0 = Worklist[0];
    const BasicBlock *Adj1 = Worklist[1];
    const BasicBlock *Adj0SingleAdj = GetSingleAdjBB(Adj0);
    const BasicBlock *Adj1SingleAdj = GetSingleAdjBB(Adj1);
    if (Adj0 == Adj1SingleAdj) {
      // cnd true -> joinBB
      // cnd false -> thenBB -> joinBB
      JoinBB = Adj0;
    } else if (Adj1 == Adj0SingleAdj) {
      // cnd true -> thenBB -> joinBB
      // cnd false -> joinBB
      JoinBB = Adj1;
    } else if (Adj0SingleAdj == Adj1SingleAdj) {
      // cnd true -> thenBB -> joinBB
      // cnd false -> elseBB -> joinBB
      JoinBB = Adj0SingleAdj;
    }
  }

  if (!JoinBB || Direction == ED_BACKWARD)
    return JoinBB;

  SmallPtrSet<const BasicBlock *, 16> Visited;
  while (!Worklist.empty()) {
    const BasicBlock *ToBB = Worklist.pop_back_val();
    if (ToBB == JoinBB)
      continue;
    if (!Visited.insert(ToBB).second) {
      const Loop *L = LI->getLoopFor(ToBB);
      if (!L || maybeEndlessLoop(*L))
        return nullptr;
      continue;
    }

    bool TransfersExecution = getOrCreateCachedOptional(
        ToBB, BlockTransferMap,
        [](const BasicBlock *BB) {
          return isGuaranteedToTransferExecutionToSuccessor(BB);
        },
        ToBB);
    if (!TransfersExecution)
      return nullptr;

    for (const BasicBlock *AdjacentBB : AdjRange(ToBB))
      Worklist.push_back(AdjacentBB);
  }

  return JoinBB;
}


const Instruction *
MustBeExecutedContextExplorer::getMustBeExecutedNextInstruction(
    MustBeExecutedIterator</* CachedOnly */ false> &It, const Instruction *PP) {

  // If we explore only inside a given basic block we stop at terminators.
  if (!ExploreCFGForward && PP->isTerminator())
    return nullptr;

  // If we explore the call graph (CG) forward and we see a call site we can
  // continue with the callee instructions if the callee has an exact
  // definition. If this is the case, we add the call site in the forward call
  // stack such that we can return to the position after the call and also
  // translate values for the user.
  if (ExploreCGForward) {
    if (ImmutableCallSite ICS = ImmutableCallSite(PP))
      if (const Function *F = ICS.getCalledFunction())
        if (F->hasExactDefinition()) {
          It.ForwardCallStack.push_back({PP});
          return &F->getEntryBlock().front();
        }
  }

  // Helper function to look at the forward call stack, pop the last call site,
  // and return the next instruction after it, assuming the call stack was not
  // empty.
  auto PopAndReturnForwardCallSiteNextInst = [&]() -> const Instruction * {
    // Check for a known call site on the forward call stack of the iterator.
    if (It.ForwardCallStack.empty())
      return nullptr;

    const Instruction *CS = It.ForwardCallStack.pop_back_val();
    return CS->getNextNode();
  };

  // The function that contains the current position.
  const Function *PPFunc = PP->getFunction();

  // At a return instruction we have two options that allow us to continue:
  // 1) We are in a function that we entered earlier, in this case we can
  //    simply pop the last call site from the forward call stack and return the
  //    instruction after the call site.
  // 2) We explore the call graph (CG) backwards trying to find a point that
  //    is know to be executed every time when the current function is.
  if (isa<ReturnInst>(PP)) {
    // Check for a known call site on the forward call stack of the iterator.
    if (!It.ForwardCallStack.empty())
      return PopAndReturnForwardCallSiteNextInst();

    // Check if backwards call graph exploration is allowed.
    if (!ExploreCGBackward)
      return nullptr;

    // We do not know all the callers for non-internal functions.
    if (!PPFunc->hasInternalLinkage())
      return nullptr;

    // TODO: We restrict it to a single call site for now but we could allow
    //       more and find a join point interprocedurally.
    if (PPFunc->getNumUses() != 1)
      return nullptr;

    // Make sure the user is a direct call.
    // TODO: Indirect calls can be fine too.
    ImmutableCallSite ICS(PPFunc->user_back());
    if (!ICS || ICS.getCalledFunction() != PPFunc)
      return nullptr;

    // We know we reached the return instruction of the callee so we will
    // continue at the next instruction after the call.
    return ICS.getInstruction()->getNextNode();
  }

  // If we do not traverse the call graph we check if we can make progress in
  // the current function. First, check if the instruction is guaranteed to
  // transfer execution to the successor.
  bool TransfersExecution = isGuaranteedToTransferExecutionToSuccessor(PP);

  // If this is not a terminator we know that there is a single instruction
  // after this one that is executed next if control is transfered. If not,
  // we can try to go back to a call site we entered earlier. If none exists, we
  // do not know any instruction that has to be executd next.
  if (!PP->isTerminator()) {
    if (TransfersExecution)
      return PP->getNextNode();

    // Try to continue at a known call site on the forward call stack of the
    // iterator.
    return PopAndReturnForwardCallSiteNextInst();
  }

  // Finally, we have to handle terminators, trivial ones first.
  assert(PP->isTerminator() && "Expected a terminator!");

  // A terminator without a successor which is not a return is not handled yet.
  // We can still try to continue at a known call site on the forward call stack
  // of the iterator.
  if (PP->getNumSuccessors() == 0)
    return PopAndReturnForwardCallSiteNextInst();

  // A terminator with a single successor, we will continue at the beginning of
  // that one.
  if (PP->getNumSuccessors() == 1)
    return &PP->getSuccessor(0)->front();

  // Multiple successors mean we need to find the join point where control flow
  // converges again. We use the findJoinPoint helper function with information
  // about the function and helper analyses, if available.
  const LoopInfo *LI = LIGetter(*PPFunc);
  const PostDominatorTree *PDT = PDTGetter(*PPFunc);

  auto AdjRange = [](const BasicBlock *BB) { return successors(BB); };
  if (const BasicBlock *JoinBB = findJoinPoint(
          ED_FORWARD, PP->getParent(), BlockTransferMap, LI, PDT, AdjRange))
    return &JoinBB->front();

  // No join point was found but we can still try to continue at a known call
  // site on the forward call stack of the iterator.
  return PopAndReturnForwardCallSiteNextInst();
}

const Instruction *
MustBeExecutedContextExplorer::getMustBeExecutedPrevInstruction(
    MustBeExecutedIterator</* CachedOnly */ false> &It, const Instruction *PP) {
  bool IsFirst = !(PP->getPrevNode());
  //errs() << "PREV: of " << PP << " : " << IsFirst << "\n";
  //errs() << "PREV: of " << *PP << "\n";

  // If we explore only inside a given basic block we stop at the first
  // instruction.
  if (!ExploreCFGBackward && IsFirst)
    return nullptr;

  // Helper function to look at the backward call stack, pop the last call site,
  // and return the call site instruction, assuming a non-empty call stack.
  auto PopAndReturnBackwardCallSiteInst = [&]() -> const Instruction * {
    // Check for a known call site on the caller stack of the iterator.
    if (It.BackwardCallStack.empty())
      return nullptr;

    const Instruction *CS = It.BackwardCallStack.pop_back_val();
    return CS;
  };

  // The block and function that contains the current position.
  const BasicBlock *PPBlock = PP->getParent();
  const Function *PPFunc = PPBlock->getParent();

  // At a first instruction in a function we have two options that allow us to
  // continue:
  // 1) We are in a function that we entered earlier, in this case we can
  //    simply pop the last call site from the backward call stack and
  //    return the instruction before the call site.
  // 2) We explore the call graph (CG) backwards trying to find a point that
  //    is know to be executed every time when the current function is.
  if (IsFirst && PPBlock == &PPFunc->getEntryBlock()) {
    // Check for a known call site on the backward call stack of the
    // iterator.
    if (!It.BackwardCallStack.empty())
      return PopAndReturnBackwardCallSiteInst();

    // Check if backwards call graph exploration is allowed.
    if (!ExploreCGBackward)
      return nullptr;

    // We do not know all the callers for non-internal functions.
    if (!PPFunc->hasInternalLinkage())
      return nullptr;

    // TODO: We restrict it to a single call site for now but we could allow
    //       more and find a join point interprocedurally.
    if (PPFunc->getNumUses() != 1)
      return nullptr;

    // Make sure the user is a direct call.
    // TODO: Indirect calls can be fine too.
    ImmutableCallSite ICS(PPFunc->user_back());
    if (!ICS || ICS.getCalledFunction() != PPFunc)
      return nullptr;

    // We know we reached the first instruction of the callee so we must have
    // executed the instruction before the call.
    return ICS.getInstruction();
  }

  // Ready the dominator tree if available.
  const DominatorTree *DT = DTGetter(*PPFunc);

  // If we are inside a block we know what instruction was executed before, the
  // previous one. However, if the previous one is a call site, we can enter the
  // callee and visit its instructions as well.
  if (!IsFirst) {
    //errs() << "not first\n";
    const Instruction *PrevPP = PP->getPrevNode();
    // If we explore the call graph (CG) forward and the instruction before the
    // current one is a call site we can continue with the callee instructions
    // if the callee has an exact definition. If this is the case, we add the
    // call site in the backward call stack such that we can return to
    // the position of the call and also translate values for the user.
    if (ExploreCGForward) {
      if (ImmutableCallSite ICS = ImmutableCallSite(PrevPP))
        if (const Function *F = ICS.getCalledFunction())
          if (F->hasExactDefinition())
            if (const Instruction *JoinPP = getOrCreateCachedOptional(
                    F, FunctionExitJoinPointMap, findFunctionExitJoinPoint, F,
                    DT)) {
              //errs() << "FJPP: " << JoinPP << "\n";
              //errs() << "FJPP: " << *JoinPP << "\n";
              It.BackwardCallStack.push_back({PrevPP});
              return JoinPP;
            }
    }

    // We did not enter a callee so we simply return the previous instruction.
    return PrevPP;
  }
  //errs() << "first\n";

  // Finally, we have to handle the case where the program point is the first in
  // a block but not in the function. We use the findJoinPoint helper function
  // with information about the function and helper analyses, if available.
  const LoopInfo *LI = LIGetter(*PPFunc);
  auto AdjRange = [](const BasicBlock *BB) { return predecessors(BB); };
  if (const BasicBlock *JoinBB = findJoinPoint(
          ED_BACKWARD, PPBlock, BlockTransferMap, LI, DT, AdjRange))
    return &JoinBB->back();

  // No join point was found but we can still try to continue at a known call
  // site on the backward call stack of the iterator.
  return PopAndReturnBackwardCallSiteInst();
}

void MustBeExecutedContextExplorer::explore(
    MustBeExecutedIterator</* CachedOnly */ false> &It,
    MustBeExecutedInterval::Position &Pos, ExplorationDirection Direction) {
  assert(Pos && Pos.Interval->isInbounds(Pos.Offset) &&
         "Explore called for an invalid position!");
  assert(!Pos.Interval->isInbounds(Pos.Offset +
                                   (Direction == ED_FORWARD ? 1 : -1)) &&
         "Explore called for a position that can be advanced!");
  assert(!(Pos + Direction) &&
         "Explore called for a position that can be advanced!");

  // Check the call stack before we compute the advanced program point because
  // doing so changes the call stacks.
  unsigned CallStackSizeBefore =
      (Direction == ED_FORWARD ? It.ForwardCallStack.size()
                               : It.BackwardCallStack.size());

  const Instruction *PP = Pos.getInstruction();
  //errs() << "Direction: " << (Direction == ED_FORWARD ? "FORWARD" : "BACKWARD") << " :: ";
  //Pos.print(errs());
  //errs() << " :: PP " << *PP << " [CSSB: " << CallStackSizeBefore << "]\n";
  const Instruction *AdvancedPP =
      Direction == ED_FORWARD ? getMustBeExecutedNextInstruction(It, PP)
                              : getMustBeExecutedPrevInstruction(It, PP);
  //errs() << "APP : "<< AdvancedPP << "\n";

  // If we failed to get an advanced program point in the requested direction
  // there is nothing we can do.
  if (!AdvancedPP) {
    Pos = MustBeExecutedInterval::Position();
    return;
  }
  //errs() << "APP : "<< *AdvancedPP << "\n";

  // To avoid chasing around when we encounter an endless loop or recusion we
  // keep track of entered blocks.

  unsigned CallStackSizeAfter =
      (Direction == ED_FORWARD ? It.ForwardCallStack.size()
                               : It.BackwardCallStack.size());
  bool CallStackChanged = CallStackSizeBefore != CallStackSizeAfter;
  if (!It.Visited.insert({AdvancedPP, Direction}).second) {
    Pos = MustBeExecutedInterval::Position();
    return;
  }

  // If we do not have an interval position for the advanced program point we
  // either add it to the interval of the current position or create a new one
  // to hold the advanced program point, depending on the backward check above.
  MustBeExecutedInterval::Position AdvancedPPPos =
      lookupIntervalPosition(AdvancedPP);
  //errs()  << "APPPP: ";
  //AdvancedPPPos.print(errs());
  //errs()  << "\n";

  // TODO
  bool AdvancedIntervalConnectsToCurrent =
      AdvancedPPPos.Interval &&
      (Direction == ED_FORWARD ? AdvancedPPPos.Interval->Prev == Pos.Interval
                               : AdvancedPPPos.Interval->Next == Pos.Interval);

  bool CanExtendPPPos = !CallStackChanged;
  if (CanExtendPPPos && !AdvancedIntervalConnectsToCurrent) {
    // Go back from the advanced program point in the opposite direction in
    // order to see if we can extend the interval of the current position or if
    // we need a new interval.
    const Instruction *BackadvancedAdvancedPP =
        Direction == ED_FORWARD
            ? getMustBeExecutedPrevInstruction(It, AdvancedPP)
            : getMustBeExecutedNextInstruction(It, AdvancedPP);
    //errs() << "BAPP: " << BackadvancedAdvancedPP << "\n";
    CanExtendPPPos = (BackadvancedAdvancedPP == PP);
}

  if (!AdvancedPPPos && CanExtendPPPos) {
    //errs() << "Append!\n";
    Pos.Interval->append(AdvancedPP, Direction);
    Pos += Direction;
    MustBeExecutedMap[AdvancedPP] = Pos;
    return;
  }

  if (!AdvancedPPPos) {
    AdvancedPPPos = getOrCreateIntervalPosition(AdvancedPP);
    assert(AdvancedPPPos && AdvancedPPPos.getInstruction() == AdvancedPP &&
           "Position did not match the instruction!");
  }
  //errs()  << "APPPP: ";
  //AdvancedPPPos.print(errs(), true);
  //AdvancedPPPos.getInstruction()->dump();
  //errs() << " @@@ " << AdvancedPPPos.Interval->NextInsts.size();
  //errs()  << " ::: " << AdvancedIntervalConnectsToCurrent << "\n";

  if (CallStackChanged) {
    Pos = AdvancedPPPos;
    return;
  }

  // If we could not extend the interval associated with position we have to
  // create a link to the interval of the advanced position now. We know that
  // there is no link in the requested direction as we otherwise would not have
  // explored.
  if (Direction == ED_FORWARD) {
    assert(!Pos.Interval->Next &&
           "Explore called for a position with an existing link!");
    //errs() << "Next 1!\n";
    Pos.Interval->Next = AdvancedPPPos.Interval;
  } else {
    assert(!Pos.Interval->Prev &&
           "Explore called for a position with an existing link!");
    //errs() << "Prev 1!\n";
    Pos.Interval->Prev = AdvancedPPPos.Interval;
  }

  // If the advanced position had an associated interval, e.g., due to a call to
  // MustBeExecutedContextExplorer::begin(AdvancedPPP), but we know we can
  // extend the interval of the current position, we can add backwards links to
  // the advanced position interval now.
  if (CanExtendPPPos) {
    if (Direction == ED_FORWARD) {
      assert((AdvancedIntervalConnectsToCurrent ||
              !AdvancedPPPos.Interval->Prev) &&
             "Explore called for a position not properly linked before!");
      //errs() << "FCS: " << It.ForwardCallStack.size() << "\n";
      //errs() << "BCS: " << It.BackwardCallStack.size() << "\n";
      assert((*AdvancedPPPos.Interval)[0]->getFunction() ==
             (*Pos.Interval)[0]->getFunction());
      AdvancedPPPos.Interval->Prev = Pos.Interval;
    } else {
      assert((AdvancedIntervalConnectsToCurrent ||
              !AdvancedPPPos.Interval->Next) &&
             "Explore called for a position not properly linked before!");
      //errs() << "FCS: " << It.ForwardCallStack.size() << "\n";
      //errs() << "BCS: " << It.BackwardCallStack.size() << "\n";
      assert((*AdvancedPPPos.Interval)[0]->getFunction() ==
             (*Pos.Interval)[0]->getFunction());
      AdvancedPPPos.Interval->Next = Pos.Interval;
    }
  }

  Pos += Direction;
  return;
}

template <>
void MustBeExecutedIterator</* CachedOnly */ true>::explore(
    MustBeExecutedInterval::Position &Pos, ExplorationDirection Direction) {
  // No exploration in cached-only mode.
  Pos += Direction;
}

template <>
void MustBeExecutedIterator</* CachedOnly */ false>::explore(
    MustBeExecutedInterval::Position &Pos, ExplorationDirection Direction) {
  // Exploration is done by the explorer.
  MustBeExecutedInterval::Position NewPos = Pos + Direction;
  if (NewPos)
    Pos = NewPos;
  else
    Explorer.explore(*this, Pos, Direction);
}
