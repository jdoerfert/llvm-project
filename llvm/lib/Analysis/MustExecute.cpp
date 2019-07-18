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

#define DEBUG_TYPE "must-execute"

static cl::opt<bool> MustExecutePrinterUseExplorer(
    "print-mustexecute-use-explorer", cl::Hidden, cl::init(false),
    cl::desc(
        "Use 'must-be-executed-context' explorer for the mustexecute printer"));

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

static Optional<uint64_t>
getConstantIntegerValueInFirstIteration(const Value &V, const Loop *L,
                                        const DominatorTree *DT,
                                        const LoopInfo *LI) {
  if (auto *Cond = dyn_cast<ConstantInt>(&V))
    return Optional<uint64_t>(Cond->getZExtValue());

  if (auto *Cond = dyn_cast<CmpInst>(&V)) {
    // TODO: this would be a lot more powerful if we used scev, but all the
    //       plumbing is currently missing to pass a pointer in from the pass
    // Check for `cmp (phi [x, predecessor] ...), y` where `pred x, y` is known
    auto SimplifyPHI = [&](Value *V) -> Value * {
      auto *PHI = dyn_cast<PHINode>(V);
      if (!PHI)
        return V;
      // TODO: Remove the handling of a special loop in favor of the loop info
      //       solution once the user is gone.
      if (L && PHI->getParent() == L->getHeader() && L->getLoopPredecessor())
        return PHI->getIncomingValueForBlock(L->getLoopPredecessor());
      const Loop *PL = LI ? LI->getLoopFor(PHI->getParent()) : nullptr;
      if (PL && PL->getHeader() == PHI->getParent() && PL->getLoopPredecessor())
        return PHI->getIncomingValueForBlock(PL->getLoopPredecessor());
      return V;
    };
    auto *LHS = SimplifyPHI(Cond->getOperand(0));
    auto *RHS = SimplifyPHI(Cond->getOperand(1));

    const DataLayout &DL = Cond->getModule()->getDataLayout();
    auto *SimpleValOrNull =
        SimplifyCmpInst(Cond->getPredicate(), LHS, RHS,
                        {DL, /*TLI*/ nullptr, DT, /*AC*/ nullptr, Cond});
    if (auto *Cst = dyn_cast_or_null<ConstantInt>(SimpleValOrNull))
      return Optional<uint64_t>(Cst->getZExtValue());
  }

  return None;
}

static const BasicBlock *
getSuccessorInFirstIteration(const Instruction &TI, const Loop *L,
                             const DominatorTree *DT, const LoopInfo *LI) {
  assert(TI.isTerminator() && "Expected a terminator");
  if (auto *BI = dyn_cast<BranchInst>(&TI)) {
    if (BI->isUnconditional())
      return BI->getSuccessor(0);

    if (L || LI) {
      Optional<uint64_t> CV = getConstantIntegerValueInFirstIteration(
          *BI->getCondition(), L, DT, LI);
      if (CV.hasValue()) {
        assert(CV.getValue() < 2 && "Expected boolean value!");
        return BI->getSuccessor(1 - CV.getValue());
      }
    }
  }
  return nullptr;
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
  const BasicBlock * SuccInFirstIteration =
      getSuccessorInFirstIteration(*CondExitBlock->getTerminator(), CurLoop, DT,
                                   /* LoopInfo */ nullptr);
  if (SuccInFirstIteration)
    return SuccInFirstIteration != ExitBlock;
  return false;
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
  MustExecuteAnnotatedWriter(const Function &F, DominatorTree &DT,
                             LoopInfo &LI) {
    if (MustExecutePrinterUseExplorer) {
      MustBeExecutedLoopSafetyInfo<false> MBELSI(&DT, nullptr, &LI);

      SmallVector<Loop *, 16> Loops;
      Loops.append(LI.begin(), LI.end());
      // Perform the isGuaranteedToExecute check loop by loop to reuse cached
      // results computed by computeLoopSafetyInfo.
      while (!Loops.empty()) {
        Loop *L = Loops.pop_back_val();
        MBELSI.computeLoopSafetyInfo(L);
        for (BasicBlock *BB : L->blocks())
          for (Instruction &I : *BB)
            if (MBELSI.isGuaranteedToExecute(I, &DT, L))
              MustExec[&I].push_back(L);
        Loops.append(L->begin(), L->end());
      }
    } else {

      for (auto &I : instructions(F)) {
        Loop *L = LI.getLoopFor(I.getParent());
        while (L) {
          if (isMustExecuteIn(I, L, &DT)) {
            MustExec[&I].push_back(L);
          }
          L = L->getParentLoop();
        };
      }
    }
  }

  void printInfoComment(const Value &V, formatted_raw_ostream &OS) override {
    if (!MustExec.count(&V)) {
      OS << " ; no mustexec loop";
      return;
    }

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

  MustBeExecutedContextExplorer Explorer(true, true, true, true, true, true,
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
  if (L.getHeader()->getParent()->hasFnAttribute(Attribute::WillReturn))
    return false;
  // TODO: Actually try to prove it is not.
  // TODO: If maybeEndlessLoop is going to be expensive, cache it.
  return true;
}

static bool mayContainIrreducibleControl(const Function &F, const LoopInfo *LI) {
  if (!LI)
    return false;
  using RPOTraversal = ReversePostOrderTraversal<const Function *>;
  RPOTraversal FuncRPOT(&F);
  return !containsIrreducibleCFG<const BasicBlock *, const RPOTraversal,
                                 const LoopInfo>(FuncRPOT, *LI);
}

static void getGuaranteedExecutedBlocks(
    const Instruction &TI,
    SmallVectorImpl<const BasicBlock *> &GuaranteedExecutedBlocks,
    const DominatorTree *DT, const LoopInfo *LI) {
  const BasicBlock *TIBlock = TI.getParent();
  bool NoThrow = TIBlock->getParent()->doesNotThrow();
  const Loop *L = LI ? LI->getLoopFor(TIBlock) : nullptr;

  if (const BasicBlock *SuccBB = getSuccessorInFirstIteration(TI, L, DT, LI))
    GuaranteedExecutedBlocks.push_back(SuccBB);

  // TODO: This might be better off in findForwardJoinPoint.
  // TODO: Check no-throw to this block in the loop not for the whole function
  if (DT && !maybeEndlessLoop(*L) && NoThrow && TI.getNumSuccessors() == 2 &&
      L->isLoopLatch(TIBlock)) {
    // TI is a latch of a finite loop. If it dominates all exiting blocks,
    // the non backedge has to be taken eventually.
    SmallVector<BasicBlock *, 4> ExitingBlocks;
    L->getExitingBlocks(ExitingBlocks);
    bool DominatesAllExitingBlocks =
        llvm::all_of(ExitingBlocks, [DT, TIBlock](BasicBlock *ExitingBB) {
          return DT->dominates(TIBlock, ExitingBB);
        });
    if (DominatesAllExitingBlocks) {
      if (TI.getSuccessor(1) == L->getHeader())
        GuaranteedExecutedBlocks.push_back(TI.getSuccessor(0));
      else
        GuaranteedExecutedBlocks.push_back(TI.getSuccessor(1));
    }
  }
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

/// Lookup \p Key in \p Map and return the result, potentially after
/// initializing the optional through \p Fn(\p args).
template <typename K, typename V, typename FnTy, typename... ArgsTy>
static V getOrCreateCachedOptional(K Key, DenseMap<K, Optional<V>> &Map,
                                   FnTy &&Fn, ArgsTy&&... args) {
  Optional<V> &OptVal = Map[Key];
  if (!OptVal.hasValue())
    OptVal = Fn(std::forward<ArgsTy>(args)...);
  return OptVal.getValue();
}

const BasicBlock *MustBeExecutedContextExplorer::findForwardJoinPoint(
    const BasicBlock *InitBB, const LoopInfo *LI,
    const PostDominatorTree *PDT) {
  LLVM_DEBUG(dbgs() << "\tFind forward join point for " << InitBB->getName()
                    << (LI ? " [LI]" : "") << (PDT ? " [PDT]" : ""));

  const Function &F = *InitBB->getParent();
  const Loop *L = LI ? LI->getLoopFor(InitBB) : nullptr;
  const BasicBlock *HeaderBB = L ? L->getHeader() : InitBB;
  bool WillReturnAndNoThrow = (F.hasFnAttribute(Attribute::WillReturn) ||
                               (L && !maybeEndlessLoop(*L))) &&
                              F.doesNotThrow();
  LLVM_DEBUG(dbgs() << (L ? " [in loop]" : "")
                    << (WillReturnAndNoThrow ? " [WillReturn] [NoUnwind]" : "")
                    << "\n");

  // Determine the adjacent blocks in the given direction but exclude (self)
  // loops under certain circumstances.
  SmallVector<const BasicBlock *, 8> Worklist;
  for (const BasicBlock *SuccBB : successors(InitBB)) {
    bool IsLatch = SuccBB == HeaderBB;
    // Loop latches are ignored in forward propagation if the loops cannot be
    // endless and may not throw: control has to go somewhere.
    if (!WillReturnAndNoThrow || !IsLatch)
      Worklist.push_back(SuccBB);
  }
  LLVM_DEBUG(dbgs() << "\t\t#Worklist: " << Worklist.size() << "\n");

  // If there are no other adjacent blocks, there is no join point.
  if (Worklist.empty())
    return nullptr;

  // If there is one adjacent block, it is the join point.
  if (Worklist.size() == 1)
    return Worklist[0];

  // Try to determine a join block through the help of the post-dominance
  // tree. If no tree was provided, we perform simple pattern matching for one
  // block conditionals only.
  const BasicBlock *JoinBB = nullptr;
  if (PDT)
    if (const auto *InitNode = PDT->getNode(InitBB))
      if (const auto *IDomNode = InitNode->getIDom())
        JoinBB = IDomNode->getBlock();

  if (!JoinBB && Worklist.size() == 2) {
    const BasicBlock *Succ0 = Worklist[0];
    const BasicBlock *Succ1 = Worklist[1];
    const BasicBlock *Succ0UniqueSucc = Succ0->getUniqueSuccessor();
    const BasicBlock *Succ1UniqueSucc = Succ1->getUniqueSuccessor();
    if (Succ0 == Succ1UniqueSucc) {
      // InitBB ->          Succ0 = JoinBB
      // InitBB -> Succ1 -> Succ0 = JoinBB
      JoinBB = Succ0;
    } else if (Succ1 == Succ0UniqueSucc) {
      // InitBB -> Succ0 -> Succ1 = JoinBB
      // InitBB ->          Succ1 = JoinBB
      JoinBB = Succ1;
    } else if (Succ0UniqueSucc == Succ1UniqueSucc) {
      // InitBB -> Succ0 -> JoinBB
      // InitBB -> Succ1 -> JoinBB
      JoinBB = Succ0UniqueSucc;
    }
  }

  if (!JoinBB && L)
    JoinBB = L->getUniqueExitBlock();

  if (!JoinBB)
    return nullptr;

  LLVM_DEBUG(dbgs() << "\t\tJoin block candidate: " << JoinBB->getName() << "\n");

  // In forward direction we check if control will for sure reach InitBB from
  // JoinBB, thus it can not be "stopped" along the way. Ways to "stop" control
  // are: infinite loops and instructions that do not necessarily transfer
  // execution to their successor. To check for them we traverse the CFG from
  // the adjacent blocks to the JoinBB, looking at all intermediate blocks.

  if (!F.hasFnAttribute(Attribute::WillReturn) || !F.doesNotThrow()) {

    auto BlockTransfersExecutionToSuccessor = [](const BasicBlock *BB) {
      return isGuaranteedToTransferExecutionToSuccessor(BB);
    };

    SmallPtrSet<const BasicBlock *, 16> Visited;
    while (!Worklist.empty()) {
      const BasicBlock *ToBB = Worklist.pop_back_val();
      if (ToBB == JoinBB)
        continue;

      // Make sure all loops in-between are finite.
      if (!Visited.insert(ToBB).second) {
        if (!F.hasFnAttribute(Attribute::WillReturn)) {
          bool MayContainIrreducibleControl = getOrCreateCachedOptional(
              &F, IrreducibleControlMap, mayContainIrreducibleControl, F, LI);
          if (MayContainIrreducibleControl)
            return nullptr;

          const Loop *L = LI->getLoopFor(ToBB);
          if (L && maybeEndlessLoop(*L))
            return nullptr;
        }

        continue;
      }

      // Make sure the block has no instructions that could stop control
      // transfer.
      bool TransfersExecution = getOrCreateCachedOptional(
          ToBB, BlockTransferMap, BlockTransfersExecutionToSuccessor, ToBB);
      if (!TransfersExecution)
        return nullptr;

      for (const BasicBlock *AdjacentBB : successors(ToBB))
        Worklist.push_back(AdjacentBB);
    }
  }

  LLVM_DEBUG(dbgs() << "\tJoin block: " << JoinBB->getName() << "\n");
  return JoinBB;
}

const BasicBlock *MustBeExecutedContextExplorer::findBackwardJoinPoint(
    const BasicBlock *InitBB, const LoopInfo *LI, const DominatorTree *DT) {
  LLVM_DEBUG(dbgs() << "\tFind backward join point for " << InitBB->getName()
                    << (LI ? " [LI]" : "") << (DT ? " [DT]" : ""));

  // Try to determine a join block through the help of the dominance tree. If no
  // tree was provided, we perform simple pattern matching for one block
  // conditionals only.
  if (DT) {
    const auto *InitNode = DT->getNode(InitBB);
    assert(InitNode && "Expected dominator tree node!");
    const auto *IDomNode = InitNode->getIDom();
    assert(IDomNode && "Expected dominator tree node to have a dominator node!");
    assert(IDomNode->getBlock() && "Expected dominator tree node to have a block!");
    return IDomNode->getBlock();
  }

  const Function &F = *InitBB->getParent();
  const Loop *L = LI ? LI->getLoopFor(InitBB) : nullptr;
  const BasicBlock *HeaderBB = L ? L->getHeader() : nullptr;

  // Determine the predecessor blocks but ignore backedges.
  SmallVector<const BasicBlock *, 8> Worklist;
  for (const BasicBlock *PredBB : predecessors(InitBB)) {
    bool IsBackedge = (PredBB == InitBB) ||
                      (HeaderBB == InitBB && L->contains(PredBB));
    // Loop backedges are ignored in backwards propagation: control has to come
    // from somewhere.
    if (!IsBackedge)
      Worklist.push_back(PredBB);
  }

  // If there are no other predecessor blocks, there is no join point.
  if (Worklist.empty())
    return nullptr;

  // If there is one predecessor block, it is the join point.
  if (Worklist.size() == 1)
    return Worklist[0];

  const BasicBlock *JoinBB = nullptr;
  if (Worklist.size() == 2) {
    const BasicBlock *Pred0 = Worklist[0];
    const BasicBlock *Pred1 = Worklist[1];
    const BasicBlock *Pred0UniquePred = Pred0->getUniquePredecessor();
    const BasicBlock *Pred1UniquePred = Pred1->getUniquePredecessor();
    if (Pred0 == Pred1UniquePred) {
      // InitBB <-          Pred0 = JoinBB
      // InitBB <- Pred1 <- Pred0 = JoinBB
      JoinBB = Pred0;
    } else if (Pred1 == Pred0UniquePred) {
      // InitBB <- Pred0 <- Pred1 = JoinBB
      // InitBB <-          Pred1 = JoinBB
      JoinBB = Pred1;
    } else if (Pred0UniquePred == Pred1UniquePred) {
      // InitBB <- Pred0 <- JoinBB
      // InitBB <- Pred1 <- JoinBB
      JoinBB = Pred0UniquePred;
    }
  }

  if (!JoinBB && L)
    JoinBB = L->getHeader();

  // In backwards direction there is no need to show termination of previous
  // instructions. If they do not terminate, the code afterward is dead, making
  // any information/transformation correct anyway.
  return JoinBB;
}

const Instruction *
MustBeExecutedContextExplorer::getMustBeExecutedNextInstruction(
    MustBeExecutedIterator &It, const Instruction *PP, bool PoppedCallStack) {
  if (!PP)
    return PP;
  LLVM_DEBUG(dbgs() << "Find next instruction for " << *PP
                    << (PoppedCallStack ? " [PoppedCallStack]" : "") << "\n");

  // If we explore only inside a given basic block we stop at terminators.
  if (!ExploreInterBlock && PP->isTerminator()) {
    LLVM_DEBUG(dbgs() << "\tReached terminator in intra-block mode, done\n");
    return nullptr;
  }

  // The function that contains the current position.
  const Function *PPFunc = PP->getFunction();

  // If we explore the call graph (CG) forward and we see a call site we can
  // continue with the callee instructions if the callee has an exact
  // definition. If this is the case, we add the call site in the forward call
  // stack such that we can return to the position after the call and also
  // translate values for the user.
  if (ExploreCGForward && !PoppedCallStack) {
    if (ImmutableCallSite ICS = ImmutableCallSite(PP))
      if (const Function *F = ICS.getCalledFunction())
        if (F->hasExactDefinition()) {
          It.ForwardCallStack.push_back({PP});
          return &F->getEntryBlock().front();
        }
  }

  // At a return instruction we have two options that allow us to continue:
  // 1) We are in a function that we entered earlier, in this case we can
  //    simply pop the last call site from the forward call stack and return the
  //    instruction after the call site. (this happens in the advance method!)
  // 2) We explore the call graph (CG) backwards trying to find a point that
  //    is know to be executed every time when the current function is.
  if (isa<ReturnInst>(PP)) {
    // The advance method will pop the stack.
    if (!It.ForwardCallStack.empty()) {
      LLVM_DEBUG(dbgs() << "\tReached return, will pop call stack\n");
      return nullptr;
    }

    // Check if backwards call graph exploration is allowed.
    if (!ExploreCGBackward) {
      LLVM_DEBUG(
          dbgs()
          << "\tReached return, no backward CG exploration allowed, done\n");
      return nullptr;
    }

    // We do not know all the callers for non-internal functions.
    if (!PPFunc->hasInternalLinkage()) {
      LLVM_DEBUG(dbgs() << "\tReached return, no unique call site (external "
                           "linkage), done\n");
      return nullptr;
    }

    // TODO: We restrict it to a single call site for now but we could allow
    //       more and find a join point interprocedurally.
    if (PPFunc->getNumUses() != 1) {
      LLVM_DEBUG(
          dbgs()
          << "\tReached return, no unique call site (multiple uses), done\n");
      return nullptr;
    }

    // Make sure the user is a direct call.
    // TODO: Indirect calls can be fine too.
    ImmutableCallSite ICS(PPFunc->user_back());
    if (!ICS || ICS.getCalledFunction() != PPFunc) {
      LLVM_DEBUG(
          dbgs()
          << "\tReached return, no unique call site (indirect call), done\n");
      return nullptr;
    }

    // We know we reached the return instruction of the callee so we will
    // continue at the next instruction after the call.
    LLVM_DEBUG(dbgs() << "\tReached return, continue after unique call site\n");
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
    const Instruction *NextPP =
        TransfersExecution ? PP->getNextNode() : nullptr;
    LLVM_DEBUG(dbgs() << "\tIntermediate instruction "
                      << (!TransfersExecution ? "does not" : "")
                      << "transfer control\n");
    return NextPP;
  }

  // Finally, we have to handle terminators, trivial ones first.
  assert(PP->isTerminator() && "Expected a terminator!");

  // A terminator without a successor which is not a return is not handled yet.
  // We can still try to continue at a known call site on the forward call stack
  // of the iterator.
  if (PP->getNumSuccessors() == 0) {
    LLVM_DEBUG(dbgs() << "\tUnhandled terminator\n");
    return nullptr;
  }

  // A terminator with a single successor, we will continue at the beginning of
  // that one.
  if (PP->getNumSuccessors() == 1) {
    LLVM_DEBUG(dbgs() << "\tUnconditional terminator, continue with successor\n");
    return &PP->getSuccessor(0)->front();
  }

  // Use flow-sensitive reasoning if allowed, e.g., to fold branches in the
  // first loop iteration.
  const LoopInfo *LI = LIGetter(*PPFunc);
  const DominatorTree *DT = DTGetter(*PPFunc);
  if (ExploreFlowSensitive) {
    SmallVector<const BasicBlock *, 2> GuaranteedExecutedBlocks;
    getGuaranteedExecutedBlocks(*PP, GuaranteedExecutedBlocks, DT, LI);
    LLVM_DEBUG(dbgs() << "\tFound " << GuaranteedExecutedBlocks.size()
                      << " blocks that are guaranteed executed as well\n");
    for (const BasicBlock *GEBB : GuaranteedExecutedBlocks) {
      LLVM_DEBUG(dbgs() << "\t\t- " << GEBB->getName() << "\n");
      It.DelayStack.insert(&GEBB->front());
    }
  }

  // Multiple successors mean we need to find the join point where control flow
  // converges again. We use the findForwardJoinPoint helper function with information
  // about the function and helper analyses, if available.
  const PostDominatorTree *PDT = PDTGetter(*PPFunc);
  if (const BasicBlock *JoinBB = findForwardJoinPoint(PP->getParent(), LI, PDT))
    return &JoinBB->front();

  LLVM_DEBUG(dbgs() << "\tNo join point found\n");
  return nullptr;
}

const Instruction *
MustBeExecutedContextExplorer::getMustBeExecutedPrevInstruction(
    MustBeExecutedIterator &It, const Instruction *PP, bool PoppedCallStack) {
  if (!PP)
    return PP;

  bool IsFirst = !(PP->getPrevNode());
  LLVM_DEBUG(dbgs() << "Find next instruction for " << *PP
                    << (PoppedCallStack ? " [PoppedCallStack]" : "")
                    << (IsFirst ? " [IsFirst]" : "") << "\n");

  // If we explore only inside a given basic block we stop at the first
  // instruction.
  if (!ExploreInterBlock && IsFirst) {
    LLVM_DEBUG(dbgs() << "\tReached block front in intra-block mode, done\n");
    return nullptr;
  }

  // The block and function that contains the current position.
  const BasicBlock *PPBlock = PP->getParent();
  const Function *PPFunc = PPBlock->getParent();

  // Ready the dominator tree if available.
  const DominatorTree *DT = DTGetter(*PPFunc);

  // If we explore the call graph (CG) forward and the current instruction
  // is a call site we can continue with the callee instructions if the callee
  // has an exact definition. If this is the case, we add the call site in the
  // backward call stack such that we can return to the position of the call and
  // also translate values for the user.
  if (ExploreCGForward && !PoppedCallStack) {
    if (ImmutableCallSite ICS = ImmutableCallSite(PP))
      if (const Function *F = ICS.getCalledFunction())
        if (F->hasExactDefinition())
          if (const Instruction *JoinPP =
                  getOrCreateCachedOptional(F, FunctionExitJoinPointMap,
                                            findFunctionExitJoinPoint, F, DT)) {
            It.BackwardCallStack.push_back({PP});
            return JoinPP;
          }
  }

  // At a first instruction in a function we have two options that allow us to
  // continue:
  // 1) We are in a function that we entered earlier, in this case we can
  //    simply pop the last call site from the backward call stack and
  //    return the instruction before the call site. (this happens in the
  //    advance method!)
  // 2) We explore the call graph (CG) backwards trying to find a point that
  //    is know to be executed every time when the current function is.
  if (IsFirst && PPBlock == &PPFunc->getEntryBlock()) {
    // The advance method will pop the stack.
    if (!It.BackwardCallStack.empty()) {
      LLVM_DEBUG(
          dbgs() << "\tReached function beginning, will pop call stack\n");
      return nullptr;
    }

    // Check if backwards call graph exploration is allowed.
    if (!ExploreCGBackward) {
      LLVM_DEBUG(dbgs() << "\tReached function beginning, no backward CG "
                           "exploration allowed, done\n");
      return nullptr;
    }

    // We do not know all the callers for non-internal functions.
    if (!PPFunc->hasInternalLinkage()) {
      LLVM_DEBUG(
          dbgs()
          << "\tReached function beginning, no unique call site (external "
             "linkage), done\n");
      return nullptr;
    }

    // TODO: We restrict it to a single call site for now but we could allow
    //       more and find a join point interprocedurally.
    if (PPFunc->getNumUses() != 1) {
      LLVM_DEBUG(dbgs() << "\tReached function beginning, no unique call site "
                           "(multiple uses), done\n");
      return nullptr;
    }

    // Make sure the user is a direct call.
    // TODO: Indirect calls can be fine too.
    ImmutableCallSite ICS(PPFunc->user_back());
    if (!ICS || ICS.getCalledFunction() != PPFunc) {
      LLVM_DEBUG(dbgs() << "\tReached function beginning, no unique call site "
                           "(indirect call), done\n");
      return nullptr;
    }

    // We know we reached the first instruction of the callee so we must have
    // executed the instruction before the call.
    LLVM_DEBUG(
        dbgs()
        << "\tReached function beginning, continue with unique call site\n");
    return ICS.getInstruction();
  }

  // If we are inside a block we know what instruction was executed before, the
  // previous one. However, if the previous one is a call site, we can enter the
  // callee and visit its instructions as well.
  if (!IsFirst) {
    const Instruction *PrevPP = PP->getPrevNode();
    LLVM_DEBUG(
        dbgs() << "\tIntermediate instruction, continue with previous\n");
    // We did not enter a callee so we simply return the previous instruction.
    return PrevPP;
  }

  // Finally, we have to handle the case where the program point is the first in
  // a block but not in the function. We use the findBackwardJoinPoint helper
  // function with information about the function and helper analyses, if
  // available.
  const LoopInfo *LI = LIGetter(*PPFunc);
  if (const BasicBlock *JoinBB = findBackwardJoinPoint(PPBlock, LI, DT))
    return &JoinBB->back();

  LLVM_DEBUG(dbgs() << "\tNo join point found\n");
  return nullptr;
}

MustBeExecutedIterator::MustBeExecutedIterator(
    MustBeExecutedContextExplorer &Explorer, const Instruction *I)
    : Visited(new VisitedSetTy()), Explorer(Explorer), CurInst(I),
      Head(nullptr), Tail(nullptr) {
  reset(I);
}

void MustBeExecutedIterator::reset(const Instruction *I) {
  Visited->clear();
  DelayStack.clear();
  ForwardCallStack.clear();
  BackwardCallStack.clear();
  resetInstruction(I);
}

void MustBeExecutedIterator::resetInstruction(const Instruction *I) {
  CurInst = I;
  Head = Tail = nullptr;
  Visited->insert({I, ExplorationDirection::FORWARD});
  Visited->insert({I, ExplorationDirection::BACKWARD});
  if (Explorer.ExploreCFGForward)
    Head = I;
  if (Explorer.ExploreCFGBackward)
    Tail = I;
}

const Instruction *MustBeExecutedIterator::advance(bool PoppedCallStack) {
  assert(CurInst && "Cannot advance an end iterator!");
  Head =
      Explorer.getMustBeExecutedNextInstruction(*this, Head, PoppedCallStack);
  if (Head && Visited->insert({Head, ExplorationDirection::FORWARD}).second)
    return Head;

  if (!ForwardCallStack.empty()) {
    Head = ForwardCallStack.pop_back_val();
    return advance(true);
  }
  Head = nullptr;

  Tail =
      Explorer.getMustBeExecutedPrevInstruction(*this, Tail, PoppedCallStack);
  if (Tail && Visited->insert({Tail, ExplorationDirection::BACKWARD}).second)
    return Tail;

  if (!BackwardCallStack.empty()) {
    Tail = BackwardCallStack.pop_back_val();
    return advance(true);
  }

  Tail = nullptr;

  if (!DelayStack.empty()) {
    const Instruction *DelayedI = DelayStack.pop_back_val();
    LLVM_DEBUG(dbgs() << "Poped new program point from delay stack: "
                      << *DelayedI << "\n");
    resetInstruction(DelayedI);
    assert(CurInst && "Expected valid instruction after reset!");
    return CurInst;
  }

  return nullptr;
}

template <bool TrackThrowingBBs, uint64_t MaxInstToExplore>
void MustBeExecutedLoopSafetyInfo<
    TrackThrowingBBs, MaxInstToExplore>::computeLoopSafetyInfo(const Loop *L) {
  assert(L && "Expected a loop!");
  LLVM_DEBUG(dbgs() << "Compute loop safety info for " << L->getName()
                    << "\n";);

  const Instruction &LoopFirstInst = L->getHeader()->front();
  It = &Explorer.begin(&LoopFirstInst);

  // Explore the context until we run out of the loop.
  // TODO: This might be a case where we want to guide the exploraiton.
  uint64_t InstExplorer = 0;
  while (const Instruction *I = **It) {
    // This assumes we explore the CFG "forward" first and the explorer is not
    // interprocedural.
    if (!L->contains(I))
      break;
    ++(*It);
    if (MaxInstToExplore && ++InstExplorer >= MaxInstToExplore)
      break;
  }
  LLVM_DEBUG(dbgs() << "done exploring the loop\n");

  if (TrackThrowingBBs) {
    // Fill the ThrowingBlocksMap with basic block -> may throw information.
    bool AnyMayThrow = false;
    for (const BasicBlock *BB : L->blocks()) {
      bool BBMayThrow = false;
      for (const Instruction &I : *BB)
        if ((BBMayThrow = I.mayThrow()))
          break;
      AnyMayThrow |= BBMayThrow;
      ThrowingBlocksMap[BB] = BBMayThrow;
    }

    // Use nullptr as key for "any" block.
    ThrowingBlocksMap[nullptr] = AnyMayThrow;
  }
}
