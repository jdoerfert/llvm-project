//===- MustExecute.h - Is an instruction known to execute--------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Contains a collection of routines for determining if a given instruction is
/// guaranteed to execute if a given point in control flow is reached. The most
/// common example is an instruction within a loop being provably executed if we
/// branch to the header of it's containing loop.
///
/// There are two interfaces available to determine if an instruction is
/// executed once a given point in the control flow is reached:
/// 1) A loop-centric one derived from LoopSafetyInfo.
/// 2) A "must be executed context"-based one implemented in the
///    MustBeExecutedContextExplorer.
/// Please refer to the class comments for more information.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MUSTEXECUTE_H
#define LLVM_ANALYSIS_MUSTEXECUTE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/InstructionPrecedenceTracking.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"

namespace llvm {

namespace {
template <typename T> using GetterTy = std::function<T *(const Function &F)>;
}

class Instruction;
class DominatorTree;
class Loop;

/// The common interface for the loop safety information.
struct LoopSafetyInfoInterface {

  /// Returns true iff the block \p BB potentially may throw exception. It can
  /// be false-positive in cases when we want to avoid complex analysis.
  virtual bool blockMayThrow(const BasicBlock *BB) const = 0;

  /// Returns true iff any block of the loop for which this info is contains an
  /// instruction that may throw or otherwise exit abnormally.
  virtual bool anyBlockMayThrow() const = 0;

  /// Computes safety information for a loop checks loop body & header for
  /// the possibility of may throw exception, it takes LoopSafetyInfo and loop
  /// as argument. Updates safety information in LoopSafetyInfo argument.
  /// Note: This is defined to clear and reinitialize an already initialized
  /// LoopSafetyInfo.  Some callers rely on this fact.
  virtual void computeLoopSafetyInfo(const Loop *CurLoop) = 0;

  /// Returns true if the instruction in a loop is guaranteed to execute at
  /// least once (under the assumption that the loop is entered).
  virtual bool isGuaranteedToExecute(const Instruction &Inst,
                                     const DominatorTree *DT,
                                     const Loop *CurLoop) const = 0;

  /// Return true if we must reach the block \p BB under assumption that the
  /// loop \p CurLoop is entered.
  virtual bool allLoopPathsLeadToBlock(const Loop *CurLoop,
                                       const BasicBlock *BB,
                                       const DominatorTree *DT) const = 0;

  /// Inform the safety info that we are planning to insert a new instruction
  /// \p Inst before \p PosI. It will make all cache updates to keep it correct
  /// after this insertion.
  virtual void insertInstructionBefore(const Instruction *NewI,
                                       const Instruction *PosI) {}

  /// Inform the safety info that we are planning to insert a new instruction
  /// \p Inst after \p PosI. It will make all cache updates to keep it correct
  /// after this insertion.
  virtual void insertInstructionAfter(const Instruction *NewI,
                                      const Instruction *PosI){};

  /// Inform safety info that we are planning to remove the instruction \p Inst
  /// from its block. It will make all cache updates to keep it correct after
  /// this removal.
  virtual void removeInstruction(const Instruction *Inst) {}
};

/// Captures loop safety information.
/// It keep information for loop blocks may throw exception or otherwise
/// exit abnormaly on any iteration of the loop which might actually execute
/// at runtime.  The primary way to consume this infromation is via
/// isGuaranteedToExecute below, but some callers bailout or fallback to
/// alternate reasoning if a loop contains any implicit control flow.
/// NOTE: LoopSafetyInfo contains cached information regarding loops and their
/// particular blocks. This information is only dropped on invocation of
/// computeLoopSafetyInfo. If the loop or any of its block is deleted, or if
/// any thrower instructions have been added or removed from them, or if the
/// control flow has changed, or in case of other meaningful modifications, the
/// LoopSafetyInfo needs to be recomputed. If a meaningful modifications to the
/// loop were made and the info wasn't recomputed properly, the behavior of all
/// methods except for computeLoopSafetyInfo is undefined.
class LoopSafetyInfo : public LoopSafetyInfoInterface {
  // Used to update funclet bundle operands.
  DenseMap<BasicBlock *, ColorVector> BlockColors;

protected:
  /// Computes block colors.
  void computeBlockColors(const Loop *CurLoop);

public:
  /// Returns block colors map that is used to update funclet operand bundles.
  const DenseMap<BasicBlock *, ColorVector> &getBlockColors() const;

  /// Copy colors of block \p Old into the block \p New.
  void copyColors(BasicBlock *New, BasicBlock *Old);

  /// Return true if we must reach the block \p BB under assumption that the
  /// loop \p CurLoop is entered.
  bool allLoopPathsLeadToBlock(const Loop *CurLoop, const BasicBlock *BB,
                               const DominatorTree *DT) const override;

  LoopSafetyInfo() = default;

  virtual ~LoopSafetyInfo() = default;
};


/// Simple and conservative implementation of LoopSafetyInfo that can give
/// false-positive answers to its queries in order to avoid complicated
/// analysis.
class SimpleLoopSafetyInfo: public LoopSafetyInfo {
  bool MayThrow = false;       // The current loop contains an instruction which
                               // may throw.
  bool HeaderMayThrow = false; // Same as previous, but specific to loop header

public:
  virtual bool blockMayThrow(const BasicBlock *BB) const;

  virtual bool anyBlockMayThrow() const;

  virtual void computeLoopSafetyInfo(const Loop *CurLoop);

  virtual bool isGuaranteedToExecute(const Instruction &Inst,
                                     const DominatorTree *DT,
                                     const Loop *CurLoop) const;

  SimpleLoopSafetyInfo() : LoopSafetyInfo() {};

  virtual ~SimpleLoopSafetyInfo() {};
};

/// This implementation of LoopSafetyInfo use ImplicitControlFlowTracking to
/// give precise answers on "may throw" queries. This implementation uses cache
/// that should be invalidated by calling the methods insertInstructionTo and
/// removeInstruction whenever we modify a basic block's contents by adding or
/// removing instructions.
class ICFLoopSafetyInfo: public LoopSafetyInfo {
  bool MayThrow = false;       // The current loop contains an instruction which
                               // may throw.
  // Contains information about implicit control flow in this loop's blocks.
  mutable ImplicitControlFlowTracking ICF;
  // Contains information about instruction that may possibly write memory.
  mutable MemoryWriteTracking MW;

public:
  virtual bool blockMayThrow(const BasicBlock *BB) const;

  virtual bool anyBlockMayThrow() const;

  virtual void computeLoopSafetyInfo(const Loop *CurLoop);

  virtual bool isGuaranteedToExecute(const Instruction &Inst,
                                     const DominatorTree *DT,
                                     const Loop *CurLoop) const;

  /// Returns true if we could not execute a memory-modifying instruction before
  /// we enter \p BB under assumption that \p CurLoop is entered.
  bool doesNotWriteMemoryBefore(const BasicBlock *BB, const Loop *CurLoop)
      const;

  /// Returns true if we could not execute a memory-modifying instruction before
  /// we execute \p I under assumption that \p CurLoop is entered.
  bool doesNotWriteMemoryBefore(const Instruction &I, const Loop *CurLoop)
      const;

  /// Inform the safety info that we are planning to insert a new instruction
  /// \p Inst into the basic block \p BB. It will make all cache updates to keep
  /// it correct after this insertion.
  void insertInstructionTo(const Instruction *Inst, const BasicBlock *BB);

  /// See LoopSafetyInfoInterface::insertInstructionBefore(...).
  void insertInstructionBefore(const Instruction *NewI,
                                       const Instruction *PosI) {
    insertInstructionTo(NewI, PosI->getParent());
  }

  /// See LoopSafetyInfoInterface::insertInstructionAfter(...).
  void insertInstructionAfter(const Instruction *NewI,
                                      const Instruction *PosI) {
    insertInstructionTo(NewI, PosI->getParent());
  };

  /// Inform safety info that we are planning to remove the instruction \p Inst
  /// from its block. It will make all cache updates to keep it correct after
  /// this removal.
  void removeInstruction(const Instruction *Inst);

  ICFLoopSafetyInfo(DominatorTree *DT) : LoopSafetyInfo(), ICF(DT), MW(DT) {};

  virtual ~ICFLoopSafetyInfo() {};
};

/// Forward declaration.
struct MustBeExecutedIterator;
struct MustBeExecutedContextExplorer;

/// Enum that allows us to spell out the direction.
enum class ExplorationDirection {
  BACKWARD = 0,
  FORWARD = 1,
};

/// Must be executed iterators visit stretches of instructions that are
/// guaranteed to be executed together, potentially with other instruction
/// executed in-between.
///
/// Given the following code, and assuming all statements are single
/// instructions which transfer execution to the successor (see
/// isGuaranteedToTransferExecutionToSuccessor), there are two possible
/// outcomes. If we start the iterator at A, B, or E, we will visit only A, B,
/// and E. If we start at C or D, we will visit all instructions A-E.
///
/// \code
///   A;
///   B;
///   if (...) {
///     C;
///     D;
///   }
///   E;
/// \endcode
///
///
/// Below is the example extneded with instructions F and G. Now we assume F
/// might not transfer execution to it's successor G. As a result we get the
/// following visit sets:
///
/// Start Instruction   | Visit Set
/// A                   | A, B,       E, F
///    B,               | A, B,       E, F
///       C             | A, B, C, D, E, F
///          D          | A, B, C, D, E, F
///             E       | A, B,       E, F
///                F    | A, B,       E, F
///                   G | A, B,       E, F, G
///
///
/// \code
///   A;
///   B;
///   if (...) {
///     C;
///     D;
///   }
///   E;
///   F;  // Might not transfer execution to its successor G.
///   G;
/// \endcode
///
///
/// A more complex example involving conditionals, loops, break, and continue
/// is shown below. We again assume all instructions will transmit control to
/// the successor and we assume we can prove the inner loop to be finite. We
/// omit non-trivial branch conditions as the exploration is oblivious to them.
/// Constant branches are assumed to be unconditional in the CFG. The resulting
/// visist sets are shown in the table below.
///
/// \code
///   A;
///   while (true) {
///     B;
///     if (...)
///       C;
///     if (...)
///       continue;
///     D;
///     if (...)
///       break;
///     do {
///       if (...)
///         continue;
///       E;
///     } while (...);
///     F;
///   }
///   G;
/// \endcode
///
/// Start Instruction   | Visit Set
/// A                   | A, B
///    B                | A, B
///       C             | A, B, C
///          D          | A, B,    D
///             E       | A, B,    D, E, F
///                F    | A, B,    D,    F
///                  G  | A, B,    D,       G
///
///
/// Note that the examples show optimal visist sets but not necessarily the ones
/// derived by the explorer depending on the available CFG analyses (see
/// MustBeExecutedContextExplorer). Also note that we, depending on the options,
/// the visit set can contain instructions from other functions.
struct MustBeExecutedIterator {
  /// Type declarations that make his class an input iterator.
  ///{
  typedef const Instruction *value_type;
  typedef std::ptrdiff_t difference_type;
  typedef const Instruction **pointer;
  typedef const Instruction *&reference;
  typedef std::input_iterator_tag iterator_category;
  ///}

  using ExplorerTy = MustBeExecutedContextExplorer;

  MustBeExecutedIterator(const MustBeExecutedIterator &Other)
      : Visited(Other.Visited), ForwardCallStack(Other.ForwardCallStack),
        BackwardCallStack(Other.BackwardCallStack),
        DelayStack(Other.DelayStack),
         Explorer(Other.Explorer),
        CurInst(Other.CurInst), Head(Other.Head), Tail(Other.Tail) {}

  MustBeExecutedIterator(MustBeExecutedIterator &&Other)
      : Visited(std::move(Other.Visited)),
        ForwardCallStack(std::move(Other.ForwardCallStack)),
        BackwardCallStack(std::move(Other.BackwardCallStack)),
        DelayStack(std::move(Other.DelayStack)),
        Explorer(Other.Explorer), CurInst(Other.CurInst), Head(Other.Head),
        Tail(Other.Tail) {}

  MustBeExecutedIterator &operator=(MustBeExecutedIterator &&Other) {
    if (this != &Other) {
      std::swap(Visited, Other.Visited);
      std::swap(ForwardCallStack, Other.ForwardCallStack);
      std::swap(BackwardCallStack, Other.BackwardCallStack);
      std::swap(DelayStack, Other.DelayStack);
      std::swap(CurInst, Other.CurInst);
      std::swap(Head, Other.Head);
      std::swap(Tail, Other.Tail);
    }
    return *this;
  }

  ~MustBeExecutedIterator() {
  }

  /// Pre- and post-increment operators.
  ///{
  MustBeExecutedIterator &operator++() {
    CurInst = advance();
    return *this;
  }

  MustBeExecutedIterator operator++(int) {
    MustBeExecutedIterator tmp(*this);
    operator++();
    return tmp;
  }
  ///}

  /// Equality and inequality operators. Note that we ignore the history here.
  ///{
  bool operator==(const MustBeExecutedIterator &Other) const {
    return CurInst == Other.CurInst && Head == Other.Head && Tail == Other.Tail;
  }

  bool operator!=(const MustBeExecutedIterator &Other) const {
    return !(*this == Other);
  }
  ///}

  /// Return the underlying instruction.
  const Instruction *&operator*() { return CurInst; }
  const Instruction *getCurrentInst() const { return CurInst; }

  /// Return true if \p I was encountered by this iterator already.
  bool count(const Instruction *I) const {
    return Visited->count({I, ExplorationDirection::FORWARD}) ||
           Visited->count({I, ExplorationDirection::BACKWARD});
  }

  /// Call stack value map interface
  ///{
  // TODO: allow the user to map values encountered in other functions back to
  //       the function the iterator was exploring initially.
  ///}

  /// Configurable print method for debugging purposes.
  ///
  /// \param OS            The stream we print to.
  /// \param PrintInst     If set, the current instruction is printed.
  /// \param PrintInterval If set, the interval containing the current
  ///                      instruction is printed.
  void print(raw_ostream &OS, bool PrintInst, bool PrintInterval) const {
    if (PrintInst)
      OS << "[" << *CurInst << "]";

    if (PrintInterval)
      OS << "[H: " << *Head << "|T: " << *Tail << "]";
  }

private:
  using VisitedSetTy =
      DenseSet<PointerIntPair<const Instruction *, 1, ExplorationDirection>>;

  /// Private constructors.
  MustBeExecutedIterator(ExplorerTy &Explorer, const Instruction *I);

  MustBeExecutedIterator(MustBeExecutedContextExplorer &Explorer,
                         const Instruction *I,
                         std::shared_ptr<VisitedSetTy> Visited);

  /// Reset the iterator to its initial state pointing at \p I.
  void reset(const Instruction *I);

  /// Reset the iterator to point at \p I, keep cached state.
  void resetInstruction(const Instruction *I);

  /// Try to advance one of the underlying positions (Head or Tail).
  ///
  /// \param PoppedCallStack Flag to indicate if we just popped an instruction
  ///                        from the call stack such that we do not enter the
  ///                        same call site again.
  ///
  /// \return The next instruction in the must be executed context, or nullptr
  ///         if none was found.
  const Instruction *advance(bool PoppedCallStack = false);

  /// A set to track the visited instructions in order to deal with endless
  /// loops and recursion.
  std::shared_ptr<VisitedSetTy> Visited;

  /// Call stack used in forward exploration of the call graph.
  SmallVector<const Instruction *, 0> ForwardCallStack;

  /// Call stack used in backward exploration of the call graph.
  SmallVector<const Instruction *, 0> BackwardCallStack;

  /// Stack used in forward exploration when multiple successors are known to be
  /// executed.
  SetVector<const Instruction *> DelayStack;

  /// A reference to the explorer that created this iterator.
  ExplorerTy &Explorer;

  /// The instruction we are currently exposing to the user. There is always an
  /// instruction that we know is executed with the given program point,
  /// initially the program point itself.
  const Instruction *CurInst;

  /// Two positions that mark the program points where this iterator will look
  /// for the next instruction. Note that the current instruction is either the
  /// one pointed to by Head, Tail, or both.
  const Instruction *Head, *Tail;

  friend struct MustBeExecutedContextExplorer;
};

/// A "must be executed context" for a given program point PP is the set of
/// instructions, before and after PP, that are executed always when PP is
/// reached. The MustBeExecutedContextExplorer is a lazy and caching interface
/// to explore "must be executed contexts" in a module.
///
/// The explorer exposes "must be executed iterators" that traverse the must be
/// executed context. There is little information sharing between iterators as
/// the expected use case involves few iterators for "far apart" instructions.
/// If that changes, we should consider caching more intermediate results.
struct MustBeExecutedContextExplorer {

  /// Create a CFG only explorer with the given analyses as support.
  ///
  /// \param ExploreInterBlock    Flag to indicate if instructions in blocks
  ///                             other than the parent of PP should be
  ///                             explored.
  /// \param ExploreCFGForward    Flag to indicate if instructions located after
  ///                             PP in the CFG, e.g., post-dominating PP,
  ///                             should be explored.
  /// \param ExploreCFGBackward   Flag to indicate if instructions located
  ///                             before PP in the CFG, e.g., dominating PP,
  ///                             should be explored.
  /// \param ExploreFlowSensitive Flag to indicate if flow-sensitive reasoning
  ///                             should be performed during exploration.
  MustBeExecutedContextExplorer(bool ExploreInterBlock, bool ExploreCFGForward,
                                bool ExploreCFGBackward,
                                bool ExploreFlowSensitive,
                                const DominatorTree *DT = nullptr,
                                const PostDominatorTree *PDT = nullptr,
                                const LoopInfo *LI = nullptr)
      : MustBeExecutedContextExplorer(
            ExploreInterBlock, ExploreCFGForward, ExploreCFGBackward, false,
            false, ExploreFlowSensitive, [=](const Function &) { return LI; },
            [=](const Function &) { return DT; },
            [=](const Function &) { return PDT; }) {}

  /// In the description of the parameters we use PP to denote a program point
  /// for which the must be executed context is explored.
  ///
  /// \param ExploreInterBlock    Flag to indicate if instructions in blocks
  ///                             other than the parent of PP should be
  ///                             explored.
  /// \param ExploreCFGForward    Flag to indicate if instructions located after
  ///                             PP in the CFG, e.g., post-dominating PP,
  ///                             should be explored.
  /// \param ExploreCFGBackward   Flag to indicate if instructions located
  ///                             before PP in the CFG, e.g., dominating PP,
  ///                             should be explored.
  /// \param ExploreCGForward     Flag to indicate if instructions located in a
  ///                             callee that is reached from PP should be
  ///                             explored. Hence if the call graph (CG) is
  ///                             traverse in forward direction.
  /// \param ExploreCGBackward    Flag to indicate if instructions located in a
  ///                             caller that leads to PP should be explored.
  ///                             Hence if the call graph (CG) is traverse in
  ///                             backwards direction.
  /// \param ExploreFlowSensitive Flag to indicate if flow-sensitive reasoning
  ///                             should be performed during exploration.
  /// \param LIGetter  Return the LoopInfo analysis for a given function or a
  ///                  nullptr if it is not available.
  /// \param DTGetter  Return the DominatorTree analysis for a given function or
  ///                  a nullptr if it is not available.
  /// \param PDTGetter Return the PostDominatorTree analysis for a given
  ///                  function or a nullptr if it is not available.
  MustBeExecutedContextExplorer(
      bool ExploreInterBlock, bool ExploreCFGForward, bool ExploreCFGBackward,
      bool ExploreCGForward, bool ExploreCGBackward, bool ExploreFlowSensitive,
      GetterTy<const LoopInfo> LIGetter =
          [](const Function &) { return nullptr; },
      GetterTy<const DominatorTree> DTGetter =
          [](const Function &) { return nullptr; },
      GetterTy<const PostDominatorTree> PDTGetter =
          [](const Function &) { return nullptr; })
      : ExploreInterBlock(ExploreInterBlock),
        ExploreCFGForward(ExploreCFGForward),
        ExploreCFGBackward(ExploreCFGBackward),
        ExploreCGForward(ExploreCGForward),
        ExploreCGBackward(ExploreCGBackward),
        ExploreFlowSensitive(ExploreFlowSensitive), LIGetter(LIGetter),
        DTGetter(DTGetter), PDTGetter(PDTGetter), EndIterator(*this, nullptr) {
    assert((ExploreInterBlock || (!ExploreCGForward && !ExploreCGBackward)) &&
           "Cannot explore the call graph if inter-block exploration is not "
           "allowed.");
  }

  /// Clean up the dynamically allocated iterators.
  ~MustBeExecutedContextExplorer() {
    DeleteContainerPointers(Iterators);
  }

  /// Iterator-based interface. \see MustBeExecutedIterator.
  ///{
  using iterator = MustBeExecutedIterator;
  using const_iterator = const MustBeExecutedIterator;

  /// Return an iterator to explore the context around \p PP.
  iterator& begin(const Instruction *PP) {
    auto *&It = InstructionIteratorMap[PP];
    if (!It) {
      It = new iterator(*this, PP);
      Iterators.push_back(It);
    }
    return *It;
  }

  /// Return an iterator to explore the cached context around \p PP.
  const_iterator& begin(const Instruction *PP) const {
    return *InstructionIteratorMap.lookup(PP);
  }

  /// Return an universal end iterator.
  ///{
  iterator& end() { return EndIterator; }
  iterator& end(const Instruction *) { return EndIterator; }

  const_iterator& end() const { return EndIterator; }
  const_iterator& end(const Instruction *) const { return EndIterator; }
  ///}

  /// Return an iterator range to explore the context around \p PP.
  llvm::iterator_range<iterator> range(const Instruction *PP) {
    return llvm::make_range(begin(PP), end(PP));
  }

  /// Return an iterator range to explore the cached context around \p PP.
  llvm::iterator_range<const_iterator> range(const Instruction *PP) const {
    return llvm::make_range(begin(PP), end(PP));
  }
  ///}

  /// Update interface
  ///{

  /// Insert the instruction \p NewI before the instruction \p PosI in the
  /// cached information.
  void insertInstructionBefore(const Instruction *NewI,
                               const Instruction *PosI) {
    bool NewTransfersEx = isGuaranteedToTransferExecutionToSuccessor(NewI);
    const Instruction *PosIPrev = PosI->getPrevNode();
    for (auto &MapIt : InstructionIteratorMap) {
      iterator &It = *MapIt.getSecond();
      if (It.Visited->count({PosIPrev, ExplorationDirection::FORWARD}) &&
          It.Visited->count({PosI, ExplorationDirection::FORWARD})) {
        if (!NewTransfersEx) {
          It.reset(MapIt.getFirst());
          continue;
        }
        It.Visited->insert({NewI, ExplorationDirection::FORWARD});
      }
      if (It.Tail != PosI &&
          It.Visited->count({PosI, ExplorationDirection::BACKWARD})) {
        It.Visited->insert({NewI, ExplorationDirection::BACKWARD});
        if (!It.Tail)
          It.Tail = NewI;
      }
    }
  }

  /// Insert the instruction \p NewI after the instruction \p PosI in the
  /// cached information. This is not allowed for the last instruction in a
  /// basic block.
  void insertInstructionAfter(const Instruction *NewI,
                              const Instruction *PosI) {
    bool NewTransfersEx = isGuaranteedToTransferExecutionToSuccessor(NewI);
    bool PosTransfersEx = isGuaranteedToTransferExecutionToSuccessor(PosI);
    const Instruction *PosINext = PosI->getNextNode();
    for (auto &MapIt : InstructionIteratorMap) {
      iterator &It = *MapIt.getSecond();
      if (PosTransfersEx) {
        if (It.Head != PosI &&
            It.Visited->count({PosI, ExplorationDirection::FORWARD})) {
          if (!NewTransfersEx) {
            It.reset(MapIt.getFirst());
            continue;
          }
          It.Visited->insert({NewI, ExplorationDirection::FORWARD});
        }
      }
      if (It.Visited->count({PosI, ExplorationDirection::BACKWARD})) {
        if (!PosINext) {
          It.reset(MapIt.getFirst());
          continue;
        }
        if (It.Visited->count({PosINext, ExplorationDirection::BACKWARD}))
          It.Visited->insert({NewI, ExplorationDirection::BACKWARD});
      }
    }
  }

  /// Remove instruction \p I from the cached information.
  void removeInstruction(const Instruction *I) {
    InstructionIteratorMap.erase(I);
    BlockTransferMap.erase(I->getParent());
    FunctionExitJoinPointMap.erase(I->getFunction());
    for (auto &It : InstructionIteratorMap) {
      It.getSecond()->Visited->erase({I, ExplorationDirection::FORWARD});
      It.getSecond()->Visited->erase({I, ExplorationDirection::BACKWARD});
    }
  }
  ///}

  /// Return the next instruction that is guaranteed to be executed after \p PP.
  ///
  /// \param It              The iterator that is used to traverse the must be
  ///                        executed context.
  /// \param PP              The program point for which the next instruction
  ///                        that is guaranteed to execute is determined.
  /// \param PoppedCallStack Flag to indicate if we just popped an instruction
  ///                        from the call stack such that we do not enter the
  ///                        same call site again.
  const Instruction *
  getMustBeExecutedNextInstruction(MustBeExecutedIterator &It,
                                   const Instruction *PP, bool PoppedCallStack);

  /// Return the previous instr. that is guaranteed to be executed before \p PP.
  ///
  /// \param It              The iterator that is used to traverse the must be
  ///                        executed context.
  /// \param PP              The program point for which the previous instr.
  ///                        that is guaranteed to execute is determined.
  /// \param PoppedCallStack Flag to indicate if we just popped an instruction
  ///                        from the call stack such that we do not enter the
  ///                        same call site again.
  const Instruction *
  getMustBeExecutedPrevInstruction(MustBeExecutedIterator &It,
                                   const Instruction *PP, bool PoppedCallStack);

  /// Find the next join point from \p InitBB in forward direction.
  const BasicBlock *findForwardJoinPoint(const BasicBlock *InitBB,
                                         const LoopInfo *LI,
                                         const PostDominatorTree *PDT);

  /// Find the next join point from \p InitBB in backward direction.
  const BasicBlock *findBackwardJoinPoint(const BasicBlock *InitBB,
                                          const LoopInfo *LI,
                                          const DominatorTree *DT);

  /// Parameter that limit the performed exploration. See the constructor for
  /// their meaning.
  ///{
  const bool ExploreInterBlock;
  const bool ExploreCFGForward;
  const bool ExploreCFGBackward;
  const bool ExploreCGForward;
  const bool ExploreCGBackward;
  const bool ExploreFlowSensitive;
  ///}

private:
  /// Getters for common CFG analyses: LoopInfo, DominatorTree, and
  /// PostDominatorTree.
  ///{
  GetterTy<const LoopInfo> LIGetter;
  GetterTy<const DominatorTree> DTGetter;
  GetterTy<const PostDominatorTree> PDTGetter;
  ///}

  /// Map to cache isGuaranteedToTransferExecutionToSuccessor results.
  DenseMap<const BasicBlock *, Optional<bool>> BlockTransferMap;

  /// Map to cache containsIrreducibleCFG results.
  DenseMap<const Function*, Optional<bool>> IrreducibleControlMap;

  /// Map to cache function exit join points.
  DenseMap<const Function *, Optional<const Instruction *>>
      FunctionExitJoinPointMap;

  /// Map from instructions to associated must be executed iterators.
  DenseMap<const Instruction *, MustBeExecutedIterator *>
      InstructionIteratorMap;

  /// Collection of all iterators in flight.
  SmallVector<MustBeExecutedIterator *, 16> Iterators;

  /// A unique end iterator.
  MustBeExecutedIterator EndIterator;
};

/// An implementation of the LoopSafetyInfoInterface using the
/// MustBeExecutedContextExplorer to determine all reached instructions.
template<bool TrackThrowingBBs = true, uint64_t MaxInstToExplore = 0>
struct MustBeExecutedLoopSafetyInfo final : public LoopSafetyInfoInterface {

  MustBeExecutedLoopSafetyInfo(const DominatorTree *DT = nullptr,
                               const PostDominatorTree *PDT = nullptr,
                               const LoopInfo *LI = nullptr)
      : Explorer(true, true, false, true, DT, PDT, LI), It(nullptr) {}

  /// See LoopSafetyInfo::bockMayThrow(...).
  bool blockMayThrow(const BasicBlock *BB) const override {
    assert(TrackThrowingBBs && "Object was created without throw tracking.");
    return ThrowingBlocksMap.lookup(BB);
  }

  /// See LoopSafetyInfo::bockMayThrow(...).
  bool anyBlockMayThrow() const override {
    assert(TrackThrowingBBs && "Object was created without throw tracking.");
    return ThrowingBlocksMap.lookup(nullptr);
  };

  /// See LoopSafetyInfo::computeLoopSafetyInfo(...).
  void computeLoopSafetyInfo(const Loop *CurLoop) override;

  /// See LoopSafetyInfo::isGuaranteedToExecute(...).
  bool isGuaranteedToExecute(const Instruction &Inst, const DominatorTree *,
                             const Loop *CurLoop) const override {
    return It && It->count(&Inst);
  }

  /// See LoopSafetyInfo::allLoopPathsLeadToBlock(...).
  bool allLoopPathsLeadToBlock(const Loop *, const BasicBlock *BB,
                               const DominatorTree *) const override {
    assert(BB && "Expected a loop and a block!");
    // We want to reach the first instruction in the block.
    const Instruction &BBFirstInst = BB->front();
    return It && It->count(&BBFirstInst);
  }

  /// See LoopSafetyInfo::insertInstructionBefore(...).
  void insertInstructionBefore(const Instruction *NewI,
                               const Instruction *PosI) override {
    Explorer.insertInstructionBefore(NewI, PosI);
  }

  /// See LoopSafetyInfo::insertInstructionAfter(...).
  void insertInstructionAfter(const Instruction *NewI,
                              const Instruction *PosI) override {
    Explorer.insertInstructionAfter(NewI, PosI);
  }

  /// See LoopSafetyInfo::removeInstruction(...).
  void removeInstruction(const Instruction *Inst) override {
    Explorer.removeInstruction(Inst);
  }

private:
  MustBeExecutedContextExplorer Explorer;
  MustBeExecutedContextExplorer::iterator *It;
  DenseMap<const BasicBlock *, bool> ThrowingBlocksMap;
};

} // namespace llvm

#endif
