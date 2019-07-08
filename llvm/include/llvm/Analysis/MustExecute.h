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
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/InstructionPrecedenceTracking.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"

STATISTIC(NumMustBeExecutedIteratorAdvancesCached,
          "Number of must-be-executed iterator advances already cached");
STATISTIC(NumMustBeExecutedIteratorAdvancesExplored,
          "Number of must-be-executed iterator advances explored");

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
struct MustBeExecutedContextExplorer;

/// Enum that allows us to spell out the direction.
enum ExplorationDirection {
  ED_BACKWARD = 0,
  ED_FORWARD = 1,
};

/// Must be executed intervals are stretches of instructions that are guaranteed
/// to be executed together without any other instruction executed in-between.
/// In addition, intervals are linked. Each knows the interval always executed
/// before and the on always executed afterward, if they exit.
///
/// Given the following code, and assuming all statements are single
/// instructions which transfer execution to the successor (see
/// isGuaranteedToTransferExecutionToSuccessor), there are three intervals as
/// shown in the table below.
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
/// No  | Instructions | Previous | Next
/// (1) |        A ; B |     None |  (3)
/// (2) |        C ; D |      (1) |  (3)
/// (3) |            E |      (1) | None
///
/// From the above table one can derive that C and D are always executed with
/// A, B, and E. However, A, B, and E, are not necessarily executed with C and
/// D but always together.
///
///
/// Below is an extended example with instructions F and G and we assume F might
/// not transfer execution to it's successor G. The resulting intervals are
/// shown in the table below.
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
/// No  | Instructions | Previous | Next
/// (1) |        A ; B |     None |  (3)
/// (2) |        C ; D |      (1) |  (3)
/// (3) |        E ; F |      (1) | None
/// (4) |            G |      (3) | None
///
/// As indicated by the table, E and F are executed always together with A and B
/// but not C, D, or G. However, G is always executed with A, B, E, and F.
///
///
/// A more complex example involving conditionals, loops, break, and continue
/// is shown below. We again assume all instructions will transmit control to
/// the successor and we assume we can prove the loops to be finite. We omit
/// non-trivial branch conditions as the exploration is oblivious to them.
/// Constant branches are assumed to be unconditional in the CFG. The resulting
/// intervals are shown in the table below.
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
///     } while (...)
///     F;
///   }
///   G;
/// \endcode
///
/// No  | Instructions | Previous | Next
/// (1) |            A |     None |  (7)
/// (2) |            B |      (1) |  (4)
/// (3) |            C |      (2) |  (4)
/// (4) |            D |      (2) |  (7)
/// (5) |            E |      (4) |  (6)
/// (6) |            F |      (4) |  (2)
/// (7) |            G |      (1) | None
///
///
/// Note that the examples show optimal intervals but not necessarily the ones
/// derived by the explorer (see MustBeExecutedContextExplorer). Also note that
/// intervals will only contain instructions from a single function but they can
/// be linked to intervals containing instructions from other functions, e.g.,
/// if you would outline instructions C and D in the first example such that
/// there is a call to a function containing them instead.
///
///
/// Implementation:
/// Intervals are centered around an instruction at position 0. From there, they
/// grow in both directions using the NextInsts and PrevInst vector,
/// respectively. The links to the next and previous interval are stored in the
/// Next and Prev pointers. Note that we avoid exposing any of the underlying
/// structures because the order in which instructions are queried can change
/// the actual layout of intervals.
///
/// Positions:
/// In order to explore an intervals one needs to have a position which can be
/// incremented/decremented. Positions combine a current interval with an
/// offset. That is, they represent the instruction offset "steps" before/after
/// the center of the interval which is at offset 0. Offset can therefore be
/// negative. If a position runs out of instructions in the interval it will
/// try to move to the next interval in the requested direction, if one exists.
/// In case there was no interval, the position becomes invalid.
struct MustBeExecutedInterval {

  /// Constructor for a new interval centered around \p I.
  MustBeExecutedInterval(const Instruction *I) : Prev(nullptr), Next(nullptr) {
    assert(I && "Cannot construct an interval for a null pointer!");
    // Put the instruction I in both vectors to make the offset and in-bounds
    // calculation easier.
    NextInsts.push_back(I);
    PrevInsts.push_back(I);
  }

  /// Return true if \p Offset is part of this interval, false otherwise.
  ///
  /// This function deals with both positive and negative offsets as measured
  /// from the center instruction at offset 0.
  bool isInbounds(int Offset) {
    return Offset >= 0 ? (int)NextInsts.size() > Offset
                       : (int)PrevInsts.size() > -Offset;
  }

  /// Subscript operator to allow easy access to the instructions based on their
  /// offset.
  const Instruction *&operator[](int Offset) {
    assert(isInbounds(Offset) && "Trying to access an interval out-of-bounds!");
    return Offset >= 0 ? NextInsts[Offset] : PrevInsts[-Offset];
  }

  /// Print method for debugging purposes.
  void print(raw_ostream &OS, bool Detailed = false) const {
    OS << "[" << Prev << ":" << this << ":" << Next << "]";
    if (Detailed) {
      OS << "\n";
      for (auto *I : PrevInsts)
        OS << "- " << *I << "\n";
      for (auto *I : NextInsts)
        OS << "- " << *I << "\n";
    }
  }

  /// Return the number of instructions contained in this interval.
  size_t size() const {
    return NextInsts.size() + PrevInsts.size() /* the center is duplicated */ - 1;
  }

  /// A position on the interval chain.
  ///
  /// See MustBeExecutedInterval for more information.
  struct Position {

    Position(MustBeExecutedInterval *Interval = nullptr)
        : Interval(Interval), Offset(0) {}

    /// Return the instruction at this position.
    const Instruction *getInstruction() const {
      assert(bool(*this) &&
             "Invalid positions do not have an associated instruction!");
      return (*Interval)[Offset];
    }

    /// Allow valid positions to evaluate to 'true' and invalid ones to 'false'
    /// when a position is converted to a boolean.
    operator bool() const { return Interval; }

    /// Equality operator.
    bool operator==(const Position &Other) const {
      return Interval == Other.Interval && Offset == Other.Offset;
    }

    /// Advance the position in the direction \p Direction.
    ///
    /// If there is no next instruction in the direction, this will result in
    /// an invalid position, hence the interval will be a nullptr.
    ///{
    friend Position operator+(Position Pos, ExplorationDirection Direction) {
      return Pos += Direction;
    }

    Position &operator+=(ExplorationDirection Direction) {
      assert(Interval && "Cannot advance in invalid position!");
      int DirectionOffset = Direction == ED_FORWARD ? 1 : -1;

      // If the offset adjusted wrt. direction is still in the associated
      // interval, we adjust the offset and are done. Otherwise, we try to go to
      // the linked interval in the direction and set the offset accordingly,
      // hence to the very beginning or end depending on the direction. If there
      // is no linked interval, all members will be set to NULL and the position
      // becomes invalid.
      if (Interval->isInbounds(Offset + DirectionOffset)) {
        Offset += DirectionOffset;
      } else if ((Interval = (Direction == ED_FORWARD ? Interval->Next
                                                      : Interval->Prev))) {
        Offset = Direction == ED_FORWARD ? 1 - Interval->PrevInsts.size()
                                         : Interval->NextInsts.size() - 1;
      } else {
        Offset = 0;
      }
      return *this;
    }
    ///}

    /// Print method for debugging purposes.
    void print(raw_ostream &OS, bool Detailed = false) const {
      OS << "{";
      if (Interval)
        Interval->print(OS, Detailed);
      else
        OS << "[NONE:NONE:NONE]";
      OS << "@" << Offset << "}";
    }

  private:
    /// The interval this position is currently associated with.
    MustBeExecutedInterval *Interval;

    /// The current offset from the center of the associated interval.
    int Offset;

    friend class MustBeExecutedContextExplorer;
  };

private:
  /// The vectors that represent the instructions in the interval.
  ///{
  using InstructionVector = SmallVector<const Instruction *, 8>;
  InstructionVector PrevInsts, NextInsts;
  ///}

  /// The links to the previous and next interval.
  struct MustBeExecutedInterval *Prev, *Next;

  /// Append the interval by \p I in the front or back, depending on \p Front.
  void append(const Instruction *I, bool Front) {
    if (Front) {
      NextInsts.push_back(I);
      assert(!Next &&
             "Cannot advance the front if a next interval is already set!");
    } else {
      PrevInsts.push_back(I);
      assert(!Prev &&
             "Cannot advance the back if a previous interval is already set!");
    }
  }

  friend struct MustBeExecutedContextExplorer;
};

/// An input iterator for must be executed intervals. Depending on the template
/// parameter \p CachedOnly, the iterator will either only visit existing
/// instructions in the interval (CachedOnly == true) or also use the
/// MustBeExecutedContextExplorer to extend the intervals if needed.
template <bool CachedOnly> struct MustBeExecutedIterator {
  /// Type declarations that make his class an input iterator.
  ///{
  typedef const Instruction *value_type;
  typedef std::ptrdiff_t difference_type;
  typedef const Instruction **pointer;
  typedef const Instruction *&reference;
  typedef std::input_iterator_tag iterator_category;
  ///}

  /// Type definition to choose the const version of the
  /// MustBeExecutedContextExplorer in cached-only mode.
  using ExplorerTy =
      typename std::conditional<CachedOnly, const MustBeExecutedContextExplorer,
                                MustBeExecutedContextExplorer>::type;

  /// Constructor replacements to create a begin and end iterator respectively.
  ///{
  static MustBeExecutedIterator
  begin(ExplorerTy &Explorer, const MustBeExecutedInterval::Position &Pos) {
    if (!Pos)
      return end(Explorer);
    return MustBeExecutedIterator(Explorer, Pos);
  }

  static MustBeExecutedIterator end(ExplorerTy &Explorer) {
    return MustBeExecutedIterator(Explorer);
  }
  ///}

  MustBeExecutedIterator(const MustBeExecutedIterator &Other)
      : Visited(Other.Visited), ForwardCallStack(Other.ForwardCallStack),
        BackwardCallStack(Other.BackwardCallStack), Explorer(Other.Explorer),
        CurInst(Other.CurInst), Head(Other.Head), Tail(Other.Tail) {}

  MustBeExecutedIterator(MustBeExecutedIterator &&Other)
      : Visited(std::move(Other.Visited)),
        ForwardCallStack(std::move(Other.ForwardCallStack)),
        BackwardCallStack(std::move(Other.BackwardCallStack)),
        Explorer(Other.Explorer), CurInst(Other.CurInst), Head(Other.Head),
        Tail(Other.Tail) {}

  MustBeExecutedIterator &operator=(MustBeExecutedIterator &&Other) {
    if (this != &Other) {
      std::swap(Visited, Other.Visited);
      std::swap(ForwardCallStack, Other.ForwardCallStack);
      std::swap(BackwardCallStack, Other.BackwardCallStack);
      std::swap(Explorer, Other.Explorer);
      std::swap(CurInst, Other.CurInst);
      std::swap(Head, Other.Head);
      std::swap(Tail, Other.Tail);
    }
    return *this;
  }

  /// Pre- and post-increment operators. Both can cause the underlying intervals
  /// to be extended if this is not a cached-only iterator.
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

  /// Return true if \p I was encountered by this iterator already.
  bool count(const Instruction *I) const {
    return Visited.count({I, ED_FORWARD}) || Visited.count({I, ED_BACKWARD});
  }

  /// Configurable print method for debugging purposes.
  ///
  /// \param OS            The stream we print to.
  /// \param PrintInst     If set, the current instruction is printed.
  /// \param PrintInterval If set, the interval containing the current
  ///                      instruction is printed.
  void print(raw_ostream &OS, bool PrintInst, bool PrintInterval) const {
    if (PrintInst)
      OS << "[" << *CurInst << "]";

    if (!PrintInterval)
      return;

    if (Head.getInstruction() == CurInst) {
      Head.print(OS, false);
    } else {
      assert(Tail.getInstruction() == CurInst &&
             "Neither head nor tail pointed at the current instruction!");
      Tail.print(OS, false);
    }
  }

private:
  /// Private constructors to disallow direct construction. Use
  /// MustBeExecutedIterator::begin and MustBeExecutedIterator::end instead.
  ///{
  MustBeExecutedIterator(ExplorerTy &Explorer)
      : Explorer(Explorer), CurInst(nullptr) {}

  MustBeExecutedIterator(ExplorerTy &Explorer,
                         const MustBeExecutedInterval::Position &Pos)
      : Explorer(Explorer), CurInst(Pos.getInstruction()) {
    if (Explorer.ExploreCFGForward)
      Head = Pos;
    if (Explorer.ExploreCFGBackward)
      Tail = Pos;
  }
  ///}

  /// Try to advance one of the underlying positions (Head or Tail).
  ///
  /// \return The next instruction in the must be executed context, or nullptr
  ///         if none was found.
  const Instruction *advance() {
    assert(CurInst && "Cannot advance an end iterator!");

    // Use the opcode to determine if we advance forward or backward first. This
    // allows to explore the context in both directions in a deterministic way.
    // We will advance in the other direction if the first try did not yield a
    // new instruction.
    //
    // TODO: Allow the user to choose different exploration styles.
    const auto &OpC = CurInst->getOpcode();
    MustBeExecutedInterval::Position *FirstPos = &Head, *SecondPos = &Tail;
    ExplorationDirection Direction = OpC % 2 ? ED_FORWARD : ED_BACKWARD;
    if (Direction == ED_BACKWARD)
      std::swap(FirstPos, SecondPos);

    if (const Instruction *I = advance(*FirstPos, Direction))
      return I;
    return advance(*SecondPos,
                   Direction == ED_FORWARD ? ED_BACKWARD : ED_FORWARD);
  }

  /// Try to advance the given position \p Pos in the direction \p Direction.
  ///
  /// \return The next instruction in the must be executed context, or nullptr
  ///         if none was found.
  const Instruction *advance(MustBeExecutedInterval::Position &Pos,
                             ExplorationDirection Direction) {
    // Check if the position is valid, if not, there is nothing to advance to.
    if (!Pos)
      return nullptr;

    // Check if we explored the new position already, if not, try to do it now.
    // Exploration can fail, e.g., there is no known "next" instruction or if
    // this a cached-only iterator. In such a case we return a nullptr.
    // Otherwise, we advance the position and indicate success by returning the
    // new instruction the position points to.
    explore(Pos, Direction);
    return Pos ? Pos.getInstruction() : nullptr;
  }

  /// Helper that delegates to Explorer.explore in non-cached-only mode and
  /// returns false otherwise.
  void explore(MustBeExecutedInterval::Position &Pos,
               ExplorationDirection Direction);

  /// A set to track the visited instructions in order to deal with endless
  /// loops and recursion.
  SmallPtrSet<PointerIntPair<const Instruction *, 1, ExplorationDirection>, 16>
      Visited;

  /// Call stack used in forward exploration of the call graph.
  SmallVector<const Instruction *, 0> ForwardCallStack;

  /// Call stack used in backward exploration of the call graph.
  SmallVector<const Instruction *, 0> BackwardCallStack;

  /// A reference to the explorer that created this iterator. The explorer is in
  /// charge of advancing the intervals the iterator works on.
  ExplorerTy &Explorer;

  /// The instruction we are currently exposing to the user. There is always an
  /// instruction that we know is executed with the given program point,
  /// initially the program point itself.
  const Instruction *CurInst;

  /// Two interval positions that mark the program points where this iterator
  /// will look for the next instruction. Note that the current instruction is
  /// either the one pointed to by Head, Tail, or both.
  MustBeExecutedInterval::Position Head, Tail;

  friend struct MustBeExecutedContextExplorer;
};

/// A "must be executed context" for a given program point PP is the set of
/// instructions, before and after PP, that are executed always when PP is
/// reached. The MustBeExecutedContextExplorer is a lazy and caching interface
/// to explore "must be executed contexts" in a module.
///
/// The explorer exposes parametrized "must be executed iterators" that traverse
/// the must be executed context according to the specified parameters.
struct MustBeExecutedContextExplorer {

  /// Create a CFG only explorer with the given analyses as support.
  MustBeExecutedContextExplorer(const DominatorTree *DT = nullptr,
                                const PostDominatorTree *PDT = nullptr,
                                const LoopInfo *LI = nullptr)
      : MustBeExecutedContextExplorer(
            true, true, false, false, [=](const Function &) { return LI; },
            [=](const Function &) { return DT; },
            [=](const Function &) { return PDT; }) {}

  /// \param ExploreCFGForward  Flag to indicate if instructions located after
  ///                           PP in the CFG, e.g., post-dominating PP, should
  ///                           be explored.
  /// \param ExploreCFGBackward Flag to indicate if instructions located before
  ///                           PP in the CFG, e.g., dominating PP, should be
  ///                           explored.
  /// \param ExploreCGForward   Flag to indicate if instructions located in a
  ///                           callee that is reached from PP should be
  ///                           explored. Hence if the call graph (CG) is
  ///                           traverse in forward direction.
  /// \param ExploreCGBackward  Flag to indicate if instructions located in a
  ///                           caller that leads to PP should be explored.
  ///                           Hence if the call graph (CG) is traverse in
  ///                           backwards direction.
  /// \param LIGetter  Return the LoopInfo analysis for a given function or a
  ///                  nullptr if it is not available.
  /// \param DTGetter  Return the DominatorTree analysis for a given function or
  ///                  a nullptr if it is not available.
  /// \param PDTGetter Return the PostDominatorTree analysis for a given
  ///                  function or a nullptr if it is not available.
  MustBeExecutedContextExplorer(
      bool ExploreCFGForward, bool ExploreCFGBackward, bool ExploreCGForward,
      bool ExploreCGBackward,
      GetterTy<const LoopInfo> LIGetter =
          [](const Function &) { return nullptr; },
      GetterTy<const DominatorTree> DTGetter =
          [](const Function &) { return nullptr; },
      GetterTy<const PostDominatorTree> PDTGetter =
          [](const Function &) { return nullptr; })
      : ExploreCFGForward(ExploreCFGForward),
        ExploreCFGBackward(ExploreCFGBackward),
        ExploreCGForward(ExploreCGForward),
        ExploreCGBackward(ExploreCGBackward), LIGetter(LIGetter),
        DTGetter(DTGetter), PDTGetter(PDTGetter) {}

  /// Clean up the dynamically allocated intervals.
  ~MustBeExecutedContextExplorer() {
    DeleteContainerPointers(Intervals);
  }

  /// Iterator-based interface. \see MustBeExecutedIterator.
  ///{
  using iterator = MustBeExecutedIterator</* CachedOnly */ false>;
  using const_iterator = MustBeExecutedIterator</* CachedOnly */ true>;

  /// Return an iterator to explore the context around \p PP.
  iterator begin(const Instruction *PP) {
    return iterator::begin(*this, getOrCreateIntervalPosition(PP));
  }

  /// Return an iterator to explore the cached context around \p PP.
  const_iterator begin(const Instruction *PP) const {
    return const_iterator::begin(*this, lookupIntervalPosition(PP));
  }

  /// Return an universal end iterator.
  ///{
  iterator end() { return iterator::end(*this); }
  iterator end(const Instruction *) { return iterator::end(*this); }

  const_iterator end() const { return const_iterator::end(*this); }
  const_iterator end(const Instruction *) const {
    return const_iterator::end(*this);
  }
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

  /// Replace the instruction \p OldI with the instruction \p NewI in the cached
  /// information.
  void replaceInstructionWith(const Instruction *OldI,
                              const Instruction *NewI) {
    // Check first if we even know anyting about the instruction, if not there
    // is nothing to do.
    MustBeExecutedInterval::Position Pos = lookupIntervalPosition(OldI);
    if (!Pos)
      return;

    assert(Pos.getInstruction() == OldI &&
           "Expected a position for the given instruction!");

    // Replace the instruction in the interval and the mapping to positions.
    (*Pos.Interval)[Pos.Offset] = NewI;
    MustBeExecutedMap.erase(OldI);
    MustBeExecutedMap[NewI] = Pos;
  }

  /// Insert the instruction \p NewI before the instruction \p PosI in the
  /// cached information.
  void insertInstructionBefore(const Instruction *NewI,
                               const Instruction *PosI)  {
    // Check first if we even know anyting about the position instruction, if
    // not there is nothing to do.
    MustBeExecutedInterval::Position Pos = lookupIntervalPosition(PosI);
    if (!Pos)
      return;

    assert(Pos.getInstruction() == PosI &&
           "Expected a position for the given instruction!");

    // First we handle the case where the position is the very last in the
    // interval.
    if (!Pos.Interval->isInbounds(Pos.Offset - 1) && !Pos.Interval->Prev) {
      // Check if we can prepend the new instruction.
      if (isGuaranteedToTransferExecutionToSuccessor(NewI)) {
        Pos.Interval->append(NewI, /* Front */ false);
        assert(Pos.Interval->isInbounds(Pos.Offset - 1) &&
               "Expected position to be valid after prepending instr.!");
      } else {
        MustBeExecutedInterval *NewInterval = new MustBeExecutedInterval(NewI);
        Intervals.push_back(NewInterval);
        Pos.Interval->Prev = NewInterval;
      }
    } else {
      // If the position is not the very last in the interval, we need insert
      // the new instruction at the right place. This will move the old ones.
      if (Pos.Offset > 0)
        Pos.Interval->NextInsts.insert(
            Pos.Interval->NextInsts.begin() + Pos.Offset, NewI);
      else
        Pos.Interval->PrevInsts.insert(
            Pos.Interval->PrevInsts.begin() - Pos.Offset + 1, NewI);
    }

    MustBeExecutedInterval::Position NewPos = Pos + ED_BACKWARD;
    assert(NewPos && NewPos.getInstruction() == NewI &&
            "Expected position to be valid and pointing to the new instr.!");
    MustBeExecutedMap[NewI] = NewPos;
  }

  /// Insert the instruction \p NewI after the instruction \p PosI in the
  /// cached information. This is not allowed for the last instruction in a
  /// basic block.
  void insertInstructionAfter(const Instruction *NewI,
                               const Instruction *PosI)  {
    assert(!PosI->isTerminator() &&
           "Cannot insert after terminator instructions!");
    assert(PosI->getNextNode() &&
           "Cannot insert after last instruction in a block!");
    return insertInstructionBefore(NewI, PosI->getNextNode());
  }

  /// Remove instruction \p I from the cached information.
  void removeInstruction(const Instruction *I) {
    // Check first if we even know anyting about the instruction, if not there
    // is nothing to do.
    MustBeExecutedInterval::Position Pos = lookupIntervalPosition(I);
    if (!Pos)
      return;

    assert(Pos.getInstruction() == I &&
           "Expected a position for the given instruction!");

    // Handle singleton intervals first.
    if (Pos.Interval->size() == 1) {
      assert(Pos.Offset == 0 && "Expect zero offset into singleton interval!");
      delete Pos.Interval;
    } else {
      // We avoid copying the buffer around by duplicating an existing
      // instruction instead. This requires a non-singleton interval.
      if (Pos.Interval->isInbounds(Pos.Offset + 1)) {
        (*Pos.Interval)[Pos.Offset] = (*Pos.Interval)[Pos.Offset + 1];
      } else {
        assert(
            Pos.Interval->isInbounds(Pos.Offset - 1) &&
            "Expected non-singleton interval to contain the previous or next "
            "instruction!");
        (*Pos.Interval)[Pos.Offset] = (*Pos.Interval)[Pos.Offset - 1];
      }
    }

    MustBeExecutedMap.erase(I);
  }
  ///}

private:
  /// Parameter that limit the performed exploration. See the constructor for
  /// their meaning.
  ///{
  bool ExploreCFGForward;
  bool ExploreCFGBackward;
  bool ExploreCGForward;
  bool ExploreCGBackward;
  ///}

  /// Getters for common CFG analyses: LoopInfo, DominatorTree, and
  /// PostDominatorTree.
  ///{
  GetterTy<const LoopInfo> LIGetter;
  GetterTy<const DominatorTree> DTGetter;
  GetterTy<const PostDominatorTree> PDTGetter;
  ///}

  /// Explore the context at position \p Pos in the direction \p Direction,
  /// update \p Pos in place.
  void explore(MustBeExecutedIterator</* CachedOnly */ false> &It,
               MustBeExecutedInterval::Position &Pos,
               ExplorationDirection Direction);

  /// Lookup the interval position for the instruction \p PP.
  MustBeExecutedInterval::Position
  lookupIntervalPosition(const Instruction *PP) const {
    return MustBeExecutedMap.lookup(PP);
  }

  /// Lookup or create a new interval position for the instruction \p PP.
  ///
  /// If \p PP was explored before, its interval position already exists and it
  /// is returned. If \p PP was not explored before, we create a new interval
  /// for it and make \p PP the center of the interval, thus the position at
  /// offset 0. The new position is cached.
  ///
  /// \returns A position for which the current instruction is \p PP.
  MustBeExecutedInterval::Position
  getOrCreateIntervalPosition(const Instruction *PP) {
    MustBeExecutedInterval::Position &Pos = MustBeExecutedMap[PP];

    if (!Pos) {
      MustBeExecutedInterval *Interval = new MustBeExecutedInterval(PP);
      Intervals.push_back(Interval);
      Pos = MustBeExecutedInterval::Position(Interval);
    }

    assert(Pos.getInstruction() == PP && "Unexpected instruction at position!");
    return Pos;
  }

  // TODO:
  const Instruction *getMustBeExecutedNextInstruction(
      MustBeExecutedIterator</* CachedOnly */ false> &It,
      const Instruction *PP);
  // TODO:
  const Instruction *getMustBeExecutedPrevInstruction(
      MustBeExecutedIterator</* CachedOnly */ false> &It,
      const Instruction *PP);

  DenseMap<const BasicBlock *, Optional<bool>> BlockTransferMap;
  DenseMap<const Function *, Optional<const Instruction *>>
      FunctionExitJoinPointMap;
  DenseMap<const Instruction *, MustBeExecutedInterval::Position>
      MustBeExecutedMap;

  SmallVector<MustBeExecutedInterval *, 16> Intervals;

  // TODO:
  friend class MustBeExecutedIterator<true>;
  friend class MustBeExecutedIterator<false>;
};

///
template<bool TrackThrowingBBs = true, uint64_t MaxInstToExplore = 0>
struct MustBeExecutedLoopSafetyInfo final : public LoopSafetyInfoInterface {

  MustBeExecutedLoopSafetyInfo(MustBeExecutedContextExplorer &Explorer)
      : Explorer(Explorer), It(Explorer.end()) {}

  /// See LoopSafetyInfoInterface::bockMayThrow(...).
  bool blockMayThrow(const BasicBlock *BB) const override {
    assert(TrackThrowingBBs && "Object was created without throw tracking.");
    return ThrowingBlocksMap.lookup(BB);
  }

  /// See LoopSafetyInfoInterface::bockMayThrow(...).
  bool anyBlockMayThrow() const override {
    assert(TrackThrowingBBs && "Object was created without throw tracking.");
    return ThrowingBlocksMap.lookup(nullptr);
  };

  /// See LoopSafetyInfoInterface::computeLoopSafetyInfo(...).
  void computeLoopSafetyInfo(const Loop *CurLoop) override {
    assert(CurLoop && "Expected a loop!");
    const Instruction &LoopFirstInst = CurLoop->getHeader()->front();
    It = Explorer.begin(&LoopFirstInst);

    // Explore the context to the fullest.
    // TODO: This might be a case where we want to guide the exploraiton.
    uint64_t InstExplorer = 0;
    while (const Instruction *I = *(++It))
      if (MaxInstToExplore && ++InstExplorer >= MaxInstToExplore)
        break;

    if (TrackThrowingBBs) {
      // Fill the ThrowingBlocksMap with basic block -> may throw information.
      bool AnyMayThrow = false;
      for (const BasicBlock *BB : CurLoop->blocks()) {
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

  /// See LoopSafetyInfoInterface::isGuaranteedToExecute(...).
  bool isGuaranteedToExecute(const Instruction &Inst, const DominatorTree *,
                             const Loop *CurLoop) const override {
    const BasicBlock *BB = Inst.getParent();
    if (allLoopPathsLeadToBlock(CurLoop, BB, /* DT */ nullptr)) {
      for (const Instruction &I : *BB) {
        if (&I == &Inst)
          return true;
        if (!isGuaranteedToTransferExecutionToSuccessor(&I))
          return false;
      }
    }
    return false;
  }

  /// See LoopSafetyInfoInterface::allLoopPathsLeadToBlock(...).
  bool allLoopPathsLeadToBlock(const Loop *, const BasicBlock *BB,
                               const DominatorTree *) const override {
    assert(BB && "Expected a loop and a block!");
    // We want to reach the first instruction in the block.
    const Instruction &BBFirstInst = BB->front();
    return It.count(&BBFirstInst);
  }

  /// See LoopSafetyInfoInterface::insertInstructionBefore(...).
  void insertInstructionBefore(const Instruction *NewI,
                               const Instruction *PosI) override {
    Explorer.insertInstructionBefore(NewI, PosI);
  }

  /// See LoopSafetyInfoInterface::insertInstructionAfter(...).
  void insertInstructionAfter(const Instruction *NewI,
                              const Instruction *PosI) override {
    Explorer.insertInstructionAfter(NewI, PosI);
  }

  /// See LoopSafetyInfoInterface::removeInstruction(...).
  void removeInstruction(const Instruction *Inst) override {
  Explorer.removeInstruction(Inst);
  }

private:
  MustBeExecutedContextExplorer Explorer;
  MustBeExecutedContextExplorer::iterator It;
  DenseMap<const BasicBlock *, bool> ThrowingBlocksMap;
};

} // namespace llvm

#endif
