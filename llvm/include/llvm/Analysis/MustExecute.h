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
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/InstructionPrecedenceTracking.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Allocator.h"

namespace llvm {

namespace {
template <typename T> using GetterTy = std::function<T *(const Function &F)>;
}

class Instruction;
class DominatorTree;
class PostDominatorTree;
class Loop;

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
class LoopSafetyInfo {
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

  /// Returns true iff the block \p BB potentially may throw exception. It can
  /// be false-positive in cases when we want to avoid complex analysis.
  virtual bool blockMayThrow(const BasicBlock *BB) const = 0;

  /// Returns true iff any block of the loop for which this info is contains an
  /// instruction that may throw or otherwise exit abnormally.
  virtual bool anyBlockMayThrow() const = 0;

  /// Return true if we must reach the block \p BB under assumption that the
  /// loop \p CurLoop is entered.
  bool allLoopPathsLeadToBlock(const Loop *CurLoop, const BasicBlock *BB,
                               const DominatorTree *DT) const;

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

  /// Inform safety info that we are planning to remove the instruction \p Inst
  /// from its block. It will make all cache updates to keep it correct after
  /// this removal.
  void removeInstruction(const Instruction *Inst);

  ICFLoopSafetyInfo(DominatorTree *DT) : LoopSafetyInfo(), ICF(DT), MW(DT) {};

  virtual ~ICFLoopSafetyInfo() {};
};

bool mayContainIrreducibleControl(const Function &F, const LoopInfo *LI);

struct MustBeExecutedContextExplorer;

/// Enum that allows us to spell out the direction.
enum class ExplorationDirection {
  BACKWARD = 0,
  FORWARD = 1,
};

/// Must be executed intervals are stretches of instructions that are guaranteed
/// to be executed together without any other instruction executed in-between.
/// In addition, intervals are linked to intervals always executed before and
/// respectively afterward, if they exit. Note that there might be other
/// instructions executed in-between linked intervals.
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
/// Int | Instructions | Previous | Next
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
/// Int | Instructions | Previous | Next
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
/// Int | Instructions | Previous | Next
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
/// derived by the implementation.
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
  MustBeExecutedInterval(const Instruction *I) {
    assert(I && "Cannot construct an interval for a null pointer!");
    // Put the instruction I in both vectors to make the offset and in-bounds
    // calculation easier.
    NextInsts.push_back(I);
    PrevInsts.push_back(I);
    Instructions.insert(I);
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

  /// Return true if \p I is in this interval
  bool count(const Instruction &I) const { return Instructions.count(&I); }

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
  int size() const {
    return NextInsts.size() + PrevInsts.size() /* center is duplicated */ - 1;
  }

  /// Return the number of instructions in this interval in \p Dir from
  /// the center.
  int size(ExplorationDirection Dir) const {
    if (Dir == ExplorationDirection::FORWARD)
      return NextInsts.size();
    return PrevInsts.size();
  }

  /// A position on the interval chain.
  ///
  /// See the position section in the MustBeExecutedInterval description for
  /// more information.
  struct Position {

    Position(MustBeExecutedInterval *Interval = nullptr)
        : Interval(Interval), Offset(0) {}

    /// Return the instruction at this position.
    const Instruction *getInstruction() const {
      assert(bool(*this) &&
             "Invalid positions do not have an associated instruction!");
      return (*Interval)[Offset];
    }

    /// Return the interval at this position.
    MustBeExecutedInterval *getInterval() const { return Interval; }

    /// Allow valid positions to evaluate to 'true' and invalid ones to 'false'
    /// when a position is converted to a boolean.
    operator bool() const { return Interval; }

    /// Equality operator.
    bool operator==(const Position &Other) const {
      return Interval == Other.Interval && Offset == Other.Offset;
    }

    /// Advance the position in the direction \p Dir.
    ///
    /// If there is no next instruction in the direction, this will result in
    /// an invalid position, hence the interval will be a nullptr.
    ///{
    friend Position operator+(Position Pos, ExplorationDirection Dir) {
      return Pos += Dir;
    }

    Position &operator+=(ExplorationDirection Dir) {
      if (!bool(*this))
        return *this;

      int DirectionOffset = Dir == ExplorationDirection::FORWARD ? 1 : -1;

      // If the offset adjusted wrt. direction is still in the associated
      // interval, we adjust the offset and are done. Otherwise, we try to go to
      // the linked interval in the direction and set the offset accordingly,
      // hence to the very beginning or end depending on the direction. If there
      // is no linked interval, all members will be set to NULL and the position
      // becomes invalid.
      if (Interval->isInbounds(Offset + DirectionOffset)) {
        Offset += DirectionOffset;
      } else if (auto *NextI =
                     Interval->getNextIntervalInDirection(Dir)) {
        Interval = NextI;
        Offset = Dir == ExplorationDirection::FORWARD
                     ? 1 - Interval->PrevInsts.size()
                     : Interval->NextInsts.size() - 1;
      } else {
        Interval = nullptr;
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

  /// Return the next interval in direction \p Dir, if any.
  MustBeExecutedInterval *
  getNextIntervalInDirection(ExplorationDirection Dir) const {
    return Dir == ExplorationDirection::FORWARD ? Next : Prev;
  }

private:
  /// Set the next interval in direction \p Dir to \p Other.
  bool setNextIntervalInDirection(ExplorationDirection Dir,
                                  MustBeExecutedInterval &Other) {
    // Do not create endless loops.
    MustBeExecutedInterval *MBEI = &Other;
    while (MBEI) {
      if (MBEI == this)
        return false;
      MBEI = Dir == ExplorationDirection::FORWARD ? MBEI->Next : MBEI->Prev;
    }

    MustBeExecutedInterval *&Ptr =
        Dir == ExplorationDirection::FORWARD ? Next : Prev;
    assert(!Ptr && "Expected open interval!");
    Ptr = &Other;
    return true;
  }

  /// Extend the interval by \p I in direction \p Dir.
  ///
  /// \returns The argument \p I if it was added and nullptr if it was already
  ///          contained.
  const Instruction *extend(const Instruction &I, ExplorationDirection Dir) {
    // If we have seen the instruction we will not add it. This happens when we
    // encountered an endless loop.
    if (!Instructions.insert(&I).second)
      return nullptr;

    if (Dir == ExplorationDirection::FORWARD) {
    NextInsts.push_back(&I);
    assert(!Next &&
           "Cannot advance the front if a next interval is already set!");
    } else {
    PrevInsts.push_back(&I);
    assert(!Prev &&
           "Cannot advance the back if a previous interval is already set!");
    }
    return &I;
  }

  /// The vectors that represent the instructions in the interval.
  ///{
  using InstructionVector = SmallVector<const Instruction *, 8>;
  InstructionVector PrevInsts, NextInsts;
  ///}

  /// The set of all instructions in this interval.
  SmallPtrSet<const Instruction *, 16> Instructions;

  /// The links to the previous and next interval.
  MustBeExecutedInterval *Prev = nullptr, *Next = nullptr;

  friend struct MustBeExecutedContextExplorer;
};

/// Must be executed iterators visit stretches of instructions, so called
/// must-be-executed-intervals, that are guaranteed to be executed together,
/// potentially with other instruction executed in-between.
///
/// See MustBeExecutedInterval for a discussion about the intervals of these
/// examples.
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
///    B                | A, B,       E, F
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
/// Start Instruction    | Visit Set
/// A                    | A, B
///    B                 | A, B
///       C              | A, B, C
///          D           | A, B,    D
///             E        | A, B,    D, E, F
///                F     | A, B,    D,    F
///                   G  | A, B,    D,       G
///
///
/// Note that the examples show optimal visist sets but not necessarily the ones
/// derived by the explorer depending on the available CFG analyses (see
/// MustBeExecutedContextExplorer). Also note that we, depending on the options,
/// the visit set can contain instructions from other functions.
///
/// Depending on the template parameter \p CachedOnly, the iterator will either
/// only visit existing instructions in the interval (\p CachedOnly == true) or
/// also use the MustBeExecutedContextExplorer to extend the intervals if
/// needed.
struct MustBeExecutedIterator {
  /// Type declarations that make his class an input iterator.
  ///{
  typedef const Instruction *value_type;
  typedef std::ptrdiff_t difference_type;
  typedef const Instruction **pointer;
  typedef const Instruction *&reference;
  typedef std::input_iterator_tag iterator_category;
  ///}

  /// Pre- and post-increment operators. Both can cause the underlying intervals
  /// to be extended if this is not a cached-only iterator, that is if an
  /// explorer was provided at creation time..
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

  /// Apply \p Pred on all known reachable intervals and return true if \p Pred
  /// returned true for any of them, false otherwise.
  bool
  anyInterval(llvm::function_ref<bool(MustBeExecutedInterval &)> Pred) const {
    return anyIntervalImpl(Head.getInterval(), ExplorationDirection::FORWARD,
                           Pred) ||
           anyIntervalImpl(Tail.getInterval(), ExplorationDirection::BACKWARD,
                           Pred);
  }

  /// Return true if \p I is known to be found by this iterator, thus to be
  /// executed with the instruction this iterator was created for. If the known
  /// context does not contain \p I, the non-const version will try to use the
  /// Explorer, if available, to extend it.
  bool isExecutedWith(const Instruction &I);
  bool isExecutedWithCached(const Instruction &I) const;

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
  /// Apply \p Pred on all known reachable intervals from \p CurI in direction
  /// \p Dir and return true if \p Pred returned true for any of them,
  /// false otherwise.
  static bool
  anyIntervalImpl(MustBeExecutedInterval *CurI, ExplorationDirection Dir,
                  llvm::function_ref<bool(MustBeExecutedInterval &)> Pred) {
    while (CurI) {
      if (Pred(*CurI))
        return true;
      CurI = CurI->getNextIntervalInDirection(Dir);
    }
    return false;
  }

  /// Private constructors.
  MustBeExecutedIterator() : Explorer(nullptr), CurInst(nullptr) {}
  MustBeExecutedIterator(const MustBeExecutedInterval::Position &Pos,
                         MustBeExecutedContextExplorer *Explorer)
      : Explorer(Explorer), CurInst(Pos.getInstruction()), Head(Pos),
        Tail(Pos) {}

  MustBeExecutedContextExplorer *Explorer;

  /// Try to advance one of the underlying positions (Head or Tail).
  ///
  /// \return The next instruction in the must be executed context, or nullptr
  ///         if none was found.
  const Instruction *tryToAdvance();

  /// Advance one of the underlying positions (Head or Tail) potentially after
  /// exploring the contex further. If this is not possible the positions are
  /// invalidated.
  ///
  /// \return The next instruction in the must be executed context, or nullptr
  ///         if none was found.
  const Instruction *advance();

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
/// instructions, potentially before and after PP, that are executed always when
/// PP is reached. The MustBeExecutedContextExplorer an interface to explore
/// "must be executed contexts" in a module through the use of
/// MustBeExecutedIterator.
///
/// The explorer exposes "must be executed iterators" that traverse the must be
/// executed context. There is little information sharing between iterators as
/// the expected use case involves few iterators for "far apart" instructions.
/// If that changes, we should consider caching more intermediate results.
struct MustBeExecutedContextExplorer {

  struct PositionPair {
    MustBeExecutedInterval::Position Forward;
    MustBeExecutedInterval::Position Backward;

    operator bool() const { return Forward || Backward; }
    MustBeExecutedInterval::Position &operator[](ExplorationDirection Dir) {
      return Dir == ExplorationDirection::FORWARD ? Forward : Backward;
    }
  };

  /// In the description of the parameters we use PP to denote a program point
  /// for which the must be executed context is explored, or put differently,
  /// for which the MustBeExecutedIterator is created.
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
  MustBeExecutedContextExplorer(
      bool ExploreInterBlock, bool ExploreCFGForward, bool ExploreCFGBackward,
      GetterTy<const LoopInfo> LIGetter =
          [](const Function &) { return nullptr; },
      GetterTy<const DominatorTree> DTGetter =
          [](const Function &) { return nullptr; },
      GetterTy<const PostDominatorTree> PDTGetter =
          [](const Function &) { return nullptr; })
      : ExploreInterBlock(ExploreInterBlock),
        ExploreCFGForward(ExploreCFGForward),
        ExploreCFGBackward(ExploreCFGBackward), LIGetter(LIGetter),
        DTGetter(DTGetter), PDTGetter(PDTGetter){}

  /// Iterator-based interface. \see MustBeExecutedIterator.
  ///{
  using iterator = MustBeExecutedIterator;
  using const_iterator = const MustBeExecutedIterator;

  /// Return an iterator to explore the context around \p PP.
  iterator begin(const Instruction &PP) {
    return iterator(getOrCreateIntervalPosition(PP), this);
  }

  /// Return an iterator to explore the cached context around \p PP.
  const_iterator cached_begin(const Instruction &PP) const {
    return iterator(lookupIntervalPosition(PP), nullptr);
  }

  /// Return an universal end iterator.
  ///{
  iterator end() { return iterator(); }
  iterator end(const Instruction &) { return iterator(); }

  const_iterator cached_end() const { return const_iterator(); }
  const_iterator cached_end(const Instruction &) const {
    return const_iterator();
  }
  ///}

  /// Return an iterator range to explore the context around \p PP.
  llvm::iterator_range<iterator> range(const Instruction &PP) {
    return llvm::make_range(begin(PP), end(PP));
  }

  /// Return an iterator range to explore the cached context around \p PP.
  llvm::iterator_range<const_iterator> cached_range(const Instruction &PP) const {
    return llvm::make_range(cached_begin(PP), cached_end(PP));
  }
  ///}

  /// Check \p Pred on all instructions in the context.
  ///
  /// This method will evaluate \p Pred and return
  /// true if \p Pred holds in every instruction.
  ///
  ///{
  bool checkForAllInContext(const Instruction &PP,
                          function_ref<bool(const Instruction *)> Pred) {
    return llvm::all_of(range(PP), Pred);
  }
  bool checkForCachedInContext(const Instruction &PP,
                          function_ref<bool(const Instruction *)> Pred) const {
    return llvm::all_of(cached_range(PP), Pred);
  }
  ///}

  /// Helper to look for \p I in the context of \p PP.
  ///
  /// In the non-const variant the context is expanded until \p I was found or
  /// no more expansion is possible. If a dominator tree getter was provided it
  /// is used to circumvent the search.
  ///
  /// \returns True, iff \p I was found.
  ///
  ///{
  bool findInContextOf(const Instruction &I, const Instruction &PP) {
    const DominatorTree *DT = DTGetter(*I.getFunction());
    if (DT && DT->dominates(&I, &PP))
      return true;
    return begin(PP).isExecutedWith(I);
  }
  bool findInCachedContextOf(const Instruction &I,
                             const Instruction &PP) const {
    const DominatorTree *DT = DTGetter(*I.getFunction());
    if (DT && DT->dominates(&I, &PP))
      return true;
    return cached_begin(PP).isExecutedWithCached(I);
  }
  ///}

  /// Return the next instruction that is guaranteed to be executed after \p PP.
  ///
  /// \param PP              The program point for which the next instruction
  ///                        that is guaranteed to execute is determined.
  const Instruction *getMustBeExecutedNextInstruction(const Instruction &PP) {
    return getMustBeExecutedNextInstruction(getOrCreateIntervalPosition(PP));
  }

  /// Return the previous instr. that is guaranteed to be executed before \p PP.
  ///
  /// \param PP              The program point for which the previous instr.
  ///                        that is guaranteed to execute is determined.
  const Instruction *getMustBeExecutedPrevInstruction(const Instruction &PP) {
    return getMustBeExecutedPrevInstruction(getOrCreateIntervalPosition(PP));
  }

  /// Return the next instruction that is guaranteed to be executed after \p P.
  ///
  /// \param P               The program point for which the next instruction
  ///                        that is guaranteed to execute is determined.
  const Instruction *
  getMustBeExecutedNextInstruction(const MustBeExecutedInterval::Position &P);

  /// Return the previous instr. that is guaranteed to be executed before \p P.
  ///
  /// \param P               The program point for which the previous instr.
  ///                        that is guaranteed to execute is determined.
  const Instruction *
  getMustBeExecutedPrevInstruction(const MustBeExecutedInterval::Position &P);

  /// Find the next join point from \p InitBB in forward direction.
  const BasicBlock *findForwardJoinPoint(const BasicBlock *InitBB);

  /// Find the next join point from \p InitBB in backward direction.
  const BasicBlock *findBackwardJoinPoint(const BasicBlock *InitBB);

  /// Parameter that limit the performed exploration. See the constructor for
  /// their meaning.
  ///{
  const bool ExploreInterBlock;
  const bool ExploreCFGForward;
  const bool ExploreCFGBackward;
  ///}

  void dump();

private:
  const Instruction *
  extendIntervalViaLink(const MustBeExecutedInterval::Position &Pos,
                        const Instruction &I, ExplorationDirection Dir) {
    assert(Pos && "Cannot extend an invalid position!");
    MustBeExecutedInterval &Interval = *Pos.Interval;

    const MustBeExecutedInterval::Position &IPos =
        getOrCreateIntervalPosition(I);
    if (Interval.setNextIntervalInDirection(Dir, *IPos.Interval))
    return &I;
    return nullptr;
  }

  const Instruction *extendInterval(const MustBeExecutedInterval::Position &Pos,
                                    const Instruction &I,
                                    ExplorationDirection Dir) {
    assert(Pos && "Cannot extend an invalid position!");
    MustBeExecutedInterval &Interval = *Pos.Interval;

    // Check if we have an interval for I already. If so we do not add I to the
    // interval Pos points into but link that interval against the existing one
    PositionPair &PosPair = InstPositionMap[&I];;
    MustBeExecutedInterval::Position &IPos = PosPair[Dir];
    if (IPos) {
      if (Interval.setNextIntervalInDirection(Dir, *IPos.Interval))
        return &I;
      return nullptr;
    }

    // Check if we can extend I, this may fail if we have an endless loop
    // contained entirely in the interval.
    if (!Interval.extend(I, Dir))
      return nullptr;

    // If we extended the interval we remember the position for I.
    IPos = Pos + Dir;
    return &I;
  }

  /// Lookup the interval position for the instruction \p PP.
  MustBeExecutedInterval::Position
  lookupIntervalPosition(const Instruction &PP) const {
    return InstPositionMap.lookup(&PP);
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
  getOrCreateIntervalPosition(const Instruction &PP) {
    //errs() << "getOrCreateIntervalPosition " << PP << "\n";
    PositionPair &PosPair = InstPositionMap[&PP];;
    //errs() << "getOrCreateIntervalPosition " << PP << " : " << (!!Pos) << "\n";

    if (!PosPair) {
      MustBeExecutedInterval *Interval =
          new (Allocator) MustBeExecutedInterval(&PP);
      Pos = MustBeExecutedInterval::Position(Interval);
      //errs() << "Newpos: " << Pos << "\n";
    }

    assert(Pos.getInstruction() == &PP && "Unexpected instruction at position!");
    return Pos;
  }

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

  /// Map from instructions to associated must be executed iterators.
  DenseMap<const Instruction *, PositionPair> InstPositionMap;

  /// The allocator used to allocate memory, e.g. for intervals.
  BumpPtrAllocator Allocator;
};

} // namespace llvm

#endif
