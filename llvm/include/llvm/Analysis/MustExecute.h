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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/InstructionPrecedenceTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

namespace {
template <typename T> using GetterTy = std::function<T *(const Function &F)>;
}

class BasicBlock;
class DominatorTree;
class Instruction;
class Loop;
class LoopInfo;
class PostDominatorTree;

/// Captures loop safety information.
/// It keep information for loop blocks may throw exception or otherwise
/// exit abnormally on any iteration of the loop which might actually execute
/// at runtime.  The primary way to consume this information is via
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
  bool blockMayThrow(const BasicBlock *BB) const override;

  bool anyBlockMayThrow() const override;

  void computeLoopSafetyInfo(const Loop *CurLoop) override;

  bool isGuaranteedToExecute(const Instruction &Inst,
                             const DominatorTree *DT,
                             const Loop *CurLoop) const override;
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
  bool blockMayThrow(const BasicBlock *BB) const override;

  bool anyBlockMayThrow() const override;

  void computeLoopSafetyInfo(const Loop *CurLoop) override;

  bool isGuaranteedToExecute(const Instruction &Inst,
                             const DominatorTree *DT,
                             const Loop *CurLoop) const override;

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
};

bool mayContainIrreducibleControl(const Function &F, const LoopInfo *LI);

struct MustBeExecutedContextExplorer;

/// Enum that allows us to spell out the direction.
enum class ExplorationDirection {
  BACKWARD = 0,
  FORWARD = 1,
};

/// Must be executed intervals are stretches of instructions that are
/// guaranteed to be executed together without any other instruction
/// executed in-between. In addition, an interval `I` can end in a link into
/// another interval `J`. This means that the instructions of the interval
/// `J`, starting from the position the link points to, are always executed
/// before or respectively after the instruction in `I`. Note that there
/// might be other instructions executed in-between linked intervals.
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
/// Int | Previous | Instructions | Next
/// (1) |     None |        A ; B |  (3)
/// (2) |      (1) |        C ; D |  (3)
/// (3) |      (1) |            E | None
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
/// Int | Previous | Instructions | Next
/// (1) |     None |        A ; B |  (3)
/// (2) |      (1) |        C ; D |  (3)
/// (3) |      (1) |        E ; F | None
/// (4) |      (3) |            G | None
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
/// Int | Previous | Instructions | Next
/// (1) |     None |            A |  (7)
/// (2) |      (1) |            B |  (4)
/// (3) |      (2) |            C |  (4)
/// (4) |      (2) |            D |  (7)
/// (5) |      (4) |            E |  (6)
/// (6) |      (4) |            F |  (2)
/// (7) |      (1) |            G | None
///
///
/// Note that the examples show optimal intervals but not necessarily the ones
/// derived by the implementation. Also note that an interval has a direction
/// so there would be an interval for the forward direction and one for the
/// backward direction. The former is linked as described by the Next column in
/// the examples the latter as described by the Previous column.
///
struct MustBeExecutedInterval {

  /// A position on an interval.
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
    unsigned getOffset() const { return Offset; }

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
    Position &operator++() { return (*this) += 1; }

    Position operator++(int) {
      Position tmp(*this);
      ++(*this);
      return tmp;
    }

    Position &operator+=(unsigned N) {
      if (!bool(*this))
        return *this;

      // If the offset adjusted wrt. direction is still in the associated
      // interval, we adjust the offset and are done. Otherwise, we try to go to
      // the linked interval in the direction and set the offset accordingly,
      // hence to the very beginning or end depending on the direction. If there
      // is no linked interval, all members will be set to NULL and the position
      // becomes invalid.
      if (Interval->isInbounds(Offset + N)) {
        Offset += N;
        return *this;
      }

      if (Interval->getPosInNextInterval()) {
        unsigned Advanced = Interval->size() - Offset - 1;
        assert(Interval->size() >= Offset + 1 && "Unexpected offset!");
        assert(N >= Advanced + 1 && "Unexpected offset!");
        *this = Interval->getPosInNextInterval();
        return (*this) += N - Advanced - 1;
      }

      Interval = nullptr;
      Offset = 0;
      return *this;
    }
    ///}

    /// Print method for debugging purposes.
    void print(raw_ostream &OS, bool Detailed = false) const {
      OS << "{" << Offset << "@";
      if (Interval)
        Interval->print(OS, false);
      else
        OS << "[NONE:NONE:N]";
      OS << "}";
    }

  private:
    /// The interval this position is currently associated with.
    MustBeExecutedInterval *Interval;

    /// The current offset from the beginning of the associated interval.
    unsigned Offset;

    friend struct MustBeExecutedContextExplorer;
  };

  /// Constructor for a new interval centered around \p I.
  MustBeExecutedInterval(const Instruction &I, ExplorationDirection Dir)
      : Dir(Dir) {
    Insts.insert(&I);
  }

  /// Return the interval that contains the instructions executed after/before
  /// the ones in this interval.
  MustBeExecutedInterval *getNextInterval() const {
    return PosInNextInterval.getInterval();
  }

  /// Return the position in the next interval at which execution is known to
  /// continue.
  const Position &getPosInNextInterval() const { return PosInNextInterval; }

  /// Return the execution direction of this interval.
  ExplorationDirection getDirection() const { return Dir; }

  /// Return true if \p Offset is part of this interval, false otherwise.
  bool isInbounds(unsigned Offset) { return Offset < size(); }

  /// Subscript operator to allow easy access to the instructions based on their
  /// offset.
  const Instruction *operator[](unsigned Offset) const { return Insts[Offset]; }

  /// Return true if \p I is in this interval
  bool count(const Instruction &I) const { return Insts.count(&I); }

  /// Print method for debugging purposes.
  void print(raw_ostream &OS, bool Detailed = false) const {
    OS << "[" << this << "(" << size() << ") + ";
    PosInNextInterval.print(OS);
    OS << " : " << unsigned(getDirection()) << "]";
    if (Detailed) {
      OS << "\n";
      for (auto *I : Insts)
        OS << "- " << *I << "\n";
    }
  }

  /// Return the number of instructions contained in this interval.
  unsigned size() const { return Insts.size(); }

  /// Return true if this interval is final, that is no more extensions are
  /// possible.
  bool isFinalized() const { return IsFinalized; }

private:
  /// Indicate the interval is not going to be modified later on.
  void finalize() {
    assert(!IsFinalized && "Interval was marked finalized already!");
    IsFinalized = true;
  }

  /// Set the next interval, return true if successful, false otherwise.
  bool setPosInNextInterval(const Position &P) {
    assert(!PosInNextInterval && "Interval already has a link to another one!");
    assert(!IsFinalized && "Interval was marked finalized already!");

    // Do not create cyclic lists. The check is potentially expensive but should
    // not be in practise because we usually explore the context to the fullest
    // which will prevent an extension after a "PrevInterval" was set. However,
    // to prevent the quadratic worst case, we have a cut-off.
    MustBeExecutedInterval *MBEI = P.getInterval();
    if (HasPrevInterval || MBEI == this) {
      unsigned MaxChainLength = 6;
      while (MBEI) {
        if (MBEI == this || (--MaxChainLength == 0))
          return false;
        MBEI = MBEI->getNextInterval();
      }
    }

    PosInNextInterval = P;
    PosInNextInterval.getInterval()->HasPrevInterval = true;
    return true;
  }

  /// Extend the interval by \p I
  bool extend(const Instruction &I) {
    assert(!PosInNextInterval &&
           "Cannot advance the front if a next interval is already set!");
    assert(!IsFinalized && "Interval was marked finalized already!");

    // If we have seen the instruction we will not add it. This happens when we
    // encountered an endless loop contained entirely in this interval.
    return Insts.insert(&I);
  }

  /// The vectors that represent the instructions in the interval.
  SmallSetVector<const Instruction *, 8> Insts;

  /// The instruction executed next after the ones in Insts.
  Position PosInNextInterval;

  /// The direction in which the instruction in this interval are executed.
  /// Forward means Insts[i] is executed before Insts[i+1], backward means it is
  /// the other way around.
  ExplorationDirection Dir;

  /// A flag to indicate the interval is not going to change anymore. Used to
  /// prevent us from trying to extend it after an earlier attempted failed.
  bool IsFinalized = false;

  /// Flag to indicate that another interval points into this one. Used to
  /// determine if we need to check for potential endless loops or if there
  /// cannot be any.
  bool HasPrevInterval = false;

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
  /// explorer was provided at creation time.
  ///{
  MustBeExecutedIterator &operator++() {
    advance();
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
    return Pos[0] == Other.Pos[0] && Pos[1] == Other.Pos[1];
  }

  bool operator!=(const MustBeExecutedIterator &Other) const {
    return !(*this == Other);
  }
  ///}

  /// Return the underlying instruction.
  const Instruction *operator*() const { return getCurrentInst(); }
  const Instruction *getCurrentInst() const { return Pos[0].getInstruction(); }

  /// Return true if \p I is known to be found by this iterator, thus to be
  /// executed with the instruction this iterator was created for. If the known
  /// context does not contain \p I, the non-const version will try to use the
  /// Explorer, if available, to extend it.
  bool isExecutedWith(const Instruction &I) const;

  /// Configurable print method for debugging purposes.
  ///
  /// \param OS            The stream we print to.
  /// \param PrintInst     If set, the current instruction is printed.
  /// \param PrintInterval If set, the interval containing the current
  ///                      instruction is printed.
  void print(raw_ostream &OS, bool PrintInst, bool PrintInterval) const {
    if (PrintInst && getCurrentInst())
      OS << "[" << *getCurrentInst() << "]";

    if (PrintInterval)
      Pos[0].print(OS, false);
  }

private:
  /// Private constructors.
  MustBeExecutedIterator() {}
  MustBeExecutedIterator(const MustBeExecutedInterval::Position &FwdPos,
                         const MustBeExecutedInterval::Position &BwdPos,
                         MustBeExecutedContextExplorer *Explorer)
      : Explorer(Explorer) {
    Pos[0] = FwdPos;
    Pos[1] = BwdPos;
  }

  /// Advance one of the underlying positions (Head or Tail) potentially after
  /// exploring the context further (using Explorer). If this is not possible
  /// the positions (Pos) are invalidated.
  bool advance();

  /// The explorer that can be used to explore the context further if an end is
  /// found. If none is given the iterator will only traverse the
  /// existing/cached context otherwise the explorer is used to explore further.
  ///
  /// TODO: Determine if we should pass the explorer where needed.
  MustBeExecutedContextExplorer *Explorer = nullptr;

  /// Two interval positions that mark the program points where this iterator
  /// will look for the next instruction. Note that the current instruction is
  /// the one pointed to by Pos[0]. Once Pos[0] is explored to the fullest and
  /// becomes invalid we swap the two positions. If both positions are invalid
  /// the iterator traversed the entire context.
  MustBeExecutedInterval::Position Pos[2];

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
  using IntervalPosition = MustBeExecutedInterval::Position;

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
        DTGetter(DTGetter), PDTGetter(PDTGetter) {}

  /// Clean up the dynamically allocated intervals.
  ~MustBeExecutedContextExplorer() {
    for (MustBeExecutedInterval *Interval : Intervals)
      Interval->~MustBeExecutedInterval();
  }

  /// Iterator-based interface. \see MustBeExecutedIterator.
  ///{
  using iterator = MustBeExecutedIterator;
  using const_iterator = const MustBeExecutedIterator;

  /// Return an iterator to explore the context around \p PP.
  iterator begin(const Instruction &PP) {
    return iterator(
        getOrCreateIntervalPosition(PP, ExplorationDirection::FORWARD),
        getOrCreateIntervalPosition(PP, ExplorationDirection::BACKWARD), this);
  }

  /// Return an iterator to explore the cached context around \p PP.
  const_iterator cached_begin(const Instruction &PP) const {
    return iterator(lookupIntervalPosition(PP, ExplorationDirection::FORWARD),
                    lookupIntervalPosition(PP, ExplorationDirection::BACKWARD),
                    nullptr);
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
  llvm::iterator_range<const_iterator>
  cached_range(const Instruction &PP) const {
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
  bool
  checkForCachedInContext(const Instruction &PP,
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
  bool findInContextOf(const Instruction &I, const Instruction &PP) {
    const DominatorTree *DT = DTGetter(*I.getFunction());
    if (DT && DT->dominates(&I, &PP))
      return true;
    return begin(PP).isExecutedWith(I);
  }

  /// Return the next instruction that is guaranteed to be executed after \p PP.
  ///
  /// \param PP              The program point for which the next instruction
  ///                        that is guaranteed to execute is determined.
  const Instruction *getMustBeExecutedNextInstruction(const Instruction &PP) {
    const IntervalPosition &Pos =
        getOrCreateIntervalPosition(PP, ExplorationDirection::FORWARD);
    return (++iterator(Pos, IntervalPosition(), this)).getCurrentInst();
  }

  /// Return the previous instr. that is guaranteed to be executed before \p PP.
  ///
  /// \param PP              The program point for which the previous instr.
  ///                        that is guaranteed to execute is determined.
  const Instruction *getMustBeExecutedPrevInstruction(const Instruction &PP) {
    const IntervalPosition &Pos =
        getOrCreateIntervalPosition(PP, ExplorationDirection::FORWARD);
    return (++iterator(Pos, IntervalPosition(), this)).getCurrentInst();
  }

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

  /// Return the next position that is guaranteed to be executed (at some
  /// point but for sure) after \p P in the direction of the underlying
  /// interval. If none is known returns an invalid position. Note that \p P has
  /// to be a valid position.
  IntervalPosition exploreNextPosition(const IntervalPosition &P);

private:
  /// Return an instruction that is known to be executed after \p PP.
  const Instruction *exploreNextExecutedInstruction(const Instruction &PP);

  /// Return an instruction that is known to be executed before \p PP.
  const Instruction *explorePrevExecutedInstruction(const Instruction &PP);

  /// Lookup the interval position for the instruction \p PP.
  IntervalPosition lookupIntervalPosition(const Instruction &PP,
                                          ExplorationDirection Dir) const {
    return InstPositionMap[unsigned(Dir)].lookup(&PP);
  }

  /// Lookup or create a new interval position for the instruction \p PP.
  ///
  /// If \p PP was explored before, its interval position already exists and it
  /// is returned. If \p PP was not explored before, we create a new interval
  /// for it and make \p PP the center of the interval, thus the position at
  /// offset 0. The new position is cached.
  ///
  /// \returns A position for which the current instruction is \p PP.
  IntervalPosition getOrCreateIntervalPosition(const Instruction &PP,
                                               ExplorationDirection Dir) {
    IntervalPosition &Pos = InstPositionMap[unsigned(Dir)][&PP];

    if (!Pos) {
      MustBeExecutedInterval *Interval =
          new (Allocator) MustBeExecutedInterval(PP, Dir);
      Intervals.push_back(Interval);
      Pos = IntervalPosition(Interval);
    }

    assert(Pos.getInstruction() == &PP &&
           "Unexpected instruction at position!");
    return Pos;
  }

  /// Getters for common CFG analyses: LoopInfo, DominatorTree, and
  /// PostDominatorTree.
  ///{
  GetterTy<const LoopInfo> LIGetter;
  GetterTy<const DominatorTree> DTGetter;
  GetterTy<const PostDominatorTree> PDTGetter;
  ///}

  /// Map to cache containsIrreducibleCFG results.
  DenseMap<const Function *, Optional<bool>> IrreducibleControlMap;

  /// Two map from instructions to associated positions, one for each direction.
  DenseMap<const Instruction *, IntervalPosition> InstPositionMap[2];

  /// All intervals allocated with the BumpPtrAllocator Allocator.
  SmallVector<MustBeExecutedInterval *, 16> Intervals;

  /// The allocator used to allocate memory, e.g. for intervals.
  BumpPtrAllocator Allocator;
};

} // namespace llvm

#endif
