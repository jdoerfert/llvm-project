//===- Transform/Utils/CodeMoverUtils.h - CodeMover Utils -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This family of functions determine movements are safe on basic blocks, and
// instructions contained within a function.
//
// Please note that this is work in progress, and the functionality is not
// ready for broader production use.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_CODEMOVERUTILS_H
#define LLVM_TRANSFORMS_UTILS_CODEMOVERUTILS_H

#include "llvm/Support/Compiler.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class Value;
class BasicBlock;
class DependenceInfo;
class DominatorTree;
class Instruction;
class PostDominatorTree;
class ScalarEvolution;

/// Represent a control condition. A control condition is a condition of a
/// terminator to decide which successors to execute. The pointer field
/// represents the address of the condition of the terminator. The integer field
/// is a bool, it is true when the basic block is executed when V is true. For
/// example, `br %cond, bb0, bb1` %cond is a control condition of bb0 with the
/// integer field equals to true, while %cond is a control condition of bb1 with
/// the integer field equals to false.
using ControlCondition = PointerIntPair<Value *, 1, bool>;

/// Represent a set of control conditions required to execute ToBB from FromBB.
class ControlConditions {
  using ConditionVectorTy = SmallVector<ControlCondition, 6>;

  /// A SmallVector of control conditions.
  ConditionVectorTy Conditions;

public:
  /// Return a ControlConditions which stores all conditions required to execute
  /// \p BB from \p Dominator. If \p MaxLookup is non-zero, it limits the
  /// number of conditions to collect. Return std::nullopt if not all conditions
  /// are collected successfully, or we hit the limit.
  static const std::optional<ControlConditions>
  collectControlConditions(const BasicBlock &BB, const BasicBlock &Dominator,
                           const DominatorTree &DT,
                           const PostDominatorTree &PDT,
                           unsigned MaxLookup = 6);

  /// Return true if there exists no control conditions required to execute ToBB
  /// from FromBB.
  bool isUnconditional() const { return Conditions.empty(); }

  /// Return a constant reference of Conditions.
  const ConditionVectorTy &getControlConditions() const { return Conditions; }

  /// Add \p V as one of the ControlCondition in Condition with IsTrueCondition
  /// equals to \p True. Return true if inserted successfully.
  bool addControlCondition(ControlCondition C);

  /// Return true if for all control conditions in Conditions, there exists an
  /// equivalent control condition in \p Other.Conditions.
  bool isEquivalent(const ControlConditions &Other) const;

  /// Return true if \p C1 and \p C2 are equivalent.
  static bool isEquivalent(const ControlCondition &C1,
                           const ControlCondition &C2);

private:
  ControlConditions() = default;

  static bool isEquivalent(const Value &V1, const Value &V2);
  static bool isInverse(const Value &V1, const Value &V2);
};

raw_ostream &operator<<(raw_ostream &OS, const ControlCondition &C);

/// Return true if \p I0 and \p I1 are control flow equivalent.
/// Two instructions are control flow equivalent if their basic blocks are
/// control flow equivalent.
bool isControlFlowEquivalent(const Instruction &I0, const Instruction &I1,
                             const DominatorTree &DT,
                             const PostDominatorTree &PDT);

/// Return true if \p BB0 and \p BB1 are control flow equivalent.
/// Two basic blocks are control flow equivalent if when one executes, the other
/// is guaranteed to execute.
bool isControlFlowEquivalent(const BasicBlock &BB0, const BasicBlock &BB1,
                             const DominatorTree &DT,
                             const PostDominatorTree &PDT);
>>>>>>> c62ee6238853 ([WIP] Instrumentor/InputGen conditions)

/// Return true if \p I can be safely moved before \p InsertPoint.
LLVM_ABI bool isSafeToMoveBefore(Instruction &I, Instruction &InsertPoint,
                                 DominatorTree &DT,
                                 const PostDominatorTree *PDT = nullptr,
                                 DependenceInfo *DI = nullptr,
                                 bool CheckForEntireBlock = false);

/// Return true if all instructions (except the terminator) in \p BB can be
/// safely moved before \p InsertPoint.
LLVM_ABI bool isSafeToMoveBefore(BasicBlock &BB, Instruction &InsertPoint,
                                 DominatorTree &DT,
                                 const PostDominatorTree *PDT = nullptr,
                                 DependenceInfo *DI = nullptr);

/// Move instructions, in an order-preserving manner, from \p FromBB to the
/// beginning of \p ToBB when proven safe.
LLVM_ABI void
moveInstructionsToTheBeginning(BasicBlock &FromBB, BasicBlock &ToBB,
                               DominatorTree &DT, const PostDominatorTree &PDT,
                               DependenceInfo &DI, ScalarEvolution &SE);
/// Move instructions, in an order-preserving manner, from \p FromBB to the end
/// of \p ToBB when proven safe.
LLVM_ABI void moveInstructionsToTheEnd(BasicBlock &FromBB, BasicBlock &ToBB,
                                       DominatorTree &DT,
                                       const PostDominatorTree &PDT,
                                       DependenceInfo &DI, ScalarEvolution &SE);

/// In case that two BBs \p ThisBlock and \p OtherBlock are control flow
/// equivalent but they do not strictly dominate and post-dominate each
/// other, we determine if \p ThisBlock is reached after \p OtherBlock
/// in the control flow.
LLVM_ABI bool nonStrictlyPostDominate(const BasicBlock *ThisBlock,
                                      const BasicBlock *OtherBlock,
                                      const DominatorTree *DT,
                                      const PostDominatorTree *PDT);

// Check if I0 is reached before I1 in the control flow.
LLVM_ABI bool isReachedBefore(const Instruction *I0, const Instruction *I1,
                              const DominatorTree *DT,
                              const PostDominatorTree *PDT);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CODEMOVERUTILS_H
