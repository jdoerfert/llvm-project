//===- Attributor.cpp - Module-wide attribute deduction -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an inter procedural pass that deduces and/or propagating
// attributes. This is done in an abstract interpretation style fixpoint
// iteration. See the Attributor.h file comment and the class descriptions in
// that file for more information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Attributor.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "attributor"

STATISTIC(NumFnWithExactDefinition,
          "Number of function with exact definitions");
STATISTIC(NumFnWithoutExactDefinition,
          "Number of function without exact definitions");
STATISTIC(NumAttributesTimedOut,
          "Number of abstract attributes timed out before fixpoint");
STATISTIC(NumAttributesValidFixpoint,
          "Number of abstract attributes in a valid fixpoint state");
STATISTIC(NumAttributesManifested,
          "Number of abstract attributes manifested in IR");
STATISTIC(NumAttributesSkippedDueToIR,
          "Number of abstract attributes skipped due to presence in IR");

STATISTIC(NumFnNoUnwind, "Number of functions marked nounwind");
STATISTIC(NumFnUniqueReturned, "Number of function with unique return");
STATISTIC(NumFnKnownReturns, "Number of function with known return values");
STATISTIC(NumFnArgumentReturned,
          "Number of function arguments marked returned");
STATISTIC(NumFnNoSync, "Number of functions marked nosync");
STATISTIC(NumFnNoFree, "Number of functions marked nofree");
STATISTIC(NumFnReturnedNonNull,
          "Number of function return values marked nonnull");
STATISTIC(NumFnArgumentNonNull, "Number of function arguments marked nonnull");
STATISTIC(NumCSArgumentNonNull, "Number of call site arguments marked nonnull");
STATISTIC(NumFnWillReturn, "Number of functions marked willreturn");
STATISTIC(NumFnArgumentNoAlias, "Number of function arguments marked noalias");
STATISTIC(NumFnReturnedDereferenceable,
          "Number of function return values marked dereferenceable");
STATISTIC(NumFnArgumentDereferenceable,
          "Number of function arguments marked dereferenceable");
STATISTIC(NumCSArgumentDereferenceable,
          "Number of call site arguments marked dereferenceable");
STATISTIC(NumFnReturnedAlign, "Number of function return values marked align");
STATISTIC(NumFnArgumentAlign, "Number of function arguments marked align");
STATISTIC(NumCSArgumentAlign, "Number of call site arguments marked align");

// TODO: Determine a good default value.
//
// In the LLVM-TS and SPEC2006, 32 seems to not induce compile time overheads
// (when run with the first 5 abstract attributes). The results also indicate
// that we never reach 32 iterations but always find a fixpoint sooner.
//
// This will become more evolved once we perform two interleaved fixpoint
// iterations: bottom-up and top-down.
static cl::opt<unsigned>
    MaxFixpointIterations("attributor-max-iterations", cl::Hidden,
                          cl::desc("Maximal number of fixpoint iterations."),
                          cl::init(32));

static cl::opt<bool> DisableAttributor(
    "attributor-disable", cl::Hidden,
    cl::desc("Disable the attributor inter-procedural deduction pass."),
    cl::init(true));

static cl::opt<bool> VerifyAttributor(
    "attributor-verify", cl::Hidden,
    cl::desc("Verify the Attributor deduction and "
             "manifestation of attributes -- may issue false-positive errors"),
    cl::init(false));

/// Logic operators for the change status enum class.
///
///{
ChangeStatus llvm::operator|(ChangeStatus l, ChangeStatus r) {
  return l == ChangeStatus::CHANGED ? l : r;
}
ChangeStatus llvm::operator&(ChangeStatus l, ChangeStatus r) {
  return l == ChangeStatus::UNCHANGED ? l : r;
}
///}

/// Helper to adjust the statistics.
static void bookkeeping(AbstractAttribute::ManifestPosition MP,
                        const Attribute &Attr) {
  if (!AreStatisticsEnabled())
    return;

  switch (Attr.getKindAsEnum()) {
  case Attribute::Alignment:
    switch (MP) {
    case AbstractAttribute::MP_RETURNED:
      NumFnReturnedAlign++;
      break;
    case AbstractAttribute::MP_ARGUMENT:
      NumFnArgumentAlign++;
      break;
    case AbstractAttribute::MP_CALL_SITE_ARGUMENT:
      NumCSArgumentAlign++;
      break;
    default:
      break;
    }
    break;
  case Attribute::Dereferenceable:
    switch (MP) {
    case AbstractAttribute::MP_RETURNED:
      NumFnReturnedDereferenceable++;
      break;
    case AbstractAttribute::MP_ARGUMENT:
      NumFnArgumentDereferenceable++;
      break;
    case AbstractAttribute::MP_CALL_SITE_ARGUMENT:
      NumCSArgumentDereferenceable++;
      break;
    default:
      break;
    }
    break;
  case Attribute::NoUnwind:
    NumFnNoUnwind++;
    return;
  case Attribute::Returned:
    NumFnArgumentReturned++;
    return;
  case Attribute::NoSync:
    NumFnNoSync++;
    break;
  case Attribute::NoFree:
    NumFnNoFree++;
    break;
  case Attribute::NonNull:
    switch (MP) {
    case AbstractAttribute::MP_RETURNED:
      NumFnReturnedNonNull++;
      break;
    case AbstractAttribute::MP_ARGUMENT:
      NumFnArgumentNonNull++;
      break;
    case AbstractAttribute::MP_CALL_SITE_ARGUMENT:
      NumCSArgumentNonNull++;
      break;
    default:
      break;
    }
    break;
  case Attribute::WillReturn:
    NumFnWillReturn++;
    break;
  case Attribute::NoAlias:
    NumFnArgumentNoAlias++;
    return;
  default:
    return;
  }
}

template <typename StateTy>
using followValueCB_t = std::function<bool(Value *, StateTy &State)>;
template <typename StateTy>
using visitValueCB_t = std::function<void(Value *, StateTy &State)>;

/// Recursively visit all values that might become \p InitV at some point. This
/// will be done by looking through cast instructions, selects, phis, and calls
/// with the "returned" attribute. The callback \p FollowValueCB is asked before
/// a potential origin value is looked at. If no \p FollowValueCB is passed, a
/// default one is used that will make sure we visit every value only once. Once
/// we cannot look through the value any further, the callback \p VisitValueCB
/// is invoked and passed the current value and the \p State. To limit how much
/// effort is invested, we will never visit more than \p MaxValues values.
template <typename StateTy>
static bool genericValueTraversal(
    Value *InitV, StateTy &State, visitValueCB_t<StateTy> &VisitValueCB,
    followValueCB_t<StateTy> *FollowValueCB = nullptr, int MaxValues = 8) {

  SmallPtrSet<Value *, 16> Visited;
  followValueCB_t<bool> DefaultFollowValueCB = [&](Value *Val, bool &) {
    return Visited.insert(Val).second;
  };

  if (!FollowValueCB)
    FollowValueCB = &DefaultFollowValueCB;

  SmallVector<Value *, 16> Worklist;
  Worklist.push_back(InitV);

  int Iteration = 0;
  do {
    Value *V = Worklist.pop_back_val();

    // Check if we should process the current value. To prevent endless
    // recursion keep a record of the values we followed!
    if (!(*FollowValueCB)(V, State))
      continue;

    // Make sure we limit the compile time for complex expressions.
    if (Iteration++ >= MaxValues)
      return false;

    // Explicitly look through calls with a "returned" attribute if we do
    // not have a pointer as stripPointerCasts only works on them.
    if (V->getType()->isPointerTy()) {
      V = V->stripPointerCasts();
    } else {
      CallSite CS(V);
      if (CS && CS.getCalledFunction()) {
        Value *NewV = nullptr;
        for (Argument &Arg : CS.getCalledFunction()->args())
          if (Arg.hasReturnedAttr()) {
            NewV = CS.getArgOperand(Arg.getArgNo());
            break;
          }
        if (NewV) {
          Worklist.push_back(NewV);
          continue;
        }
      }
    }

    // Look through select instructions, visit both potential values.
    if (auto *SI = dyn_cast<SelectInst>(V)) {
      Worklist.push_back(SI->getTrueValue());
      Worklist.push_back(SI->getFalseValue());
      continue;
    }

    // Look through phi nodes, visit all operands.
    if (auto *PHI = dyn_cast<PHINode>(V)) {
      Worklist.append(PHI->op_begin(), PHI->op_end());
      continue;
    }

    // Once a leaf is reached we inform the user through the callback.
    VisitValueCB(V, State);
  } while (!Worklist.empty());

  // All values have been visited.
  return true;
}

/// Return true if \p New is equal or worse than \p Old.
static bool isEqualOrWorse(const Attribute &New, const Attribute &Old) {
  if (!Old.isIntAttribute())
    return true;

  return Old.getValueAsInt() >= New.getValueAsInt();
}

/// Return true if the information provided by \p Attr was added to the
/// attribute list \p Attrs. This is only the case if it was not already present
/// in \p Attrs at the position describe by \p MP and \p ArgNo.
static bool addIfNotExistent(LLVMContext &Ctx, const Attribute &Attr,
                             AttributeList &Attrs,
                             AbstractAttribute::ManifestPosition MP,
                             unsigned ArgNo) {
  unsigned AttrIdx = AbstractAttribute::getAttrIndex(MP, ArgNo);

  if (Attr.isEnumAttribute()) {
    Attribute::AttrKind Kind = Attr.getKindAsEnum();
    if (Attrs.hasAttribute(AttrIdx, Kind))
      if (isEqualOrWorse(Attr, Attrs.getAttribute(AttrIdx, Kind)))
        return false;
    Attrs = Attrs.addAttribute(Ctx, AttrIdx, Attr);
    return true;
  }
  if (Attr.isStringAttribute()) {
    StringRef Kind = Attr.getKindAsString();
    if (Attrs.hasAttribute(AttrIdx, Kind))
      if (isEqualOrWorse(Attr, Attrs.getAttribute(AttrIdx, Kind)))
        return false;
    Attrs = Attrs.addAttribute(Ctx, AttrIdx, Attr);
    return true;
  }
  if (Attr.isIntAttribute()) {
    Attribute::AttrKind Kind = Attr.getKindAsEnum();
    if (Attrs.hasAttribute(AttrIdx, Kind))
      if (isEqualOrWorse(Attr, Attrs.getAttribute(AttrIdx, Kind)))
        return false;
    Attrs = Attrs.removeAttribute(Ctx, AttrIdx, Kind);
    Attrs = Attrs.addAttribute(Ctx, AttrIdx, Attr);
    return true;
  }

  llvm_unreachable("Expected enum or string attribute!");
}

ChangeStatus AbstractAttribute::update(Attributor &A) {
  ChangeStatus HasChanged = ChangeStatus::UNCHANGED;
  if (getState().isAtFixpoint())
    return HasChanged;

  LLVM_DEBUG(dbgs() << "[Attributor] Update: " << *this << "\n");

  HasChanged = updateImpl(A);

  LLVM_DEBUG(dbgs() << "[Attributor] Update " << HasChanged << " " << *this
                    << "\n");

  return HasChanged;
}

ChangeStatus AbstractAttribute::manifest(Attributor &A) {
  assert(getState().isValidState() &&
         "Attempted to manifest an invalid state!");
  assert(getAssociatedValue() &&
         "Attempted to manifest an attribute without associated value!");

  ChangeStatus HasChanged = ChangeStatus::UNCHANGED;
  SmallVector<Attribute, 4> DeducedAttrs;
  getDeducedAttributes(DeducedAttrs);

  Function &ScopeFn = getAnchorScope();
  LLVMContext &Ctx = ScopeFn.getContext();
  ManifestPosition MP = getManifestPosition();

  AttributeList Attrs;
  SmallVector<unsigned, 4> ArgNos;

  // In the following some generic code that will manifest attributes in
  // DeducedAttrs if they improve the current IR. Due to the different
  // annotation positions we use the underlying AttributeList interface.
  // Note that MP_CALL_SITE_ARGUMENT can annotate multiple locations.

  switch (MP) {
  case MP_ARGUMENT:
    ArgNos.push_back(cast<Argument>(getAssociatedValue())->getArgNo());
    Attrs = ScopeFn.getAttributes();
    break;
  case MP_FUNCTION:
  case MP_RETURNED:
    ArgNos.push_back(0);
    Attrs = ScopeFn.getAttributes();
    break;
  case MP_CALL_SITE_ARGUMENT: {
    CallSite CS(&getAnchoredValue());
    for (unsigned u = 0, e = CS.getNumArgOperands(); u != e; u++)
      if (CS.getArgOperand(u) == getAssociatedValue())
        ArgNos.push_back(u);
    Attrs = CS.getAttributes();
  }
  }

  for (const Attribute &Attr : DeducedAttrs) {
    for (unsigned ArgNo : ArgNos) {
      if (!addIfNotExistent(Ctx, Attr, Attrs, MP, ArgNo))
        continue;

      HasChanged = ChangeStatus::CHANGED;
      bookkeeping(MP, Attr);
    }
  }

  if (HasChanged == ChangeStatus::UNCHANGED)
    return HasChanged;

  switch (MP) {
  case MP_ARGUMENT:
  case MP_FUNCTION:
  case MP_RETURNED:
    ScopeFn.setAttributes(Attrs);
    break;
  case MP_CALL_SITE_ARGUMENT:
    CallSite(&getAnchoredValue()).setAttributes(Attrs);
  }

  return HasChanged;
}

Function &AbstractAttribute::getAnchorScope() {
  Value &V = getAnchoredValue();
  if (isa<Function>(V))
    return cast<Function>(V);
  if (isa<Argument>(V))
    return *cast<Argument>(V).getParent();
  if (isa<Instruction>(V))
    return *cast<Instruction>(V).getFunction();
  llvm_unreachable("No scope for anchored value found!");
}

const Function &AbstractAttribute::getAnchorScope() const {
  return const_cast<AbstractAttribute *>(this)->getAnchorScope();
}

/// Helper to identify the correct offset into an attribute list.
unsigned AbstractAttribute::getAttrIndex(ManifestPosition MP, int ArgNo) {
  switch (MP) {
  case AbstractAttribute::MP_ARGUMENT:
  case AbstractAttribute::MP_CALL_SITE_ARGUMENT:
    assert(ArgNo >= 0 && "Expected non-negative argument number");
    return ArgNo + AttributeList::FirstArgIndex;
  case AbstractAttribute::MP_FUNCTION:
    return AttributeList::FunctionIndex;
  case AbstractAttribute::MP_RETURNED:
    return AttributeList::ReturnIndex;
  }
  llvm_unreachable("Unknown manifest position!");
}

void AbstractAttribute::getAttrIndices(SmallVectorImpl<unsigned> &AttrIndices,
                                       const Value &V, int ArgNo) {
  if (ArgNo >= 0) {
    AttrIndices.push_back(getAttrIndex(MP_ARGUMENT, ArgNo));
  } else {
    AttrIndices.push_back(getAttrIndex(MP_FUNCTION));
    AttrIndices.push_back(getAttrIndex(MP_RETURNED));
  }
}

/// -----------------------NoUnwind Function Attribute--------------------------

struct AANoUnwindFunction : StatefulAbstractAttribute<AANoUnwind, BooleanState> {

  AANoUnwindFunction(Function &F, InformationCache &InfoCache)
      : StatefulAbstractAttribute(&F, F, -2, InfoCache) {}

  void initialize(Attributor &A) override {
    StateType::addKnownFrom(StateType::template getFromIR<AANoUnwindFunction>(
        getAnchorScope(), getAnchoredValue(), getArgNo()));
  }

  const std::string getAsStr() const override {
    return getAssumed() ? "nounwind" : "may-unwind";
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  /// See AANoUnwind::isAssumedNoUnwind().
  bool isAssumedNoUnwind() const override { return getAssumed(); }

  /// See AANoUnwind::isKnownNoUnwind().
  bool isKnownNoUnwind() const override { return getKnown(); }
};

ChangeStatus AANoUnwindFunction::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();

  // The map from instruction opcodes to those instructions in the function.
  auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(F);
  auto Opcodes = {
      (unsigned)Instruction::Invoke,      (unsigned)Instruction::CallBr,
      (unsigned)Instruction::Call,        (unsigned)Instruction::CleanupRet,
      (unsigned)Instruction::CatchSwitch, (unsigned)Instruction::Resume};

  bool AnyAssumed = false;
  for (unsigned Opcode : Opcodes) {
    for (Instruction *I : OpcodeInstMap[Opcode]) {
      if (!I->mayThrow())
        continue;

      if (!getAssumedOrKnown<AANoUnwind>(A, *this, AnyAssumed, *I, -2))
        return indicatePessimisticFixpoint();
    }
  }

  if (!AnyAssumed)
    indicateOptimisticFixpoint();

  return ChangeStatus::UNCHANGED;
}

/// --------------------- Function Return Values -------------------------------

struct ReturnedValuesStates : public AbstractState,
                              AttributeCompatibleAbstractState {

  /// Mapping of values potentially returned by the associated function to the
  /// return instructions that might return them.
  DenseMap<const Value *, SmallPtrSet<ReturnInst *, 2>> ReturnedValues;

  /// State flags
  ///
  ///{
  bool IsFixed;
  bool IsValidState;
  bool HasOverdefinedReturnedCalls;
  ///}

  /// See AbstractState::isAtFixpoint().
  bool isAtFixpoint() const override { return IsFixed; }

  /// See AbstractState::isValidState().
  bool isValidState() const override { return IsValidState; }

  /// See AbstractState::indicateOptimisticFixpoint(...).
  ChangeStatus indicateOptimisticFixpoint() override {
    IsFixed = true;
    IsValidState &= true;
    return ChangeStatus::UNCHANGED;
  }
  ChangeStatus indicatePessimisticFixpoint() override {
    IsFixed = true;
    IsValidState = false;
    return ChangeStatus::CHANGED;
  }

  static const Argument *getKnownReturnedArg(const Function &F) {
    for (const Argument &Arg : F.args())
      if (Arg.hasReturnedAttr())
        return &Arg;
    return nullptr;
  }

  template <typename AAType>
  static Attribute getFromIR(const Function &Scope, const Value &V, int ArgNo) {
    assert(&Scope == &V);
    if (const Argument *Arg = getKnownReturnedArg(Scope))
      return Scope.getAttribute(Arg->getArgNo(), Attribute::Returned);
    return Attribute();
  }

  /// The best state is any "returned" argument.
  static bool isBestState(const Attribute &Attr) {
    return Attr.getKindAsEnum() == Attribute::Returned;
  }
};

/// "Attribute" that collects all potential returned values and the return
/// instructions that they arise from.
///
/// If there is a unique returned value R, the manifest method will:
///   - mark R with the "returned" attribute, if R is an argument.
class AAReturnedValuesImpl final
    : public StatefulAbstractAttribute<AAReturnedValues, ReturnedValuesStates> {

  /// Collect values that could become \p V in the set \p Values, each mapped to
  /// \p ReturnInsts.
  void collectValuesRecursively(
      Attributor &A, const Value *V, SmallPtrSetImpl<ReturnInst *> &ReturnInsts,
      DenseMap<const Value *, SmallPtrSet<ReturnInst *, 2>> &Values) {

    auto VisitValueCB = [&](const Value *Val, bool &) {
      assert(!isa<Instruction>(Val) ||
             &getAnchorScope() == cast<Instruction>(Val)->getFunction());
      Values[Val].insert(ReturnInsts.begin(), ReturnInsts.end());
      return true;
    };

    bool UnusedBool;
    bool Success = genericValueTraversal(V, UnusedBool, VisitValueCB);

    // If we did abort the above traversal we haven't see all the values.
    // Consequently, we cannot know if the information we would derive is
    // accurate so we give up early.
    if (!Success)
      indicatePessimisticFixpoint();
  }

public:
  /// See AbstractAttribute::AbstractAttribute(...).
  AAReturnedValuesImpl(Function &F, InformationCache &InfoCache)
      : StatefulAbstractAttribute(/* AssociatedVal */ nullptr, F, MP_FUNCTION,
                                  InfoCache) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // Reset the state.
    AssociatedVal = nullptr;
    IsFixed = false;
    IsValidState = true;
    HasOverdefinedReturnedCalls = false;
    ReturnedValues.clear();

    Function &F = cast<Function>(getAnchoredValue());

    // Look through all arguments, if one is marked as returned we are done.
    for (Argument &Arg : F.args()) {
      if (Arg.hasReturnedAttr()) {
        addKnownFrom(Arg);
        return;
      }
    }

    // The map from instruction opcodes to those instructions in the function.
    auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(F);

    // If no argument was marked as returned we look at all return instructions
    // and collect potentially returned values.
    for (Instruction *RI : OpcodeInstMap[Instruction::Ret]) {
      SmallPtrSet<ReturnInst *, 1> RISet({cast<ReturnInst>(RI)});
      collectValuesRecursively(A, cast<ReturnInst>(RI)->getReturnValue(), RISet,
                               ReturnedValues);
    }
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override;

  /// See AbstractAttribute::updateImpl(Attributor &A).
  ChangeStatus updateImpl(Attributor &A) override;

  /// Return the number of potential return values, -1 if unknown.
  size_t getNumReturnValues() const {
    return isValidState() ? ReturnedValues.size() : -1;
  }

  /// Provided via AbstractStateCompatibleWith<..., Attribute>.
  ChangeStatus addKnownFrom(const Attribute &Attr) override {
    if (isAtFixpoint())
      return ChangeStatus::UNCHANGED;
    assert(Attr.getKindAsEnum() == Attribute::Returned);
    const Argument *Arg = getKnownReturnedArg(getAnchorScope());
    assert(Arg);
    addKnownFrom(*Arg);
    return ChangeStatus::CHANGED;
  }

  void addKnownFrom(const Argument &Arg) {
    Function &F = getAnchorScope();
    // The map from instruction opcodes to those instructions in the function.
    auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(F);

    auto &ReturnInstSet = ReturnedValues[&Arg];
    for (Instruction *RI : OpcodeInstMap[Instruction::Ret])
      ReturnInstSet.insert(cast<ReturnInst>(RI));

    indicateOptimisticFixpoint();
  }

  /// Return an assumed unique return value if a single candidate is found. If
  /// there cannot be one, return a nullptr. If it is not clear yet, return the
  /// Optional::NoneType.
  Optional<const Value *> getAssumedUniqueReturnValue(Attributor &A) const;

  #if 0
  /// See AbstractState::checkForallReturnedValues(...).
  bool
  checkForallReturnedValues(Attributor &A,
                            ReturnValuePredicateFuncTy &Pred) const override;
  #endif
  /// See AbstractState::checkForallReturnedValues(...).
  bool
  checkForallReturnedValues(std::function<bool(const Value &)> &Pred) const override;

  /// Pretty print the attribute similar to the IR representation.
  const std::string getAsStr() const override;
};

ChangeStatus AAReturnedValuesImpl::manifest(Attributor &A) {
  ChangeStatus Changed = ChangeStatus::UNCHANGED;

  // Bookkeeping.
  assert(isValidState());
  NumFnKnownReturns++;

  // Check if we have an assumed unique return value that we could manifest.
  Optional<const Value *> UniqueRV = getAssumedUniqueReturnValue(A);

  if (!UniqueRV.hasValue() || !UniqueRV.getValue())
    return Changed;

  // Bookkeeping.
  NumFnUniqueReturned++;

  // If the assumed unique return value is an argument, annotate it.
  if (auto *UniqueRVArg = dyn_cast<Argument>(UniqueRV.getValue())) {
    AttrIdx = UniqueRVArg->getArgNo();
    AssociatedVal = const_cast<Value *>(cast<Value>(UniqueRVArg));
    Changed = AbstractAttribute::manifest(A) | Changed;
  }

  return Changed;
}

const std::string AAReturnedValuesImpl::getAsStr() const {
  return (isAtFixpoint() ? "returns(#" : "may-return(#") +
         (isValidState() ? std::to_string(getNumReturnValues()) : "?") + ")";
}

Optional<const Value *>
AAReturnedValuesImpl::getAssumedUniqueReturnValue(Attributor &A) const {
  // If checkForallReturnedValues provides a unique value, ignoring potential
  // undef values that can also be present, it is assumed to be the actual
  // return value and forwarded to the caller of this method. If there are
  // multiple, a nullptr is returned indicating there cannot be a unique
  // returned value.
  Optional<const Value *> UniqueRV;

  //ReturnValuePredicateFuncTy Pred =
      //[&](Attributor &, const Value &RV,
          //const SmallPtrSetImpl<ReturnInst *> &) -> bool {
  std::function<bool(const Value &)> Pred = [&](const Value &RV) -> bool {
    // If we found a second returned value and neither the current nor the saved
    // one is an undef, there is no unique returned value. Undefs are special
    // since we can pretend they have any value.
    if (UniqueRV.hasValue() && UniqueRV != &RV &&
        !(isa<UndefValue>(RV) || isa<UndefValue>(UniqueRV.getValue()))) {
      UniqueRV = nullptr;
      return false;
    }

    // Do not overwrite a value with an undef.
    if (!UniqueRV.hasValue() || !isa<UndefValue>(RV))
      UniqueRV = &RV;

    return true;
  };

  //if (!checkForallReturnedValues(A, Pred))
  if (!checkForallReturnedValues(Pred))
    UniqueRV = nullptr;

  return UniqueRV;
}

//bool AAReturnedValuesImpl::checkForallReturnedValues(
    //Attributor &A, ReturnValuePredicateFuncTy &Pred) const {
bool AAReturnedValuesImpl::checkForallReturnedValues(
    std::function<bool(const Value &)> &Pred) const {
  if (!isValidState())
    return false;

  // Check all returned values but ignore call sites as long as we have not
  // encountered an overdefined one during an update.
  for (auto &It : ReturnedValues) {
    const Value *RV = It.first;

    ImmutableCallSite ICS(RV);
    if (ICS && !HasOverdefinedReturnedCalls)
      continue;

    //if (!Pred(A, *RV, It.second))
    if (!Pred(*RV))
      return false;
  }

  return true;
}

ChangeStatus AAReturnedValuesImpl::updateImpl(Attributor &A) {

  // Check if we know of any values returned by the associated function,
  // if not, we are done.
  if (getNumReturnValues() == 0) {
    indicateOptimisticFixpoint();
    return ChangeStatus::UNCHANGED;
  }

  // Check if any of the returned values is a call site we can refine.
  decltype(ReturnedValues) AddRVs;
  bool HasCallSite = false;

  // Look at all returned call sites.
  for (auto &It : ReturnedValues) {
    SmallPtrSet<ReturnInst *, 2> &ReturnInsts = It.second;
    const Value *RV = It.first;
    LLVM_DEBUG(dbgs() << "[AAReturnedValues] Potentially returned value " << *RV
                      << "\n");

    // Only call sites can change during an update, ignore the rest.
    ImmutableCallSite RetCS(RV);
    if (!RetCS)
      continue;

    // For now, any call site we see will prevent us from directly fixing the
    // state. However, if the information on the callees is fixed, the call
    // sites will be removed and we will fix the information for this state.
    HasCallSite = true;

    if (!RetCS.getCalledFunction()) {
      HasOverdefinedReturnedCalls = true;
      LLVM_DEBUG(dbgs() << "[AAReturnedValues] Returned call site (" << *RV
                        << ") without called function\n");
      continue;
    }

    // Try to find a assumed unique return value for the called function.
    auto *RetCSAA =
        A.getAAFor<AAReturnedValuesImpl>(*this, *RetCS.getCalledFunction());
    if (!RetCSAA) {
      HasOverdefinedReturnedCalls = true;
      LLVM_DEBUG(dbgs() << "[AAReturnedValues] Returned call site (" << *RV
                        << ") with " << (RetCSAA ? "invalid" : "no")
                        << " associated state\n");
      continue;
    }

    // Try to find a assumed unique return value for the called function.
    Optional<const Value *> AssumedUniqueRV = RetCSAA->getAssumedUniqueReturnValue(A);

    // If no assumed unique return value was found due to the lack of
    // candidates, we may need to resolve more calls (through more update
    // iterations) or the called function will not return. Either way, we simply
    // stick with the call sites as return values. Because there were not
    // multiple possibilities, we do not treat it as overdefined.
    if (!AssumedUniqueRV.hasValue())
      continue;

    // If multiple, non-refinable values were found, there cannot be a unique
    // return value for the called function. The returned call is overdefined!
    if (!AssumedUniqueRV.getValue()) {
      HasOverdefinedReturnedCalls = true;
      LLVM_DEBUG(dbgs() << "[AAReturnedValues] Returned call site has multiple "
                           "potentially returned values\n");
      continue;
    }

    LLVM_DEBUG({
      bool UniqueRVIsKnown = RetCSAA->isAtFixpoint();
      dbgs() << "[AAReturnedValues] Returned call site "
             << (UniqueRVIsKnown ? "known" : "assumed")
             << " unique return value: " << *AssumedUniqueRV << "\n";
    });

    // The assumed unique return value.
    const Value *AssumedRetVal = AssumedUniqueRV.getValue();

    // If the assumed unique return value is an argument, lookup the matching
    // call site operand and recursively collect new returned values.
    // If it is not an argument, it is just put into the set of returned values
    // as we would have already looked through casts, phis, and similar values.
    if (const Argument *AssumedRetArg = dyn_cast<Argument>(AssumedRetVal))
      collectValuesRecursively(A,
                               RetCS.getArgOperand(AssumedRetArg->getArgNo()),
                               ReturnInsts, AddRVs);
    else
      AddRVs[AssumedRetVal].insert(ReturnInsts.begin(), ReturnInsts.end());
  }

  // Keep track of any change to trigger updates on dependent attributes.
  ChangeStatus Changed = ChangeStatus::UNCHANGED;

  for (auto &It : AddRVs) {
    assert(!It.second.empty() && "Entry does not add anything.");
    auto &ReturnInsts = ReturnedValues[It.first];
    for (ReturnInst *RI : It.second)
      if (ReturnInsts.insert(RI).second) {
        LLVM_DEBUG(dbgs() << "[AAReturnedValues] Add new returned value "
                          << *It.first << " => " << *RI << "\n");
        Changed = ChangeStatus::CHANGED;
      }
  }

  // If there is no call site in the returned values we are done.
  if (!HasCallSite) {
    indicateOptimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  return Changed;
}

/// ------------------------ NoSync Function Attribute -------------------------

struct AANoSyncFunction
    : public StatefulAbstractAttribute<AANoSync, BooleanState> {

  AANoSyncFunction(Function &F, InformationCache &InfoCache)
      : StatefulAbstractAttribute(&F, F, MP_FUNCTION, InfoCache) {}

  void initialize(Attributor &A) override {
    StateType::addKnownFrom(StateType::template getFromIR<AANoSyncFunction>(
        getAnchorScope(), getAnchoredValue(), getArgNo()));
  }

  const std::string getAsStr() const override {
    return getAssumed() ? "nosync" : "may-sync";
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  /// See AANoSync::isAssumedNoSync()
  bool isAssumedNoSync() const override { return getAssumed(); }

  /// See AANoSync::isKnownNoSync()
  bool isKnownNoSync() const override { return getKnown(); }

  /// Helper function used to determine whether an instruction is non-relaxed
  /// atomic. In other words, if an atomic instruction does not have unordered
  /// or monotonic ordering
  static bool isNonRelaxedAtomic(const Instruction *I);

  /// Helper function used to determine whether an instruction is volatile.
  static bool isVolatile(const Instruction *I);

  /// Helper function uset to check if intrinsic is volatile (memcpy, memmove,
  /// memset).
  static bool isNoSyncIntrinsic(const Instruction *I);
};

#if 0
/// TODO
template<typename AAType>
uint32_t AANoSync::getFromIR(const Function &F, const Value &V, int ArgNo) {
  if (ArgNo == -2) {
    if (const Instruction *I = dyn_cast<Instruction>(&V)) {
      if (ImmutableCallSite(I)) {
        if (isa<IntrinsicInst>(I) && AANoSyncFunction::isNoSyncIntrinsic(I))
          return /* nosync */ true;
      } else {
        if (!AANoSyncFunction::isVolatile(I) &&
            !AANoSyncFunction::isNonRelaxedAtomic(I))
          return /* nosync */ true;
      }
    }
  }
  return AbstractAttribute::getFromIR<AAType>(F, V, ArgNo);
}
#endif

bool AANoSyncFunction::isNonRelaxedAtomic(const Instruction *I) {
  if (!I->isAtomic())
    return false;

  AtomicOrdering Ordering;
  switch (I->getOpcode()) {
  case Instruction::AtomicRMW:
    Ordering = cast<AtomicRMWInst>(I)->getOrdering();
    break;
  case Instruction::Store:
    Ordering = cast<StoreInst>(I)->getOrdering();
    break;
  case Instruction::Load:
    Ordering = cast<LoadInst>(I)->getOrdering();
    break;
  case Instruction::Fence: {
    auto *FI = cast<FenceInst>(I);
    if (FI->getSyncScopeID() == SyncScope::SingleThread)
      return false;
    Ordering = FI->getOrdering();
    break;
  }
  case Instruction::AtomicCmpXchg: {
    AtomicOrdering Success = cast<AtomicCmpXchgInst>(I)->getSuccessOrdering();
    AtomicOrdering Failure = cast<AtomicCmpXchgInst>(I)->getFailureOrdering();
    // Only if both are relaxed, than it can be treated as relaxed.
    // Otherwise it is non-relaxed.
    if (Success != AtomicOrdering::Unordered &&
        Success != AtomicOrdering::Monotonic)
      return true;
    if (Failure != AtomicOrdering::Unordered &&
        Failure != AtomicOrdering::Monotonic)
      return true;
    return false;
  }
  default:
    llvm_unreachable(
        "New atomic operations need to be known in the attributor.");
  }

  // Relaxed.
  if (Ordering == AtomicOrdering::Unordered ||
      Ordering == AtomicOrdering::Monotonic)
    return false;
  return true;
}

/// Checks if an intrinsic is nosync. Currently only checks mem* intrinsics.
/// FIXME: We should ipmrove the handling of intrinsics.
bool AANoSyncFunction::isNoSyncIntrinsic(const Instruction *I) {
  if (auto *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    /// Element wise atomic memory intrinsics are can only be unordered,
    /// therefore nosync.
    case Intrinsic::memset_element_unordered_atomic:
    case Intrinsic::memmove_element_unordered_atomic:
    case Intrinsic::memcpy_element_unordered_atomic:
      return true;
    case Intrinsic::memset:
    case Intrinsic::memmove:
    case Intrinsic::memcpy:
      if (!cast<MemIntrinsic>(II)->isVolatile())
        return true;
      return false;
    default:
      return false;
    }
  }
  return false;
}

bool AANoSyncFunction::isVolatile(const Instruction *I) {
  assert(!ImmutableCallSite(I) && !isa<CallBase>(I) &&
         "Calls should not be checked here");

  switch (I->getOpcode()) {
  case Instruction::AtomicRMW:
    return cast<AtomicRMWInst>(I)->isVolatile();
  case Instruction::Store:
    return cast<StoreInst>(I)->isVolatile();
  case Instruction::Load:
    return cast<LoadInst>(I)->isVolatile();
  case Instruction::AtomicCmpXchg:
    return cast<AtomicCmpXchgInst>(I)->isVolatile();
  default:
    return false;
  }
}

ChangeStatus AANoSyncFunction::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();

  bool AnyAssumed = false;

  /// We are looking for volatile instructions or Non-Relaxed atomics.
  /// FIXME: We should ipmrove the handling of intrinsics.
  for (Instruction *I : InfoCache.getReadOrWriteInstsForFunction(F))
    ;
    //if (!getAssumedOrKnown<AANoSyncFunction>(A, *this, AnyAssumed, *I, -2))
      //return indicatePessimisticFixpoint();

  auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(F);
  auto Opcodes = {(unsigned)Instruction::Invoke, (unsigned)Instruction::CallBr,
                  (unsigned)Instruction::Call};

  for (unsigned Opcode : Opcodes) {
    for (Instruction *I : OpcodeInstMap[Opcode]) {
      // At this point we handled all read/write effects and they are all
      // nosync, so they can be skipped.
      if (I->mayReadOrWriteMemory())
        continue;

      ImmutableCallSite ICS(I);

      // non-convergent and readnone imply nosync.
      if (!ICS.isConvergent())
        continue;

      return indicatePessimisticFixpoint();
    }
  }

  if (!AnyAssumed)
    indicateOptimisticFixpoint();

  return ChangeStatus::UNCHANGED;
}

/// ------------------------ No-Free Attributes ----------------------------

struct AANoFreeFunction : StatefulAbstractAttribute<AbstractAttribute, BooleanState> {

  /// See AbstractAttribute::AbstractAttribute(...).
  AANoFreeFunction(Function &F, InformationCache &InfoCache)
      : StatefulAbstractAttribute(&F, F, MP_FUNCTION, InfoCache) {}

  void initialize(Attributor &A) override {
    StateType::addKnownFrom(StateType::template getFromIR<AANoFreeFunction>(
        getAnchorScope(), getAnchoredValue(), getArgNo()));
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return getAssumed() ? "nofree" : "may-free";
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  /// See AbstractAttribute::getAttrKind().
  Attribute::AttrKind getAttrKind() const override { return ID; }

  /// Return true if "nofree" is assumed.
  bool isAssumedNoFree() const { return getAssumed(); }

  /// Return true if "nofree" is known.
  bool isKnownNoFree() const { return getKnown(); }

  /// The identifier used by the Attributor for this class of attributes.
  static constexpr Attribute::AttrKind ID = Attribute::NoFree;
};

ChangeStatus AANoFreeFunction::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();

  // The map from instruction opcodes to those instructions in the function.
  auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(F);

  bool AnyAssumed = false;
  for (unsigned Opcode :
       {(unsigned)Instruction::Invoke, (unsigned)Instruction::CallBr,
        (unsigned)Instruction::Call}) {
    for (Instruction *I : OpcodeInstMap[Opcode])
      ;
      //if (!getAssumedOrKnown<AANoFreeFunction>(A, *this, AnyAssumed, *I, -2))
        //return indicatePessimisticFixpoint();
  }

  if (!AnyAssumed)
    indicateOptimisticFixpoint();

  return ChangeStatus::UNCHANGED;
}

/// ------------------------ NonNull Argument Attribute ------------------------
struct AANonNullImpl : StatefulAbstractAttribute<AANonNull, BooleanState> {

  AANonNullImpl(Value *AssociatedVal, Value &AnchoredValue, int AttrIdx,
                InformationCache &InfoCache)
      : StatefulAbstractAttribute(AssociatedVal, AnchoredValue, AttrIdx,
                                  InfoCache) {}

  void initialize(Attributor &A) override {
    StateType::addKnownFrom(StateType::template getFromIR<AANonNullImpl>(
        getAnchorScope(), getAnchoredValue(), getArgNo()));
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return getAssumed() ? "nonnull" : "may-null";
  }

  /// See AANonNull::isAssumedNonNull().
  bool isAssumedNonNull() const override { return getAssumed(); }

  /// See AANonNull::isKnownNonNull().
  bool isKnownNonNull() const override { return getKnown(); }
};

  #if 0
/// TODO
template<typename AAType>
Attribute AANonNull::getFromIR(const Function &F, ManifestPosition MP, const Value &V,
                                   int ArgNo) {
  assert(0);
  ImmutableCallSite ICS = ImmutableCallSite(&V);
  if (ArgNo >= 0 && ICS) {
    if (isKnownNonZero(ICS.getArgOperand(ArgNo),
                       F.getParent()->getDataLayout()))
      return 1;
  } else if (isKnownNonZero(&V, F.getParent()->getDataLayout())) {
      return 1;
  }
  return AbstractAttribute::getFromIR<AAType>(F, V, ArgNo);
}
#endif

/// NonNull attribute for function return value.
struct AANonNullReturned : AANonNullImpl {

  AANonNullReturned(Function &F, InformationCache &InfoCache)
      : AANonNullImpl(&F, F, MP_RETURNED, InfoCache) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;
};

ChangeStatus AANonNullReturned::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();

  bool AnyAssumed = false;
  //if (!A.checkForallReturnedValues<AANonNull>(*this, F, AnyAssumed))
    //return indicatePessimisticFixpoint();

  if (!AnyAssumed)
    indicateOptimisticFixpoint();

  return ChangeStatus::UNCHANGED;
}

/// NonNull attribute for function argument.
struct AANonNullArgument : AANonNullImpl {

  AANonNullArgument(Argument &A, InformationCache &InfoCache)
      : AANonNullImpl(&A, A, A.getArgNo(), InfoCache) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;
};

/// NonNull attribute for a call site argument.
struct AANonNullCallSiteArgument : AANonNullImpl {

  /// See AANonNullImpl::AANonNullImpl(...).
  AANonNullCallSiteArgument(Instruction &I, InformationCache &InfoCache,
                            unsigned ArgNo)
      : AANonNullImpl(CallSite(&I).getArgOperand(ArgNo), I, ArgNo, InfoCache) {}

  /// See AbstractAttribute::updateImpl(Attributor &A).
  ChangeStatus updateImpl(Attributor &A) override;
};

ChangeStatus AANonNullArgument::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();
  Argument &Arg = cast<Argument>(getAnchoredValue());

  bool AnyAssumed = false;
  unsigned ArgNo = Arg.getArgNo();

  if (!A.checkForAllCallSites<AANonNull>(*this, F, AnyAssumed, ArgNo, true))
    return indicatePessimisticFixpoint();

  if (!AnyAssumed)
    indicateOptimisticFixpoint();

  return ChangeStatus::UNCHANGED;
}

ChangeStatus AANonNullCallSiteArgument::updateImpl(Attributor &A) {
  bool Assumed = false;
  //if (!getAssumedOrKnown<AANonNullImpl>(A, *this, Assumed,
                                          //*getAssociatedValue(), -1))
    //return indicatePessimisticFixpoint();

  if (!Assumed)
    indicateOptimisticFixpoint();

  return ChangeStatus::UNCHANGED;
}

/// ------------------------ Will-Return Attributes ----------------------------

struct AAWillReturnFunction : public StatefulAbstractAttribute<AAWillReturn, BooleanState> {

  /// See AbstractAttribute::AbstractAttribute(...).
  AAWillReturnFunction(Function &F, InformationCache &InfoCache)
      : StatefulAbstractAttribute(&F, F, MP_FUNCTION, InfoCache) {}

  /// See AAWillReturn::isKnownWillReturn().
  bool isKnownWillReturn() const override { return getKnown(); }

  /// See AAWillReturn::isAssumedWillReturn().
  bool isAssumedWillReturn() const override { return getAssumed(); }

  /// See AbstractAttribute::getAsStr()
  const std::string getAsStr() const override {
    return getAssumed() ? "willreturn" : "may-noreturn";
  }

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override;

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;
};

// Helper function that checks whether a function has any cycle.
// TODO: Replace with more efficent code
bool containsCycle(Function &F) {
  SmallPtrSet<BasicBlock *, 32> Visited;

  // Traverse BB by dfs and check whether successor is already visited.
  for (BasicBlock *BB : depth_first(&F)) {
    Visited.insert(BB);
    for (auto *SuccBB : successors(BB)) {
      if (Visited.count(SuccBB))
        return true;
    }
  }
  return false;
}

// Helper function that checks the function have a loop which might become an
// endless loop
// FIXME: Any cycle is regarded as endless loop for now.
//        We have to allow some patterns.
bool containsPossiblyEndlessLoop(Function &F) { return containsCycle(F); }

void AAWillReturnFunction::initialize(Attributor &A) {
  StateType::addKnownFrom(StateType::template getFromIR<AAWillReturnFunction>(
      getAnchorScope(), getAnchoredValue(), getArgNo()));

  Function &F = getAnchorScope();

  if (containsPossiblyEndlessLoop(F))
    indicatePessimisticFixpoint();
}

ChangeStatus AAWillReturnFunction::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();

  // The map from instruction opcodes to those instructions in the function.
  auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(F);

  bool AnyAssumed = false;
  for (unsigned Opcode :
       {(unsigned)Instruction::Invoke, (unsigned)Instruction::CallBr,
        (unsigned)Instruction::Call}) {
    for (Instruction *I : OpcodeInstMap[Opcode]) {

      bool Assumed = false;
      //if (getAssumedOrKnown<AAWillReturnFunction>(A, *this, Assumed, *I, -2)) {
        //if (!Assumed)
          //continue;

        //AnyAssumed = true;
        //if (getAssumedOrKnown<AANoRecurse>(A, *this, AnyAssumed, *I, -2))
          //continue;
      //}

      return indicatePessimisticFixpoint();
    }
  }

  if (!AnyAssumed)
    indicateOptimisticFixpoint();

  return ChangeStatus::UNCHANGED;
}

/// ------------------------ NoAlias Argument Attribute ------------------------

struct AANoAliasImpl : StatefulAbstractAttribute<AANoAlias, BooleanState> {

  AANoAliasImpl(Value *AssociatedVal, Value &AnchoredValue, int AttrIdx,
                InformationCache &InfoCache)
      : StatefulAbstractAttribute(AssociatedVal, AnchoredValue, AttrIdx,
                                  InfoCache) {}

  void initialize(Attributor &A) override {
    StateType::addKnownFrom(StateType::template getFromIR<AANoAliasImpl>(
        getAnchorScope(), getAnchoredValue(), getArgNo()));
  }

  const std::string getAsStr() const override {
    return getAssumed() ? "noalias" : "may-alias";
  }

  /// See AANoAlias::isAssumedNoAlias().
  bool isAssumedNoAlias() const override { return getAssumed(); }

  /// See AANoAlias::isKnowndNoAlias().
  bool isKnownNoAlias() const override { return getKnown(); }
};

/// NoAlias attribute for function return value.
struct AANoAliasReturned : AANoAliasImpl {

  AANoAliasReturned(Function &F, InformationCache &InfoCache)
      : AANoAliasImpl(&F, F, MP_RETURNED, InfoCache) {}

  /// See AbstractAttribute::updateImpl(...).
  virtual ChangeStatus updateImpl(Attributor &A) override;
};

ChangeStatus AANoAliasReturned::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();

  auto *AARetValImpl = A.getAAFor<AAReturnedValuesImpl>(*this, F);
  if (!AARetValImpl) {
    indicatePessimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  std::function<bool(Value &)> Pred = [&](Value &RV) -> bool {
    if (Constant *C = dyn_cast<Constant>(&RV))
      if (C->isNullValue() || isa<UndefValue>(C))
        return true;

    /// For now, we can only deduce noalias if we have call sites.
    /// FIXME: add more support.
    ImmutableCallSite ICS(&RV);
    if (!ICS)
      return false;

    auto *NoAliasAA = A.getAAFor<AANoAlias>(*this, RV);

    if (!ICS.returnDoesNotAlias() &&
        (!NoAliasAA || !NoAliasAA->isAssumedNoAlias()))
      return false;

    /// FIXME: We can improve capture check in two ways:
    /// 1. Use the AANoCapture facilities.
    /// 2. Use the location of return insts for escape queries.
    if (PointerMayBeCaptured(&RV, /* ReturnCaptures */ false,
                             /* StoreCaptures */ true))
      return false;

    return true;
  };

  //if (!AARetValImpl->checkForallReturnedValues(Pred)) {
    //indicatePessimisticFixpoint();
    //return ChangeStatus::CHANGED;
  //}

  return ChangeStatus::UNCHANGED;
}

/// -------------------AAIsDead Function Attribute-----------------------

struct AAIsDeadFunction : StatefulAbstractAttribute<AAIsDead, BooleanState> {

  AAIsDeadFunction(Function &F, InformationCache &InfoCache)
      : StatefulAbstractAttribute(&F, F, MP_FUNCTION, InfoCache) {}

  void initialize(Attributor &A) override {
    Function &F = getAnchorScope();

    ToBeExploredPaths.insert(&(F.getEntryBlock().front()));
    AssumedLiveBlocks.insert(&(F.getEntryBlock()));
    for (size_t i = 0; i < ToBeExploredPaths.size(); ++i)
      explorePath(A, ToBeExploredPaths[i]);
  }

  /// Explores new instructions starting from \p I. If instruction is dead, stop
  /// and return true if it discovered a new instruction.
  bool explorePath(Attributor &A, Instruction *I);

  const std::string getAsStr() const override {
    return "LiveBBs(" + std::to_string(AssumedLiveBlocks.size()) + "/" +
           std::to_string(getAnchorScope().size()) + ")";
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    assert(getState().isValidState() &&
           "Attempted to manifest an invalid state!");

    ChangeStatus HasChanged = ChangeStatus::UNCHANGED;

    for (Instruction *I : NoReturnCalls) {
      BasicBlock *BB = I->getParent();

      /// Invoke is replaced with a call and unreachable is placed after it.
      if (auto *II = dyn_cast<InvokeInst>(I)) {
        changeToCall(II);
        changeToUnreachable(BB->getTerminator(), /* UseLLVMTrap */ false);
        LLVM_DEBUG(dbgs() << "[AAIsDead] Replaced invoke with call inst\n");
        continue;
      }

      SplitBlock(BB, I->getNextNode());
      changeToUnreachable(BB->getTerminator(), /* UseLLVMTrap */ false);
      HasChanged = ChangeStatus::CHANGED;
    }

    return HasChanged;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  /// See AAIsDead::isAssumedDead().
  bool isAssumedDead(BasicBlock *BB) const override {
    if (!getAssumed())
      return false;
    return !AssumedLiveBlocks.count(BB);
  }

  /// See AAIsDead::isKnownDead().
  bool isKnownDead(BasicBlock *BB) const override {
    if (!getKnown())
      return false;
    return !AssumedLiveBlocks.count(BB);
  }

  /// Collection of to be explored paths.
  SmallSetVector<Instruction *, 8> ToBeExploredPaths;

  /// Collection of all assumed live BasicBlocks.
  DenseSet<BasicBlock *> AssumedLiveBlocks;

  /// Collection of calls with noreturn attribute, assumed or knwon.
  SmallSetVector<Instruction *, 4> NoReturnCalls;
};

bool AAIsDeadFunction::explorePath(Attributor &A, Instruction *I) {
  BasicBlock *BB = I->getParent();

  while (I) {
    ImmutableCallSite ICS(I);

    if (ICS) {
      auto *NoReturnAA = A.getAAFor<AANoReturn>(*this, *I);

      if (NoReturnAA && NoReturnAA->isAssumedNoReturn()) {
        if (!NoReturnCalls.insert(I))
          // If I is already in the NoReturnCalls set, then it stayed noreturn
          // and we didn't discover any new instructions.
          return false;

        // Discovered new noreturn call, return true to indicate that I is not
        // noreturn anymore and should be deleted from NoReturnCalls.
        return true;
      }

      if (ICS.hasFnAttr(Attribute::NoReturn)) {
        if (!NoReturnCalls.insert(I))
          return false;

        return true;
      }
    }

    I = I->getNextNode();
  }

  // get new paths (reachable blocks).
  for (BasicBlock *SuccBB : successors(BB)) {
    Instruction *Inst = &(SuccBB->front());
    AssumedLiveBlocks.insert(SuccBB);
    ToBeExploredPaths.insert(Inst);
  }

  return true;
}

ChangeStatus AAIsDeadFunction::updateImpl(Attributor &A) {
  // Temporary collection to iterate over existing noreturn instructions. This
  // will alow easier modification of NoReturnCalls collection
  SmallVector<Instruction *, 8> NoReturnChanged;
  ChangeStatus Status = ChangeStatus::UNCHANGED;

  for (Instruction *I : NoReturnCalls)
    NoReturnChanged.push_back(I);

  for (Instruction *I : NoReturnChanged) {
    size_t Size = ToBeExploredPaths.size();

    // Still noreturn.
    if (!explorePath(A, I))
      continue;

    NoReturnCalls.remove(I);

    // No new paths.
    if (Size == ToBeExploredPaths.size())
      continue;

    // At least one new path.
    Status = ChangeStatus::CHANGED;

    // explore new paths.
    while (Size != ToBeExploredPaths.size())
      explorePath(A, ToBeExploredPaths[Size++]);
  }

  LLVM_DEBUG(
      dbgs() << "[AAIsDead] AssumedLiveBlocks: " << AssumedLiveBlocks.size()
             << "Total number of blocks: " << getAnchorScope().size() << "\n");

  return Status;
}

/// -------------------- Dereferenceable Argument Attribute --------------------

struct DerefState : public AbstractState,
                    AttributeCompatibleAbstractState,
                    AbstractStateCompatibleWith<DerefState> {

  /// State representing for dereferenceable bytes.
  IntegerState<> DerefBytesState;

  /// State representing that whether the value is nonnull or global.
  IntegerState<> NonNullGlobalState;

  /// Bits encoding for NonNullGlobalState.
  enum {
    DEREF_NONNULL = 1 << 0,
    DEREF_GLOBAL = 1 << 1,
  };

  /// See AbstractState::isValidState()
  bool isValidState() const override { return DerefBytesState.isValidState(); }

  // See AbstractState::isAtFixpoint()
  bool isAtFixpoint() const override {
    return DerefBytesState.isAtFixpoint() && NonNullGlobalState.isAtFixpoint();
  }

  /// Provided via AbstractStateCompatibleWith<..., Attribute>.
  virtual ChangeStatus addKnownFrom(const Attribute &Attr) override {
    if (Attr.getKindAsEnum() != Attribute::Dereferenceable &&
        Attr.getKindAsEnum() != Attribute::DereferenceableOrNull)
      return ChangeStatus::UNCHANGED;
    auto StateBefore = *this;
    if (Attr.getKindAsEnum() == Attribute::Dereferenceable)
      NonNullGlobalState.addKnownBits(DEREF_NONNULL);
    takeKnownDerefBytesMaximum(Attr.getValueAsInt());
    return StateBefore == *this ? ChangeStatus::UNCHANGED
                                : ChangeStatus::CHANGED;
  }

  /// Provided via AbstractStateCompatibleWith<..., DerefState>.
  virtual ChangeStatus addKnownFrom(const DerefState &DS) override {
    auto StateBefore = *this;
    DerefBytesState.addKnownFrom(DS.DerefBytesState);
    NonNullGlobalState.addKnownFrom(DS.NonNullGlobalState);
    return StateBefore == *this ? ChangeStatus::UNCHANGED
                                : ChangeStatus::CHANGED;
  }

  static bool isBestState(const Attribute &Attr) {
    // The attribute \p Attr is in the best possible state if it is a
    // dereferenceable attribute (not _or_null) and has the maximal value
    // allowed by the IR (checked by
    // AttributeCompatibleAbstractState::isBestState(Attr)).
    return Attr.getKindAsEnum() == Attribute::Dereferenceable &&
           AttributeCompatibleAbstractState::isBestState(Attr);
  }

  static bool isBestState(const DerefState &DS) {
    return decltype(DS.DerefBytesState)::isBestState(DS.DerefBytesState) &&
           decltype(DS.NonNullGlobalState)::isBestState(DS.NonNullGlobalState);
  }

  /// See AbstractState::indicateOptimisticFixpoint(...)
  ChangeStatus indicateOptimisticFixpoint() override {
    return DerefBytesState.indicateOptimisticFixpoint() |
           NonNullGlobalState.indicateOptimisticFixpoint();
  }

  /// See AbstractState::indicatePessimisticFixpoint(...)
  ChangeStatus indicatePessimisticFixpoint() override {
    return DerefBytesState.indicatePessimisticFixpoint() |
           NonNullGlobalState.indicatePessimisticFixpoint();
  }

  /// Update known dereferenceable bytes.
  void takeKnownDerefBytesMaximum(uint64_t Bytes) {
    DerefBytesState.takeKnownMaximum(Bytes);
  }

  /// Update assumed dereferenceable bytes.
  void takeAssumedDerefBytesMinimum(uint64_t Bytes) {
    DerefBytesState.takeAssumedMinimum(Bytes);
  }

  /// Update assumed NonNullGlobalState
  void updateAssumedNonNullGlobalState(bool IsNonNull, bool IsGlobal) {
    if (!IsNonNull)
      NonNullGlobalState.removeAssumedBits(DEREF_NONNULL);
    if (!IsGlobal)
      NonNullGlobalState.removeAssumedBits(DEREF_GLOBAL);
  }

  /// Equality for DerefState.
  bool operator==(const DerefState &R) {
    return this->DerefBytesState == R.DerefBytesState &&
           this->NonNullGlobalState == R.NonNullGlobalState;
  }

  static DerefState getWorstState() {
    DerefState DS;
    DS.indicatePessimisticFixpoint();
    return DS;
  }
};

struct AADereferenceableImpl : StatefulAbstractAttribute<AADereferenceable, DerefState> {

  AADereferenceableImpl(Value *AssociatedVal, Value &AnchoredValue, int AttrIdx,
                        InformationCache &InfoCache)
      : StatefulAbstractAttribute(AssociatedVal, AnchoredValue, AttrIdx,
                                  InfoCache) {}

  void initialize(Attributor &A) override {
    StateType::addKnownFrom(StateType::template getFromIR<AADereferenceableImpl>(
        getAnchorScope(), getAnchoredValue(), getArgNo()));
  }

  /// See AADereferenceable::getAssumedDereferenceableBytes().
  uint32_t getAssumedDereferenceableBytes() const override {
    return DerefBytesState.getAssumed();
  }

  /// See AADereferenceable::getKnownDereferenceableBytes().
  uint32_t getKnownDereferenceableBytes() const override {
    return DerefBytesState.getKnown();
  }

  // Helper function for syncing nonnull state.
  void syncNonNull(const AANonNull *NonNullAA) {
    if (!NonNullAA) {
      NonNullGlobalState.removeAssumedBits(DEREF_NONNULL);
      return;
    }

    if (NonNullAA->isKnownNonNull())
      NonNullGlobalState.addKnownBits(DEREF_NONNULL);

    if (!NonNullAA->isAssumedNonNull())
      NonNullGlobalState.removeAssumedBits(DEREF_NONNULL);
  }

  /// See AADereferenceable::isAssumedGlobal().
  bool isAssumedGlobal() const override {
    return NonNullGlobalState.isAssumed(DEREF_GLOBAL);
  }

  /// See AADereferenceable::isKnownGlobal().
  bool isKnownGlobal() const override {
    return NonNullGlobalState.isKnown(DEREF_GLOBAL);
  }

  /// See AADereferenceable::isAssumedNonNull().
  bool isAssumedNonNull() const override {
    return NonNullGlobalState.isAssumed(DEREF_NONNULL);
  }

  /// See AADereferenceable::isKnownNonNull().
  bool isKnownNonNull() const override {
    return NonNullGlobalState.isKnown(DEREF_NONNULL);
  }

  void getDeducedAttributes(SmallVectorImpl<Attribute> &Attrs) const override {
    LLVMContext &Ctx = AnchoredVal.getContext();

    // TODO: Add *_globally support
    if (isAssumedNonNull())
      Attrs.emplace_back(Attribute::getWithDereferenceableBytes(
          Ctx, getAssumedDereferenceableBytes()));
    else
      Attrs.emplace_back(Attribute::getWithDereferenceableOrNullBytes(
          Ctx, getAssumedDereferenceableBytes()));
  }
  uint64_t computeAssumedDerefenceableBytes(Attributor &A, const Value &V,
                                            bool &IsNonNull, bool &IsGlobal);

  #if 0
  void initialize(Attributor &A) override {
    Function &F = getAnchorScope();
    unsigned AttrIdx =
        getAttrIndex(getManifestPosition(), getArgNo());

    for (Attribute::AttrKind AK :
         {Attribute::Dereferenceable, Attribute::DereferenceableOrNull})
      if (F.getAttributes().hasAttribute(AttrIdx, AK))
        takeKnownDerefBytesMaximum(F.getAttribute(AttrIdx, AK).getValueAsInt());
  }
  #endif

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    if (!getAssumedDereferenceableBytes())
      return "unknown-dereferenceable";
    return std::string("dereferenceable") +
           (isAssumedNonNull() ? "" : "_or_null") +
           (isAssumedGlobal() ? "_globally" : "") + "<" +
           std::to_string(getKnownDereferenceableBytes()) + "-" +
           std::to_string(getAssumedDereferenceableBytes()) + ">";
  }
};

struct AADereferenceableReturned : AADereferenceableImpl {
  AADereferenceableReturned(Function &F, InformationCache &InfoCache)
      : AADereferenceableImpl(&F, F, MP_RETURNED, InfoCache) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;
};

// Helper function that returns dereferenceable bytes.
static uint64_t calcDifferenceIfBaseIsNonNull(int64_t DerefBytes,
                                              int64_t Offset, bool IsNonNull) {
  if (!IsNonNull)
    return 0;
  return std::max((int64_t)0, DerefBytes - Offset);
}

uint64_t AADereferenceableImpl::computeAssumedDerefenceableBytes(
    Attributor &A, const Value &V, bool &IsNonNull, bool &IsGlobal) {
  // TODO: Tracking the globally flag.
  IsGlobal = false;

  // First, we try to get information about V from Attributor.
  if (auto *DerefAA = A.getAAFor<AADereferenceable>(*this, V)) {
    IsNonNull &= DerefAA->isAssumedNonNull();
    return DerefAA->getAssumedDereferenceableBytes();
  }

  // Otherwise, we try to compute assumed bytes from base pointer.
  const DataLayout &DL = getAnchorScope().getParent()->getDataLayout();
  unsigned IdxWidth =
      DL.getIndexSizeInBits(V.getType()->getPointerAddressSpace());
  APInt Offset(IdxWidth, 0);
  const Value *Base = V.stripAndAccumulateInBoundsConstantOffsets(DL, Offset);

  if (auto *BaseDerefAA = A.getAAFor<AADereferenceable>(*this, *Base)) {
    IsNonNull &= Offset != 0;
    return calcDifferenceIfBaseIsNonNull(
        BaseDerefAA->getAssumedDereferenceableBytes(), Offset.getSExtValue(),
        Offset != 0 || BaseDerefAA->isAssumedNonNull());
  }

  // Then, use IR information.

  if (isDereferenceablePointer(Base, Base->getType(), DL))
    return calcDifferenceIfBaseIsNonNull(
        DL.getTypeStoreSize(Base->getType()->getPointerElementType()),
        Offset.getSExtValue(),
        !NullPointerIsDefined(&getAnchorScope(),
                              V.getType()->getPointerAddressSpace()));

  IsNonNull = false;
  return 0;
}
ChangeStatus AADereferenceableReturned::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();
  auto BeforeState = static_cast<DerefState>(*this);

  syncNonNull(A.getAAFor<AANonNull>(*this, F));

  auto *AARetVal = A.getAAFor<AAReturnedValues>(*this, F);
  if (!AARetVal)
    return indicatePessimisticFixpoint();

  bool IsNonNull = isAssumedNonNull();
  bool IsGlobal = isAssumedGlobal();

  std::function<bool(const Value &)> Pred = [&](const Value &RV) -> bool {
    takeAssumedDerefBytesMinimum(
        computeAssumedDerefenceableBytes(A, RV, IsNonNull, IsGlobal));
    return isValidState();
  };

  //if (AARetVal->checkForallReturnedValues(Pred)) {
    //updateAssumedNonNullGlobalState(IsNonNull, IsGlobal);
    //return BeforeState == static_cast<DerefState>(*this)
               //? ChangeStatus::UNCHANGED
               //: ChangeStatus::CHANGED;
  //}
  return indicatePessimisticFixpoint();
}

struct AADereferenceableArgument : AADereferenceableImpl {
  AADereferenceableArgument(Argument &A, InformationCache &InfoCache)
      : AADereferenceableImpl(&A, A, A.getArgNo(), InfoCache) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;
};

ChangeStatus AADereferenceableArgument::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();
  Argument &Arg = cast<Argument>(getAnchoredValue());

  auto BeforeState = static_cast<DerefState>(*this);

  unsigned ArgNo = Arg.getArgNo();

  syncNonNull(A.getAAFor<AANonNull>(*this, F, ArgNo));

  bool IsNonNull = isAssumedNonNull();
  bool IsGlobal = isAssumedGlobal();

  // Callback function
  std::function<bool(CallSite)> CallSiteCheck = [&](CallSite CS) -> bool {
    assert(CS && "Sanity check: Call site was not initialized properly!");

    // Check that DereferenceableAA is AADereferenceableCallSiteArgument.
    if (auto *DereferenceableAA =
            A.getAAFor<AADereferenceable>(*this, *CS.getInstruction(), ArgNo)) {
      ImmutableCallSite ICS(&DereferenceableAA->getAnchoredValue());
      if (ICS && CS.getInstruction() == ICS.getInstruction()) {
        takeAssumedDerefBytesMinimum(
            DereferenceableAA->getAssumedDereferenceableBytes());
        IsNonNull &= DereferenceableAA->isAssumedNonNull();
        IsGlobal &= DereferenceableAA->isAssumedGlobal();
        return isValidState();
      }
    }

    takeAssumedDerefBytesMinimum(computeAssumedDerefenceableBytes(
        A, *CS.getArgOperand(ArgNo), IsNonNull, IsGlobal));

    return isValidState();
  };

  //if (!A.checkForAllCallSites(F, CallSiteCheck, true)) {
    //indicatePessimisticFixpoint();
    //return ChangeStatus::CHANGED;
  //}

  updateAssumedNonNullGlobalState(IsNonNull, IsGlobal);

  return BeforeState == static_cast<DerefState>(*this) ? ChangeStatus::UNCHANGED
                                                       : ChangeStatus::CHANGED;
}

/// Dereferenceable attribute for a call site argument.
struct AADereferenceableCallSiteArgument : AADereferenceableImpl {

  /// See AADereferenceableImpl::AADereferenceableImpl(...).
  AADereferenceableCallSiteArgument(Instruction &CBInst,
                                    InformationCache &InfoCache, unsigned ArgNo)
      : AADereferenceableImpl(CallSite(&CBInst).getArgOperand(ArgNo), CBInst,
                              ArgNo, InfoCache) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    CallSite CS(&getAnchoredValue());
    if (CS.paramHasAttr(getArgNo(), Attribute::Dereferenceable))
      takeKnownDerefBytesMaximum(CS.getDereferenceableBytes(getArgNo()));

    if (CS.paramHasAttr(getArgNo(), Attribute::DereferenceableOrNull))
      takeKnownDerefBytesMaximum(CS.getDereferenceableOrNullBytes(getArgNo()));
  }

  /// See AbstractAttribute::updateImpl(Attributor &A).
  ChangeStatus updateImpl(Attributor &A) override;
};

ChangeStatus AADereferenceableCallSiteArgument::updateImpl(Attributor &A) {
  // NOTE: Never look at the argument of the callee in this method.
  //       If we do this, "dereferenceable" is always deduced because of the
  //       assumption.

  Value &V = *getAssociatedValue();

  auto BeforeState = static_cast<DerefState>(*this);

  syncNonNull(A.getAAFor<AANonNull>(*this, getAnchoredValue(), getArgNo()));
  bool IsNonNull = isAssumedNonNull();
  bool IsGlobal = isKnownGlobal();

  takeAssumedDerefBytesMinimum(
      computeAssumedDerefenceableBytes(A, V, IsNonNull, IsGlobal));
  updateAssumedNonNullGlobalState(IsNonNull, IsGlobal);

  return BeforeState == static_cast<DerefState>(*this) ? ChangeStatus::UNCHANGED
                                                       : ChangeStatus::CHANGED;
}

// ------------------------ Align Argument Attribute ------------------------

struct AAAlignImpl
    : StatefulAbstractAttribute<AAAlign, IntegerState<AAAlign::MAX_ALIGN>> {

  AAAlignImpl(Value *AssociatedVal, Value &AnchoredValue, int AttrIdx,
              InformationCache &InfoCache)
      : StatefulAbstractAttribute(AssociatedVal, AnchoredValue, AttrIdx,
                                  InfoCache) {}

  void initialize(Attributor &A) override {
    StateType::addKnownFrom(StateType::template getFromIR<AAAlignImpl>(
        getAnchorScope(), getAnchoredValue(), getArgNo()));
  }

  virtual const std::string getAsStr() const override {
    return getAssumedAlign() ? ("align<" + std::to_string(getKnownAlign()) +
                                "-" + std::to_string(getAssumedAlign()) + ">")
                             : "unknown-align";
  }

  /// See AAAlign::getAssumedAlign().
  unsigned getAssumedAlign() const override { return getAssumed(); }

  /// See AAAlign::getKnownAlign().
  unsigned getKnownAlign() const override { return getKnown(); }

  /// See AbstractAttribute::getDeducedAttributes
  virtual void
  getDeducedAttributes(SmallVectorImpl<Attribute> &Attrs) const override {
    LLVMContext &Ctx = AnchoredVal.getContext();

    Attrs.emplace_back(Attribute::getWithAlignment(Ctx, getAssumedAlign()));
  }
};

/// Align attribute for function return value.
struct AAAlignReturned : AAAlignImpl {

  AAAlignReturned(Function &F, InformationCache &InfoCache)
      : AAAlignImpl(&F, F, MP_RETURNED, InfoCache) {}

  /// See AbstractAttribute::updateImpl(...).
  virtual ChangeStatus updateImpl(Attributor &A) override;
};

ChangeStatus AAAlignReturned::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();
  auto *AARetValImpl = A.getAAFor<AAReturnedValuesImpl>(*this, F);
  if (!AARetValImpl) {
    indicatePessimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  // Currently, align<n> is deduced if alignments in return values are assumed
  // as greater than n. We reach pessimistic fixpoint if any of the return value
  // wouldn't have align. If no assumed state was used for reasoning, an
  // optimistic fixpoint is reached earlier.

  base_t BeforeState = getAssumed();
  std::function<bool(const Value &)> Pred = [&](const Value &RV) -> bool {
    auto *AlignAA = A.getAAFor<AAAlign>(*this, RV);

    if (AlignAA)
      takeAssumedMinimum(AlignAA->getAssumedAlign());
    else
      // Use IR information.
      takeAssumedMinimum(RV.getPointerAlignment(
          getAnchorScope().getParent()->getDataLayout()));

    return isValidState();
  };

  if (!AARetValImpl->checkForallReturnedValues(Pred)) {
    indicatePessimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  return (getAssumed() != BeforeState) ? ChangeStatus::CHANGED
                                       : ChangeStatus::UNCHANGED;
}

/// Align attribute for function argument.
struct AAAlignArgument : AAAlignImpl {

  AAAlignArgument(Argument &A, InformationCache &InfoCache)
      : AAAlignImpl(&A, A, A.getArgNo(), InfoCache) {}

  /// See AbstractAttribute::updateImpl(...).
  virtual ChangeStatus updateImpl(Attributor &A) override;
};

ChangeStatus AAAlignArgument::updateImpl(Attributor &A) {

  Function &F = getAnchorScope();
  Argument &Arg = cast<Argument>(getAnchoredValue());

  unsigned ArgNo = Arg.getArgNo();
  const DataLayout &DL = F.getParent()->getDataLayout();

  auto BeforeState = getAssumed();

  // Callback function
  std::function<bool(CallSite)> CallSiteCheck = [&](CallSite CS) {
    assert(CS && "Sanity check: Call site was not initialized properly!");

    auto *AlignAA = A.getAAFor<AAAlign>(*this, *CS.getInstruction(), ArgNo);

    // Check that AlignAA is AAAlignCallSiteArgument.
    if (AlignAA) {
      ImmutableCallSite ICS(&AlignAA->getAnchoredValue());
      if (ICS && CS.getInstruction() == ICS.getInstruction()) {
        takeAssumedMinimum(AlignAA->getAssumedAlign());
        return isValidState();
      }
    }

    Value *V = CS.getArgOperand(ArgNo);
    takeAssumedMinimum(V->getPointerAlignment(DL));
    return isValidState();
  };

  //if (!A.checkForAllCallSites(F, CallSiteCheck, true))
    //indicatePessimisticFixpoint();

  return BeforeState == getAssumed() ? ChangeStatus::UNCHANGED
                                     : ChangeStatus ::CHANGED;
}

struct AAAlignCallSiteArgument : AAAlignImpl {

  /// See AANonNullImpl::AANonNullImpl(...).
  AAAlignCallSiteArgument(Instruction &CBInst, InformationCache &InfoCache,
                          unsigned ArgNo)
      : AAAlignImpl(CallSite(&CBInst).getArgOperand(ArgNo), CBInst, ArgNo,
                    InfoCache) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    CallSite CS(&getAnchoredValue());
    takeKnownMaximum(getAssociatedValue()->getPointerAlignment(
        getAnchorScope().getParent()->getDataLayout()));
  }

  /// See AbstractAttribute::updateImpl(Attributor &A).
  ChangeStatus updateImpl(Attributor &A) override;
};

ChangeStatus AAAlignCallSiteArgument::updateImpl(Attributor &A) {
  // NOTE: Never look at the argument of the callee in this method.
  //       If we do this, "align" is always deduced because of the assumption.

  auto BeforeState = getAssumed();

  Value &V = *getAssociatedValue();

  auto *AlignAA = A.getAAFor<AAAlign>(*this, V);

  if (AlignAA)
    takeAssumedMinimum(AlignAA->getAssumedAlign());
  else
    indicatePessimisticFixpoint();

  return BeforeState == getAssumed() ? ChangeStatus::UNCHANGED
                                     : ChangeStatus::CHANGED;
}

/// ----------------------------------------------------------------------------
///                               Attributor
/// ----------------------------------------------------------------------------

template<typename AAType>
bool Attributor::checkForAllCallSites(AbstractAttribute &QueryingAA,
                                      const Function &F, bool &Assumed,
                                      int ArgNo, bool RequireAllCallSites) {
  // We can try to determine information from
  // the call sites. However, this is only possible all call sites are known,
  // hence the function has internal linkage.
  if (RequireAllCallSites && !F.hasInternalLinkage()) {
    LLVM_DEBUG(
        dbgs()
        << "Attributor: Function " << F.getName()
        << " has no internal linkage, hence not all call sites are known\n");
    return false;
  }

  // Callback function
  auto CallSiteCheck = [&](CallSite CS) {
//    return QueryingAA.getAssumedOrKnown<AAType>(
//        *this, QueryingAA, Assumed, *CS.getInstruction(), ArgNo,
//        /* CallSiteTraversal */ true, false);
    return true;
  };

  for (const Use &U : F.uses()) {

    CallSite CS(U.getUser());
    if (!CS || !CS.isCallee(&U) || !CS.getCaller()->hasExactDefinition()) {
      if (!RequireAllCallSites)
        continue;

      LLVM_DEBUG(dbgs() << "Attributor: User " << *U.getUser()
                        << " is an invalid use of " << F.getName() << "\n");
      return false;
    }

    if (CallSiteCheck(CS))
      continue;

    LLVM_DEBUG(dbgs() << "Attributor: Call site callback failed for "
                      << *CS.getInstruction() << "\n");
    return false;
  }

  return true;
}

#if 1
template <typename AAType>
Attribute Attributor::checkForallReturnedValues(AbstractAttribute &QueryingAA,
                                                const Function &F,
                                                bool &Assumed) {
  Attribute Attr;
  if (const Argument *Arg = ReturnedValuesStates::getKnownReturnedArg(F)) {
    if (AAType *AA = getAAFor<AAType>(QueryingAA, *Arg)) {
      assert(isa<Argument>(AA->getAssociatedValue()));
      //Attr = QueryingAA.getAssumedOrKnown<AAType>(*this, QueryingAA, Assumed,
          //*AA->getAssociatedValue());
    }
    // Assumed,
  }

  auto *AARetVal = getAAFor<AAReturnedValues>(QueryingAA, F);
  //if (!AARetVal)
    //return false;

  ReturnValuePredicateFuncTy Pred =
      [&](Attributor &A, const Value &RV,
          const SmallPtrSetImpl<ReturnInst *> &ReturnInsts) -> bool {
    //return QueryingAA.getAssumedOrKnown<AAType>(*this, QueryingAA, Assumed, RV, -1);
        return false;
  };

  //return AARetVal->checkForallReturnedValues(*this, Pred);
}
#endif

ChangeStatus Attributor::run() {
  // Initialize all abstract attributes.
  for (AbstractAttribute *AA : AllAbstractAttributes)
    AA->initialize(*this);

  LLVM_DEBUG(dbgs() << "[Attributor] Identified and initialized "
                    << AllAbstractAttributes.size()
                    << " abstract attributes.\n");

  // Now that all abstract attributes are collected and initialized we start
  // the abstract analysis.

  unsigned IterationCounter = 1;

  SmallVector<AbstractAttribute *, 64> ChangedAAs;
  SetVector<AbstractAttribute *> Worklist;
  Worklist.insert(AllAbstractAttributes.begin(), AllAbstractAttributes.end());

  do {
    LLVM_DEBUG(dbgs() << "\n\n[Attributor] #Iteration: " << IterationCounter
                      << ", Worklist size: " << Worklist.size() << "\n");

    // Add all abstract attributes that are potentially dependent on one that
    // changed to the work list.
    for (AbstractAttribute *ChangedAA : ChangedAAs) {
      auto &QuerriedAAs = QueryMap[ChangedAA];
      Worklist.insert(QuerriedAAs.begin(), QuerriedAAs.end());
    }

    // Reset the changed set.
    ChangedAAs.clear();

    // Update all abstract attribute in the work list and record the ones that
    // changed.
    for (AbstractAttribute *AA : Worklist)
      if (AA->update(*this) == ChangeStatus::CHANGED)
        ChangedAAs.push_back(AA);

    // Reset the work list and repopulate with the changed abstract attributes.
    // Note that dependent ones are added above.
    Worklist.clear();
    Worklist.insert(ChangedAAs.begin(), ChangedAAs.end());

  } while (!Worklist.empty() && ++IterationCounter < MaxFixpointIterations);

  LLVM_DEBUG(dbgs() << "\n[Attributor] Fixpoint iteration done after: "
                    << IterationCounter << "/" << MaxFixpointIterations
                    << " iterations\n");

  bool FinishedAtFixpoint = Worklist.empty();

  // Reset abstract arguments not settled in a sound fixpoint by now. This
  // happens when we stopped the fixpoint iteration early. Note that only the
  // ones marked as "changed" *and* the ones transitively depending on them
  // need to be reverted to a pessimistic state. Others might not be in a
  // fixpoint state but we can use the optimistic results for them anyway.
  SmallPtrSet<AbstractAttribute *, 32> Visited;
  for (unsigned u = 0; u < ChangedAAs.size(); u++) {
    AbstractAttribute *ChangedAA = ChangedAAs[u];
    if (!Visited.insert(ChangedAA).second)
      continue;

    AbstractState &State = ChangedAA->getState();
    if (!State.isAtFixpoint()) {
      State.indicatePessimisticFixpoint();

      NumAttributesTimedOut++;
    }

    auto &QuerriedAAs = QueryMap[ChangedAA];
    ChangedAAs.append(QuerriedAAs.begin(), QuerriedAAs.end());
  }

  LLVM_DEBUG({
    if (!Visited.empty())
      dbgs() << "\n[Attributor] Finalized " << Visited.size()
             << " abstract attributes.\n";
  });

  unsigned NumManifested = 0;
  unsigned NumAtFixpoint = 0;
  ChangeStatus ManifestChange = ChangeStatus::UNCHANGED;
  for (AbstractAttribute *AA : AllAbstractAttributes) {
    AbstractState &State = AA->getState();

    // If there is not already a fixpoint reached, we can now take the
    // optimistic state. This is correct because we enforced a pessimistic one
    // on abstract attributes that were transitively dependent on a changed one
    // already above.
    if (!State.isAtFixpoint())
      State.indicateOptimisticFixpoint();

    // If the state is invalid, we do not try to manifest it.
    if (!State.isValidState())
      continue;

    // Manifest the state and record if we changed the IR.
    ChangeStatus LocalChange = AA->manifest(*this);
    ManifestChange = ManifestChange | LocalChange;

    NumAtFixpoint++;
    NumManifested += (LocalChange == ChangeStatus::CHANGED);
  }

  (void)NumManifested;
  (void)NumAtFixpoint;
  LLVM_DEBUG(dbgs() << "\n[Attributor] Manifested " << NumManifested
                    << " arguments while " << NumAtFixpoint
                    << " were in a valid fixpoint state\n");

  // If verification is requested, we finished this run at a fixpoint, and the
  // IR was changed, we re-run the whole fixpoint analysis, starting at
  // re-initialization of the arguments. This re-run should not result in an IR
  // change. Though, the (virtual) state of attributes at the end of the re-run
  // might be more optimistic than the known state or the IR state if the better
  // state cannot be manifested.
  if (VerifyAttributor && FinishedAtFixpoint &&
      ManifestChange == ChangeStatus::CHANGED) {
    VerifyAttributor = false;
    ChangeStatus VerifyStatus = run();
    if (VerifyStatus != ChangeStatus::UNCHANGED)
      llvm_unreachable(
          "Attributor verification failed, re-run did result in an IR change "
          "even after a fixpoint was reached in the original run. (False "
          "positives possible!)");
    VerifyAttributor = true;
  }

  NumAttributesManifested += NumManifested;
  NumAttributesValidFixpoint += NumAtFixpoint;

  return ManifestChange;
}

/// Helper function that checks if an abstract attribute of type \p AAType
/// should be created for \p V (with argument number \p ArgNo) and if so creates
/// and registers it with the Attributor \p A.
///
/// This method will look at the provided whitelist. If one is given and the
/// kind \p AAType::ID is not contained, no abstract attribute is created.
///
/// This method will look at the IR and if the information there is sufficient
/// to determine the property, no abstract attribute is created.
///
/// \returns The created abstract argument, or nullptr if none was created.
template <typename AAType, typename ValueType, typename... ArgsTy>
static AAType *
registerAAIfNeeded(const Function &F, Attributor &A,
                   DenseSet</* Attribute::AttrKind */ unsigned> *Whitelist,
                   InformationCache &InfoCache, ValueType &V, int ArgNo,
                   ArgsTy... Args) {
  if (Whitelist && !Whitelist->count(AAType::ID))
    return nullptr;

  // If the IR contains already the best possible information derivable by this
  // AAType we do not create an object.
  if (AAType::StateType::isBestState(
          AAType::StateType::template getFromIR<AAType>(F, V, ArgNo))) {
    ++NumAttributesSkippedDueToIR;
    return nullptr;
  }

  return &A.registerAA<AAType>(*new AAType(V, InfoCache, Args...), ArgNo);
}

void Attributor::identifyDefaultAbstractAttributes(
    Function &F, InformationCache &InfoCache,
    DenseSet</* Attribute::AttrKind */ unsigned> *Whitelist) {

  {
    // Attributes at the "function" (scope) position.

    // Every function can be nounwind.
    registerAAIfNeeded<AANoUnwindFunction>(F, *this, Whitelist, InfoCache, F,
                                           -1);

    // Every function might be marked "nosync"
    registerAAIfNeeded<AANoSyncFunction>(F, *this, Whitelist, InfoCache, F, -1);

    // Every function might be "no-free".
    registerAAIfNeeded<AANoFreeFunction>(F, *this, Whitelist, InfoCache, F, -1);

    // Every function might be "will-return".
    registerAAIfNeeded<AAWillReturnFunction>(F, *this, Whitelist, InfoCache, F,
                                             -1);

    // Check for dead BasicBlocks in every function.
    registerAAIfNeeded<AAIsDeadFunction>(F, *this, Whitelist, InfoCache, F, -1);
  }

  {
    // Attributes at the "returned" position.

    // Return attributes are only appropriate if the return type is non void.
    Type *ReturnType = F.getReturnType();
    if (!ReturnType->isVoidTy()) {

      // Argument attribute "returned" --- Create only one per function even
      // though it is an argument attribute.
      //registerAAIfNeeded<AAReturnedValuesImpl>(F, *this,
                                               //Whitelist, InfoCache, F, -1);

      if (ReturnType->isPointerTy()) {
        // Every function with pointer return type might be marked align.
        registerAAIfNeeded<AAAlignReturned>(F, *this, Whitelist, InfoCache, F,
                                            -1);

        // Every function with pointer return type might be marked nonnull.
        registerAAIfNeeded<AANonNullReturned>(F, *this, Whitelist, InfoCache, F,
                                              -1);

        // Every function with pointer return type might be marked noalias.
        registerAAIfNeeded<AANoAliasReturned>(F, *this, Whitelist, InfoCache, F,
                                              -1);

        // Every function with pointer return type might be marked
        // dereferenceable.
        registerAAIfNeeded<AADereferenceableReturned>(F, *this, Whitelist,
                                                      InfoCache, F, -1);
      }
    }
  }

  {
    // Attributes at the "argument" position.

    for (Argument &Arg : F.args()) {
      if (Arg.getType()->isPointerTy()) {
        // Every argument with pointer type might be marked nonnull.
        registerAAIfNeeded<AANonNullArgument>(F, *this, Whitelist, InfoCache,
                                              Arg, Arg.getArgNo());

        // Every argument with pointer type might be marked dereferenceable.
        registerAAIfNeeded<AADereferenceableArgument>(
            F, *this, Whitelist, InfoCache, Arg, Arg.getArgNo());

        // Every argument with pointer type might be marked align.
        registerAAIfNeeded<AAAlignArgument>(F, *this, Whitelist, InfoCache, Arg,
                                            Arg.getArgNo());
      }
    }
  }

  // Walk all instructions to find more attribute opportunities and also
  // interesting instructions that might be queried by abstract attributes
  // during their initialization or update.
  auto &ReadOrWriteInsts = InfoCache.FuncRWInstsMap[&F];
  auto &InstOpcodeMap = InfoCache.FuncInstOpcodeMap[&F];

  for (Instruction &I : instructions(&F)) {
    bool IsInterestingOpcode = false;

    // To allow easy access to all instructions in a function with a given
    // opcode we store them in the InfoCache. As not all opcodes are interesting
    // to concrete attributes we only cache the ones that are as identified in
    // the following switch.
    // Note: There are no concrete attributes now so this is initially empty.
    switch (I.getOpcode()) {
    default:
      assert((!ImmutableCallSite(&I)) && (!isa<CallBase>(&I)) &&
             "New call site/base instruction type needs to be known int the "
             "attributor.");
      break;
    case Instruction::Call:
    case Instruction::CallBr:
    case Instruction::Invoke:
    case Instruction::CleanupRet:
    case Instruction::CatchSwitch:
    case Instruction::Resume:
    case Instruction::Ret:
      IsInterestingOpcode = true;
    }
    if (IsInterestingOpcode)
      InstOpcodeMap[I.getOpcode()].push_back(&I);
    if (I.mayReadOrWriteMemory())
      ReadOrWriteInsts.push_back(&I);

    {
      // Attributes at the "call site argument" position.

      CallSite CS(&I);
      if (CS && CS.getCalledFunction()) {
        for (int i = 0, e = CS.getCalledFunction()->arg_size(); i < e; i++) {
          if (!CS.getArgument(i)->getType()->isPointerTy())
            continue;

          // Call site argument attribute "non-null".
          registerAAIfNeeded<AANonNullCallSiteArgument>(F, *this, Whitelist,
                                                        InfoCache, I, i, i);

          // Call site argument attribute "dereferenceable".
          registerAAIfNeeded<AADereferenceableCallSiteArgument>(
              F, *this, Whitelist, InfoCache, I, i, i);

          // Call site argument attribute "align".
          registerAAIfNeeded<AAAlignCallSiteArgument>(F, *this, Whitelist,
                                                      InfoCache, I, i, i);
        }
      }
    }
  }
}

/// Helpers to ease debugging through output streams and print calls.
///
///{
raw_ostream &llvm::operator<<(raw_ostream &OS, ChangeStatus S) {
  return OS << (S == ChangeStatus::CHANGED ? "changed" : "unchanged");
}

raw_ostream &llvm::operator<<(raw_ostream &OS,
                              AbstractAttribute::ManifestPosition AP) {
  switch (AP) {
  default:
    llvm_unreachable("Unhandled manifest position!");
  case AbstractAttribute::MP_ARGUMENT:
    return OS << "arg";
  case AbstractAttribute::MP_CALL_SITE_ARGUMENT:
    return OS << "cs_arg";
  case AbstractAttribute::MP_FUNCTION:
    return OS << "fn";
  case AbstractAttribute::MP_RETURNED:
    return OS << "fn_ret";
  }
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const AbstractState &S) {
  return OS << (!S.isValidState() ? "top" : (S.isAtFixpoint() ? "fix" : ""));
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const IntegerState<> &IS) {
  return OS << "IS<" << IS.getAssumed() << "-" << IS.getKnown() << ">";
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const AbstractAttribute &AA) {
  AA.print(OS);
  return OS;
}

void AbstractAttribute::print(raw_ostream &OS) const {
  OS << "[" << getManifestPosition() << "][" << getAsStr() << "]["
     << AnchoredVal.getName() << "]";
}
///}

bool AttributeCompatibleAbstractState::isBestState(const Attribute &Attr) {
  // All but "None" enum attributes are in their best state (they are on/off).
  if (Attr.isEnumAttribute())
    return Attr.getKindAsEnum() != Attribute::None;

  // For integer attributes we look at the value and check it agains known
  // maximal values.
  if (Attr.isIntAttribute()) {
    uint64_t IntVal = Attr.getValueAsInt();
    switch (Attr.getKindAsEnum()) {
    case Attribute::Alignment:
      return IntVal == AAAlign::MAX_ALIGN;
    case Attribute::Dereferenceable:
    case Attribute::DereferenceableOrNull:
      // TODO: What is the maximal value here?
      break;
    default:
      break;
    }
    return IntVal == std::numeric_limits<decltype(IntVal)>::max();
  }
  // TODO: Categorize string attributes here as well.
  return false;
}

/// ----------------------------------------------------------------------------
///                       Pass (Manager) Boilerplate
/// ----------------------------------------------------------------------------

static bool runAttributorOnModule(Module &M) {
  if (DisableAttributor)
    return false;

  LLVM_DEBUG(dbgs() << "[Attributor] Run on module with " << M.size()
                    << " functions.\n");

  // Create an Attributor and initially empty information cache that is filled
  // while we identify default attribute opportunities.
  Attributor A;
  InformationCache InfoCache;

  for (Function &F : M) {
    // TODO: Not all attributes require an exact definition. Find a way to
    //       enable deduction for some but not all attributes in case the
    //       definition might be changed at runtime, see also
    //       http://lists.llvm.org/pipermail/llvm-dev/2018-February/121275.html.
    // TODO: We could always determine abstract attributes and if sufficient
    //       information was found we could duplicate the functions that do not
    //       have an exact definition.
    if (!F.hasExactDefinition()) {
      NumFnWithoutExactDefinition++;
      continue;
    }

    // For now we ignore naked and optnone functions.
    if (F.hasFnAttribute(Attribute::Naked) ||
        F.hasFnAttribute(Attribute::OptimizeNone))
      continue;

    NumFnWithExactDefinition++;

    // Populate the Attributor with abstract attribute opportunities in the
    // function and the information cache with IR information.
    A.identifyDefaultAbstractAttributes(F, InfoCache);
  }

  return A.run() == ChangeStatus::CHANGED;
}

PreservedAnalyses AttributorPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (runAttributorOnModule(M)) {
    // FIXME: Think about passes we will preserve and add them here.
    return PreservedAnalyses::none();
  }
  return PreservedAnalyses::all();
}

namespace {

struct AttributorLegacyPass : public ModulePass {
  static char ID;

  AttributorLegacyPass() : ModulePass(ID) {
    initializeAttributorLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;
    return runAttributorOnModule(M);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // FIXME: Think about passes we will preserve and add them here.
    AU.setPreservesCFG();
  }
};

} // end anonymous namespace

Pass *llvm::createAttributorLegacyPass() { return new AttributorLegacyPass(); }

char AttributorLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(AttributorLegacyPass, "attributor",
                      "Deduce and propagate attributes", false, false)
INITIALIZE_PASS_END(AttributorLegacyPass, "attributor",
                    "Deduce and propagate attributes", false, false)
