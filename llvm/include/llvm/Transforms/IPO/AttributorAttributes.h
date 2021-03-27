//===- AttributorAttributes.h --- Attributes for the Attributor -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// See Attributor.h
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_ATTRIBUTOR_ATTRIBUTES_H
#define LLVM_TRANSFORMS_IPO_ATTRIBUTOR_ATTRIBUTES_H

#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/IPO/Attributor.h"

#include <functional>

namespace llvm {

/// Simple state with integers encoding.
///
/// The interface ensures that the assumed bits are always a subset of the known
/// bits. Users can only add known bits and, except through adding known bits,
/// they can only remove assumed bits. This should guarantee monotoniticy and
/// thereby the existence of a fixpoint (if used corretly). The fixpoint is
/// reached when the assumed and known state/bits are equal. Users can
/// force/inidicate a fixpoint. If an optimistic one is indicated, the known
/// state will catch up with the assumed one, for a pessimistic fixpoint it is
/// the other way around.
template <typename base_ty, base_ty BestState, base_ty WorstState>
struct IntegerStateBase : public AbstractState {
  using base_t = base_ty;

  IntegerStateBase() {}
  IntegerStateBase(base_t Assumed) : Assumed(Assumed) {}

  /// Return the best possible representable state.
  static constexpr base_t getBestState() { return BestState; }
  static constexpr base_t getBestState(const IntegerStateBase &) {
    return getBestState();
  }

  /// Return the worst possible representable state.
  static constexpr base_t getWorstState() { return WorstState; }
  static constexpr base_t getWorstState(const IntegerStateBase &) {
    return getWorstState();
  }

  /// See AbstractState::isValidState()
  /// NOTE: For now we simply pretend that the worst possible state is invalid.
  bool isValidState() const override { return Assumed != getWorstState(); }

  /// See AbstractState::isAtFixpoint()
  bool isAtFixpoint() const override { return Assumed == Known; }

  /// See AbstractState::indicateOptimisticFixpoint(...)
  ChangeStatus indicateOptimisticFixpoint() override {
    Known = Assumed;
    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractState::indicatePessimisticFixpoint(...)
  ChangeStatus indicatePessimisticFixpoint() override {
    Assumed = Known;
    return ChangeStatus::CHANGED;
  }

  /// Return the known state encoding
  base_t getKnown() const { return Known; }

  /// Return the assumed state encoding.
  base_t getAssumed() const { return Assumed; }

  /// Equality for IntegerStateBase.
  bool
  operator==(const IntegerStateBase<base_t, BestState, WorstState> &R) const {
    return this->getAssumed() == R.getAssumed() &&
           this->getKnown() == R.getKnown();
  }

  /// Inequality for IntegerStateBase.
  bool
  operator!=(const IntegerStateBase<base_t, BestState, WorstState> &R) const {
    return !(*this == R);
  }

  /// "Clamp" this state with \p R. The result is subtype dependent but it is
  /// intended that only information assumed in both states will be assumed in
  /// this one afterwards.
  void operator^=(const IntegerStateBase<base_t, BestState, WorstState> &R) {
    handleNewAssumedValue(R.getAssumed());
  }

  /// "Clamp" this state with \p R. The result is subtype dependent but it is
  /// intended that information known in either state will be known in
  /// this one afterwards.
  void operator+=(const IntegerStateBase<base_t, BestState, WorstState> &R) {
    handleNewKnownValue(R.getKnown());
  }

  void operator|=(const IntegerStateBase<base_t, BestState, WorstState> &R) {
    joinOR(R.getAssumed(), R.getKnown());
  }

  void operator&=(const IntegerStateBase<base_t, BestState, WorstState> &R) {
    joinAND(R.getAssumed(), R.getKnown());
  }

protected:
  /// Handle a new assumed value \p Value. Subtype dependent.
  virtual void handleNewAssumedValue(base_t Value) = 0;

  /// Handle a new known value \p Value. Subtype dependent.
  virtual void handleNewKnownValue(base_t Value) = 0;

  /// Handle a  value \p Value. Subtype dependent.
  virtual void joinOR(base_t AssumedValue, base_t KnownValue) = 0;

  /// Handle a new assumed value \p Value. Subtype dependent.
  virtual void joinAND(base_t AssumedValue, base_t KnownValue) = 0;

  /// The known state encoding in an integer of type base_t.
  base_t Known = getWorstState();

  /// The assumed state encoding in an integer of type base_t.
  base_t Assumed = getBestState();
};

/// Specialization of the integer state for a bit-wise encoding.
template <typename base_ty = uint32_t, base_ty BestState = ~base_ty(0),
          base_ty WorstState = 0>
struct BitIntegerState
    : public IntegerStateBase<base_ty, BestState, WorstState> {
  using base_t = base_ty;

  /// Return true if the bits set in \p BitsEncoding are "known bits".
  bool isKnown(base_t BitsEncoding) const {
    return (this->Known & BitsEncoding) == BitsEncoding;
  }

  /// Return true if the bits set in \p BitsEncoding are "assumed bits".
  bool isAssumed(base_t BitsEncoding) const {
    return (this->Assumed & BitsEncoding) == BitsEncoding;
  }

  /// Add the bits in \p BitsEncoding to the "known bits".
  BitIntegerState &addKnownBits(base_t Bits) {
    // Make sure we never miss any "known bits".
    this->Assumed |= Bits;
    this->Known |= Bits;
    return *this;
  }

  /// Remove the bits in \p BitsEncoding from the "assumed bits" if not known.
  BitIntegerState &removeAssumedBits(base_t BitsEncoding) {
    return intersectAssumedBits(~BitsEncoding);
  }

  /// Remove the bits in \p BitsEncoding from the "known bits".
  BitIntegerState &removeKnownBits(base_t BitsEncoding) {
    this->Known = (this->Known & ~BitsEncoding);
    return *this;
  }

  /// Keep only "assumed bits" also set in \p BitsEncoding but all known ones.
  BitIntegerState &intersectAssumedBits(base_t BitsEncoding) {
    // Make sure we never loose any "known bits".
    this->Assumed = (this->Assumed & BitsEncoding) | this->Known;
    return *this;
  }

private:
  void handleNewAssumedValue(base_t Value) override {
    intersectAssumedBits(Value);
  }
  void handleNewKnownValue(base_t Value) override { addKnownBits(Value); }
  void joinOR(base_t AssumedValue, base_t KnownValue) override {
    this->Known |= KnownValue;
    this->Assumed |= AssumedValue;
  }
  void joinAND(base_t AssumedValue, base_t KnownValue) override {
    this->Known &= KnownValue;
    this->Assumed &= AssumedValue;
  }
};

/// Specialization of the integer state for an increasing value, hence ~0u is
/// the best state and 0 the worst.
template <typename base_ty = uint32_t, base_ty BestState = ~base_ty(0),
          base_ty WorstState = 0>
struct IncIntegerState
    : public IntegerStateBase<base_ty, BestState, WorstState> {
  using super = IntegerStateBase<base_ty, BestState, WorstState>;
  using base_t = base_ty;

  IncIntegerState() : super() {}
  IncIntegerState(base_t Assumed) : super(Assumed) {}

  /// Return the best possible representable state.
  static constexpr base_t getBestState() { return BestState; }
  static constexpr base_t
  getBestState(const IncIntegerState<base_ty, BestState, WorstState> &) {
    return getBestState();
  }

  /// Take minimum of assumed and \p Value.
  IncIntegerState &takeAssumedMinimum(base_t Value) {
    // Make sure we never loose "known value".
    this->Assumed = std::max(std::min(this->Assumed, Value), this->Known);
    return *this;
  }

  /// Take maximum of known and \p Value.
  IncIntegerState &takeKnownMaximum(base_t Value) {
    // Make sure we never loose "known value".
    this->Assumed = std::max(Value, this->Assumed);
    this->Known = std::max(Value, this->Known);
    return *this;
  }

private:
  void handleNewAssumedValue(base_t Value) override {
    takeAssumedMinimum(Value);
  }
  void handleNewKnownValue(base_t Value) override { takeKnownMaximum(Value); }
  void joinOR(base_t AssumedValue, base_t KnownValue) override {
    this->Known = std::max(this->Known, KnownValue);
    this->Assumed = std::max(this->Assumed, AssumedValue);
  }
  void joinAND(base_t AssumedValue, base_t KnownValue) override {
    this->Known = std::min(this->Known, KnownValue);
    this->Assumed = std::min(this->Assumed, AssumedValue);
  }
};

/// Specialization of the integer state for a decreasing value, hence 0 is the
/// best state and ~0u the worst.
template <typename base_ty = uint32_t>
struct DecIntegerState : public IntegerStateBase<base_ty, 0, ~base_ty(0)> {
  using base_t = base_ty;

  /// Take maximum of assumed and \p Value.
  DecIntegerState &takeAssumedMaximum(base_t Value) {
    // Make sure we never loose "known value".
    this->Assumed = std::min(std::max(this->Assumed, Value), this->Known);
    return *this;
  }

  /// Take minimum of known and \p Value.
  DecIntegerState &takeKnownMinimum(base_t Value) {
    // Make sure we never loose "known value".
    this->Assumed = std::min(Value, this->Assumed);
    this->Known = std::min(Value, this->Known);
    return *this;
  }

private:
  void handleNewAssumedValue(base_t Value) override {
    takeAssumedMaximum(Value);
  }
  void handleNewKnownValue(base_t Value) override { takeKnownMinimum(Value); }
  void joinOR(base_t AssumedValue, base_t KnownValue) override {
    this->Assumed = std::min(this->Assumed, KnownValue);
    this->Assumed = std::min(this->Assumed, AssumedValue);
  }
  void joinAND(base_t AssumedValue, base_t KnownValue) override {
    this->Assumed = std::max(this->Assumed, KnownValue);
    this->Assumed = std::max(this->Assumed, AssumedValue);
  }
};

/// Simple wrapper for a single bit (boolean) state.
struct BooleanState : public IntegerStateBase<bool, 1, 0> {
  using super = IntegerStateBase<bool, 1, 0>;
  using base_t = IntegerStateBase::base_t;

  BooleanState() : super() {}
  BooleanState(base_t Assumed) : super(Assumed) {}

  /// Set the assumed value to \p Value but never below the known one.
  void setAssumed(bool Value) { Assumed &= (Known | Value); }

  /// Set the known and asssumed value to \p Value.
  void setKnown(bool Value) {
    Known |= Value;
    Assumed |= Value;
  }

  /// Return true if the state is assumed to hold.
  bool isAssumed() const { return getAssumed(); }

  /// Return true if the state is known to hold.
  bool isKnown() const { return getKnown(); }

private:
  void handleNewAssumedValue(base_t Value) override {
    if (!Value)
      Assumed = Known;
  }
  void handleNewKnownValue(base_t Value) override {
    if (Value)
      Known = (Assumed = Value);
  }
  void joinOR(base_t AssumedValue, base_t KnownValue) override {
    Known |= KnownValue;
    Assumed |= AssumedValue;
  }
  void joinAND(base_t AssumedValue, base_t KnownValue) override {
    Known &= KnownValue;
    Assumed &= AssumedValue;
  }
};

/// State for an integer range.
struct IntegerRangeState : public AbstractState {

  /// Bitwidth of the associated value.
  uint32_t BitWidth;

  /// State representing assumed range, initially set to empty.
  ConstantRange Assumed;

  /// State representing known range, initially set to [-inf, inf].
  ConstantRange Known;

  IntegerRangeState(uint32_t BitWidth)
      : BitWidth(BitWidth), Assumed(ConstantRange::getEmpty(BitWidth)),
        Known(ConstantRange::getFull(BitWidth)) {}

  IntegerRangeState(const ConstantRange &CR)
      : BitWidth(CR.getBitWidth()), Assumed(CR),
        Known(getWorstState(CR.getBitWidth())) {}

  /// Return the worst possible representable state.
  static ConstantRange getWorstState(uint32_t BitWidth) {
    return ConstantRange::getFull(BitWidth);
  }

  /// Return the best possible representable state.
  static ConstantRange getBestState(uint32_t BitWidth) {
    return ConstantRange::getEmpty(BitWidth);
  }
  static ConstantRange getBestState(const IntegerRangeState &IRS) {
    return getBestState(IRS.getBitWidth());
  }

  /// Return associated values' bit width.
  uint32_t getBitWidth() const { return BitWidth; }

  /// See AbstractState::isValidState()
  bool isValidState() const override {
    return BitWidth > 0 && !Assumed.isFullSet();
  }

  /// See AbstractState::isAtFixpoint()
  bool isAtFixpoint() const override { return Assumed == Known; }

  /// See AbstractState::indicateOptimisticFixpoint(...)
  ChangeStatus indicateOptimisticFixpoint() override {
    Known = Assumed;
    return ChangeStatus::CHANGED;
  }

  /// See AbstractState::indicatePessimisticFixpoint(...)
  ChangeStatus indicatePessimisticFixpoint() override {
    Assumed = Known;
    return ChangeStatus::CHANGED;
  }

  /// Return the known state encoding
  ConstantRange getKnown() const { return Known; }

  /// Return the assumed state encoding.
  ConstantRange getAssumed() const { return Assumed; }

  /// Unite assumed range with the passed state.
  void unionAssumed(const ConstantRange &R) {
    // Don't loose a known range.
    Assumed = Assumed.unionWith(R).intersectWith(Known);
  }

  /// See IntegerRangeState::unionAssumed(..).
  void unionAssumed(const IntegerRangeState &R) {
    unionAssumed(R.getAssumed());
  }

  /// Unite known range with the passed state.
  void unionKnown(const ConstantRange &R) {
    // Don't loose a known range.
    Known = Known.unionWith(R);
    Assumed = Assumed.unionWith(Known);
  }

  /// See IntegerRangeState::unionKnown(..).
  void unionKnown(const IntegerRangeState &R) { unionKnown(R.getKnown()); }

  /// Intersect known range with the passed state.
  void intersectKnown(const ConstantRange &R) {
    Assumed = Assumed.intersectWith(R);
    Known = Known.intersectWith(R);
  }

  /// See IntegerRangeState::intersectKnown(..).
  void intersectKnown(const IntegerRangeState &R) {
    intersectKnown(R.getKnown());
  }

  /// Equality for IntegerRangeState.
  bool operator==(const IntegerRangeState &R) const {
    return getAssumed() == R.getAssumed() && getKnown() == R.getKnown();
  }

  /// "Clamp" this state with \p R. The result is subtype dependent but it is
  /// intended that only information assumed in both states will be assumed in
  /// this one afterwards.
  IntegerRangeState operator^=(const IntegerRangeState &R) {
    // NOTE: `^=` operator seems like `intersect` but in this case, we need to
    // take `union`.
    unionAssumed(R);
    return *this;
  }

  IntegerRangeState operator&=(const IntegerRangeState &R) {
    // NOTE: `&=` operator seems like `intersect` but in this case, we need to
    // take `union`.
    unionKnown(R);
    unionAssumed(R);
    return *this;
  }
};
/// Helper struct necessary as the modular build fails if the virtual method
/// IRAttribute::manifest is defined in the Attributor.cpp.
struct IRAttributeManifest {
  static ChangeStatus manifestAttrs(Attributor &A, const IRPosition &IRP,
                                    const ArrayRef<Attribute> &DeducedAttrs);
};

/// Helper to tie a abstract state implementation to an abstract attribute.
template <typename StateTy, typename BaseType, class... Ts>
struct StateWrapper : public BaseType, public StateTy {
  /// Provide static access to the type of the state.
  using StateType = StateTy;

  StateWrapper(const IRPosition &IRP, Ts... Args)
      : BaseType(IRP), StateTy(Args...) {}

  /// See AbstractAttribute::getState(...).
  StateType &getState() override { return *this; }

  /// See AbstractAttribute::getState(...).
  const StateType &getState() const override { return *this; }
};

/// Helper class that provides common functionality to manifest IR attributes.
template <Attribute::AttrKind AK, typename BaseType>
struct IRAttribute : public BaseType {
  IRAttribute(const IRPosition &IRP) : BaseType(IRP) {}

  /// See AbstractAttribute::initialize(...).
  virtual void initialize(Attributor &A) override {
    const IRPosition &IRP = this->getIRPosition();
    if (isa<UndefValue>(IRP.getAssociatedValue()) ||
        this->hasAttr(getAttrKind(), /* IgnoreSubsumingPositions */ false,
                      &A)) {
      this->getState().indicateOptimisticFixpoint();
      return;
    }

    bool IsFnInterface = IRP.isFnInterfaceKind();
    const Function *FnScope = IRP.getAnchorScope();
    // TODO: Not all attributes require an exact definition. Find a way to
    //       enable deduction for some but not all attributes in case the
    //       definition might be changed at runtime, see also
    //       http://lists.llvm.org/pipermail/llvm-dev/2018-February/121275.html.
    // TODO: We could always determine abstract attributes and if sufficient
    //       information was found we could duplicate the functions that do not
    //       have an exact definition.
    if (IsFnInterface && (!FnScope || !A.isFunctionIPOAmendable(*FnScope)))
      this->getState().indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    if (isa<UndefValue>(this->getIRPosition().getAssociatedValue()))
      return ChangeStatus::UNCHANGED;
    SmallVector<Attribute, 4> DeducedAttrs;
    getDeducedAttributes(this->getAnchorValue().getContext(), DeducedAttrs);
    return IRAttributeManifest::manifestAttrs(A, this->getIRPosition(),
                                              DeducedAttrs);
  }

  /// Return the kind that identifies the abstract attribute implementation.
  Attribute::AttrKind getAttrKind() const { return AK; }

  /// Return the deduced attributes in \p Attrs.
  virtual void getDeducedAttributes(LLVMContext &Ctx,
                                    SmallVectorImpl<Attribute> &Attrs) const {
    Attrs.emplace_back(Attribute::get(Ctx, getAttrKind()));
  }
};

/// Forward declarations of output streams for debug purposes.
///
///{
template <typename base_ty, base_ty BestState, base_ty WorstState>
raw_ostream &
operator<<(raw_ostream &OS,
           const IntegerStateBase<base_ty, BestState, WorstState> &S) {
  return OS << "(" << S.getKnown() << "-" << S.getAssumed() << ")"
            << static_cast<const AbstractState &>(S);
}
raw_ostream &operator<<(raw_ostream &OS, const IntegerRangeState &State);
///}

/// ----------------------------------------------------------------------------
///                       Abstract Attribute Classes
/// ----------------------------------------------------------------------------

/// Helper function to clamp a state \p S of type \p StateType with the
/// information in \p R and indicate/return if \p S did change (as-in update is
/// required to be run again).
template <typename StateType>
ChangeStatus clampStateAndIndicateChange(StateType &S, const StateType &R) {
  auto Assumed = S.getAssumed();
  S ^= R;
  return Assumed == S.getAssumed() ? ChangeStatus::UNCHANGED
                                   : ChangeStatus::CHANGED;
}

template <typename BaseAA>
struct AACallGraphBottomUpCheckerFunction : public BaseAA {
  AACallGraphBottomUpCheckerFunction(const IRPosition &IRP, Attributor &A)
      : BaseAA(IRP, A) {}

  using CallbackTy = std::function<bool(Instruction &)>;

  virtual CallbackTy checkAllCallLikeInstructions(Attributor &) {
    return nullptr;
  }

  virtual CallbackTy checkAllReadWriteInstructions(Attributor &) {
    return nullptr;
  }

  virtual CallbackTy checkAllOpcodes(Attributor &,
                                     SmallVectorImpl<unsigned> &Opcodes) {
    return nullptr;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    const auto &S = this->getState();
    if (const auto &CB = this->checkAllCallLikeInstructions(A))
      if (!A.checkForAllCallLikeInstructions(CB, *this))
        return this->indicatePessimisticFixpoint();
    if (const auto &CB = this->checkAllReadWriteInstructions(A))
      if (!A.checkForAllReadWriteInstructions(CB, *this))
        return this->indicatePessimisticFixpoint();
    SmallVector<unsigned> Opcodes;
    if (const auto &CB = this->checkAllOpcodes(A, Opcodes))
      if (!A.checkForAllInstructions(CB, *this, Opcodes))
        return this->indicatePessimisticFixpoint();
    return S == this->getState() ? ChangeStatus::UNCHANGED
                                 : ChangeStatus::CHANGED;
  }
};

template <typename BaseAA>
struct AACallGraphBottomUpCheckerCallSite : public BaseAA {
  AACallGraphBottomUpCheckerCallSite(const IRPosition &IRP, Attributor &A)
      : BaseAA(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    BaseAA::initialize(A);
    Function *F = this->getAssociatedFunction();
    if (!F || F->isDeclaration())
      this->indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Function *F = this->getAssociatedFunction();
    const IRPosition &FnPos = IRPosition::function(*F);
    auto &FnAA = A.getAAFor<BaseAA>(*this, FnPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(this->getState(), FnAA.getState());
  }
};

/// An abstract attribute for the returned values of a function.
struct AAReturnedValues
    : public IRAttribute<Attribute::Returned, AbstractAttribute> {
  AAReturnedValues(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// Return an assumed unique return value if a single candidate is found. If
  /// there cannot be one, return a nullptr. If it is not clear yet, return the
  /// Optional::NoneType.
  Optional<Value *> getAssumedUniqueReturnValue(Attributor &A) const;

  /// Check \p Pred on all returned values.
  ///
  /// This method will evaluate \p Pred on returned values and return
  /// true if (1) all returned values are known, and (2) \p Pred returned true
  /// for all returned values.
  ///
  /// Note: Unlike the Attributor::checkForAllReturnedValuesAndReturnInsts
  /// method, this one will not filter dead return instructions.
  virtual bool checkForAllReturnedValuesAndReturnInsts(
      function_ref<bool(Value &, const SmallSetVector<ReturnInst *, 4> &)> Pred)
      const = 0;

  using iterator =
      MapVector<Value *, SmallSetVector<ReturnInst *, 4>>::iterator;
  using const_iterator =
      MapVector<Value *, SmallSetVector<ReturnInst *, 4>>::const_iterator;
  virtual llvm::iterator_range<iterator> returned_values() = 0;
  virtual llvm::iterator_range<const_iterator> returned_values() const = 0;

  virtual size_t getNumReturnValues() const = 0;
  virtual const SmallSetVector<CallBase *, 4> &getUnresolvedCalls() const = 0;

  /// Create an abstract attribute view for the position \p IRP.
  static AAReturnedValues &createForPosition(const IRPosition &IRP,
                                             Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAReturnedValues"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAReturnedValues
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

struct AANoUnwind
    : public IRAttribute<Attribute::NoUnwind,
                         StateWrapper<BooleanState, AbstractAttribute>> {
  AANoUnwind(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// Returns true if nounwind is assumed.
  bool isAssumedNoUnwind() const { return getAssumed(); }

  /// Returns true if nounwind is known.
  bool isKnownNoUnwind() const { return getKnown(); }

  /// Create an abstract attribute view for the position \p IRP.
  static AANoUnwind &createForPosition(const IRPosition &IRP, Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AANoUnwind"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AANoUnwind
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

struct AANoIntrinsicCall
    : public StateWrapper<BitIntegerState<uint32_t>, AbstractAttribute> {
  using Base = StateWrapper<BitIntegerState<uint32_t>, AbstractAttribute>;
  AANoIntrinsicCall(const IRPosition &IRP, Attributor &A,
                    ArrayRef<Intrinsic::ID> IIDs)
      : Base(IRP) {
    for (Intrinsic::ID ID : IIDs)
      IIDMap[ID] = IIDMap.size();
  }

  /// Returns true if "nosync" is assumed.
  bool isAssumedNotCalling(Intrinsic::ID ID) const {
    return isAssumed(getBitMaskForIntrinsic(ID));
  }

  /// Returns true if "nosync" is known.
  bool isKnownNotCalling(Intrinsic::ID ID) const {
    return isKnown(getBitMaskForIntrinsic(ID));
  }

  /// Create an abstract attribute view for the position \p IRP.
  static AANoIntrinsicCall &createForPosition(const IRPosition &IRP,
                                              Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AANoIntrinsicCall"; }

  /// See AbstractState::getAsStr().
  const std::string getAsStr() const override { return "TODO"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AANoIntrinsicCall
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;

protected:
  uint32_t getBitMaskForIntrinsic(Intrinsic::ID ID) const {
    return IIDMap.lookup(ID);
  }

  SmallMapVector<Intrinsic::ID, uint32_t, 16> IIDMap;
};

struct AANoSync
    : public IRAttribute<Attribute::NoSync,
                         StateWrapper<BooleanState, AbstractAttribute>> {
  AANoSync(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// Returns true if "nosync" is assumed.
  bool isAssumedNoSync() const { return getAssumed(); }

  /// Returns true if "nosync" is known.
  bool isKnownNoSync() const { return getKnown(); }

  /// Create an abstract attribute view for the position \p IRP.
  static AANoSync &createForPosition(const IRPosition &IRP, Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AANoSync"; }

  /// See AbstractState::getAsStr().
  const std::string getAsStr() const override {
    return getAssumed() ? "nosync" : "may-sync";
  }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AANoSync
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract interface for all nonnull attributes.
struct AANonNull
    : public IRAttribute<Attribute::NonNull,
                         StateWrapper<BooleanState, AbstractAttribute>> {
  AANonNull(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// Return true if we assume that the underlying value is nonnull.
  bool isAssumedNonNull() const { return getAssumed(); }

  /// Return true if we know that underlying value is nonnull.
  bool isKnownNonNull() const { return getKnown(); }

  /// Create an abstract attribute view for the position \p IRP.
  static AANonNull &createForPosition(const IRPosition &IRP, Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AANonNull"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AANonNull
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract attribute for norecurse.
struct AANoRecurse
    : public IRAttribute<Attribute::NoRecurse,
                         StateWrapper<BooleanState, AbstractAttribute>> {
  AANoRecurse(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// Return true if "norecurse" is assumed.
  bool isAssumedNoRecurse() const { return getAssumed(); }

  /// Return true if "norecurse" is known.
  bool isKnownNoRecurse() const { return getKnown(); }

  /// Create an abstract attribute view for the position \p IRP.
  static AANoRecurse &createForPosition(const IRPosition &IRP, Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AANoRecurse"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AANoRecurse
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract attribute for willreturn.
struct AAWillReturn
    : public IRAttribute<Attribute::WillReturn,
                         StateWrapper<BooleanState, AbstractAttribute>> {
  AAWillReturn(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// Return true if "willreturn" is assumed.
  bool isAssumedWillReturn() const { return getAssumed(); }

  /// Return true if "willreturn" is known.
  bool isKnownWillReturn() const { return getKnown(); }

  /// Create an abstract attribute view for the position \p IRP.
  static AAWillReturn &createForPosition(const IRPosition &IRP, Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAWillReturn"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AAWillReturn
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract attribute for undefined behavior.
struct AAUndefinedBehavior
    : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAUndefinedBehavior(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Return true if "undefined behavior" is assumed.
  bool isAssumedToCauseUB() const { return getAssumed(); }

  /// Return true if "undefined behavior" is assumed for a specific instruction.
  virtual bool isAssumedToCauseUB(Instruction *I) const = 0;

  /// Return true if "undefined behavior" is known.
  bool isKnownToCauseUB() const { return getKnown(); }

  /// Return true if "undefined behavior" is known for a specific instruction.
  virtual bool isKnownToCauseUB(Instruction *I) const = 0;

  /// Create an abstract attribute view for the position \p IRP.
  static AAUndefinedBehavior &createForPosition(const IRPosition &IRP,
                                                Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAUndefinedBehavior"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAUndefineBehavior
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract interface to determine reachability of point A to B.
struct AAReachability : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAReachability(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Returns true if 'From' instruction is assumed to reach, 'To' instruction.
  /// Users should provide two positions they are interested in, and the class
  /// determines (and caches) reachability.
  bool isAssumedReachable(Attributor &A, const Instruction &From,
                          const Instruction &To) const {
    return A.getInfoCache().getPotentiallyReachable(From, To);
  }

  /// Returns true if 'From' instruction is known to reach, 'To' instruction.
  /// Users should provide two positions they are interested in, and the class
  /// determines (and caches) reachability.
  bool isKnownReachable(Attributor &A, const Instruction &From,
                        const Instruction &To) const {
    return A.getInfoCache().getPotentiallyReachable(From, To);
  }

  /// Create an abstract attribute view for the position \p IRP.
  static AAReachability &createForPosition(const IRPosition &IRP,
                                           Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAReachability"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAReachability
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract interface for all noalias attributes.
struct AANoAlias
    : public IRAttribute<Attribute::NoAlias,
                         StateWrapper<BooleanState, AbstractAttribute>> {
  AANoAlias(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// Return true if we assume that the underlying value is alias.
  bool isAssumedNoAlias() const { return getAssumed(); }

  /// Return true if we know that underlying value is noalias.
  bool isKnownNoAlias() const { return getKnown(); }

  /// Create an abstract attribute view for the position \p IRP.
  static AANoAlias &createForPosition(const IRPosition &IRP, Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AANoAlias"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AANoAlias
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An AbstractAttribute for nofree.
struct AANoFree
    : public IRAttribute<Attribute::NoFree,
                         StateWrapper<BooleanState, AbstractAttribute>> {
  AANoFree(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// Return true if "nofree" is assumed.
  bool isAssumedNoFree() const { return getAssumed(); }

  /// Return true if "nofree" is known.
  bool isKnownNoFree() const { return getKnown(); }

  /// Create an abstract attribute view for the position \p IRP.
  static AANoFree &createForPosition(const IRPosition &IRP, Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AANoFree"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AANoFree
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An AbstractAttribute for noreturn.
struct AANoReturn
    : public IRAttribute<Attribute::NoReturn,
                         StateWrapper<BooleanState, AbstractAttribute>> {
  AANoReturn(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// Return true if the underlying object is assumed to never return.
  bool isAssumedNoReturn() const { return getAssumed(); }

  /// Return true if the underlying object is known to never return.
  bool isKnownNoReturn() const { return getKnown(); }

  /// Create an abstract attribute view for the position \p IRP.
  static AANoReturn &createForPosition(const IRPosition &IRP, Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AANoReturn"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AANoReturn
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract interface for liveness abstract attribute.
struct AAIsDead : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAIsDead(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

protected:
  /// The query functions are protected such that other attributes need to go
  /// through the Attributor interfaces: `Attributor::isAssumedDead(...)`

  /// Returns true if the underlying value is assumed dead.
  virtual bool isAssumedDead() const = 0;

  /// Returns true if the underlying value is known dead.
  virtual bool isKnownDead() const = 0;

  /// Returns true if \p BB is assumed dead.
  virtual bool isAssumedDead(const BasicBlock *BB) const = 0;

  /// Returns true if \p BB is known dead.
  virtual bool isKnownDead(const BasicBlock *BB) const = 0;

  /// Returns true if \p I is assumed dead.
  virtual bool isAssumedDead(const Instruction *I) const = 0;

  /// Returns true if \p I is known dead.
  virtual bool isKnownDead(const Instruction *I) const = 0;

  /// This method is used to check if at least one instruction in a collection
  /// of instructions is live.
  template <typename T> bool isLiveInstSet(T begin, T end) const {
    for (const auto &I : llvm::make_range(begin, end)) {
      assert(I->getFunction() == getIRPosition().getAssociatedFunction() &&
             "Instruction must be in the same anchor scope function.");

      if (!isAssumedDead(I))
        return true;
    }

    return false;
  }

public:
  /// Create an abstract attribute view for the position \p IRP.
  static AAIsDead &createForPosition(const IRPosition &IRP, Attributor &A);

  /// Determine if \p F might catch asynchronous exceptions.
  static bool mayCatchAsynchronousExceptions(const Function &F) {
    return F.hasPersonalityFn() && !canSimplifyInvokeNoUnwind(&F);
  }

  /// Return if the edge from \p From BB to \p To BB is assumed dead.
  /// This is specifically useful in AAReachability.
  virtual bool isEdgeDead(const BasicBlock *From, const BasicBlock *To) const {
    return false;
  }

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAIsDead"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AAIsDead
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;

  friend struct Attributor;
};

/// State for dereferenceable attribute
struct DerefState : AbstractState {

  static DerefState getBestState() { return DerefState(); }
  static DerefState getBestState(const DerefState &) { return getBestState(); }

  /// Return the worst possible representable state.
  static DerefState getWorstState() {
    DerefState DS;
    DS.indicatePessimisticFixpoint();
    return DS;
  }
  static DerefState getWorstState(const DerefState &) {
    return getWorstState();
  }

  /// State representing for dereferenceable bytes.
  IncIntegerState<> DerefBytesState;

  /// Map representing for accessed memory offsets and sizes.
  /// A key is Offset and a value is size.
  /// If there is a load/store instruction something like,
  ///   p[offset] = v;
  /// (offset, sizeof(v)) will be inserted to this map.
  /// std::map is used because we want to iterate keys in ascending order.
  std::map<int64_t, uint64_t> AccessedBytesMap;

  /// Helper function to calculate dereferenceable bytes from current known
  /// bytes and accessed bytes.
  ///
  /// int f(int *A){
  ///    *A = 0;
  ///    *(A+2) = 2;
  ///    *(A+1) = 1;
  ///    *(A+10) = 10;
  /// }
  /// ```
  /// In that case, AccessedBytesMap is `{0:4, 4:4, 8:4, 40:4}`.
  /// AccessedBytesMap is std::map so it is iterated in accending order on
  /// key(Offset). So KnownBytes will be updated like this:
  ///
  /// |Access | KnownBytes
  /// |(0, 4)| 0 -> 4
  /// |(4, 4)| 4 -> 8
  /// |(8, 4)| 8 -> 12
  /// |(40, 4) | 12 (break)
  void computeKnownDerefBytesFromAccessedMap() {
    int64_t KnownBytes = DerefBytesState.getKnown();
    for (auto &Access : AccessedBytesMap) {
      if (KnownBytes < Access.first)
        break;
      KnownBytes = std::max(KnownBytes, Access.first + (int64_t)Access.second);
    }

    DerefBytesState.takeKnownMaximum(KnownBytes);
  }

  /// State representing that whether the value is globaly dereferenceable.
  BooleanState GlobalState;

  /// See AbstractState::isValidState()
  bool isValidState() const override { return DerefBytesState.isValidState(); }

  /// See AbstractState::isAtFixpoint()
  bool isAtFixpoint() const override {
    return !isValidState() ||
           (DerefBytesState.isAtFixpoint() && GlobalState.isAtFixpoint());
  }

  /// See AbstractState::indicateOptimisticFixpoint(...)
  ChangeStatus indicateOptimisticFixpoint() override {
    DerefBytesState.indicateOptimisticFixpoint();
    GlobalState.indicateOptimisticFixpoint();
    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractState::indicatePessimisticFixpoint(...)
  ChangeStatus indicatePessimisticFixpoint() override {
    DerefBytesState.indicatePessimisticFixpoint();
    GlobalState.indicatePessimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  /// Update known dereferenceable bytes.
  void takeKnownDerefBytesMaximum(uint64_t Bytes) {
    DerefBytesState.takeKnownMaximum(Bytes);

    // Known bytes might increase.
    computeKnownDerefBytesFromAccessedMap();
  }

  /// Update assumed dereferenceable bytes.
  void takeAssumedDerefBytesMinimum(uint64_t Bytes) {
    DerefBytesState.takeAssumedMinimum(Bytes);
  }

  /// Add accessed bytes to the map.
  void addAccessedBytes(int64_t Offset, uint64_t Size) {
    uint64_t &AccessedBytes = AccessedBytesMap[Offset];
    AccessedBytes = std::max(AccessedBytes, Size);

    // Known bytes might increase.
    computeKnownDerefBytesFromAccessedMap();
  }

  /// Equality for DerefState.
  bool operator==(const DerefState &R) const {
    return this->DerefBytesState == R.DerefBytesState &&
           this->GlobalState == R.GlobalState;
  }

  /// Inequality for DerefState.
  bool operator!=(const DerefState &R) const { return !(*this == R); }

  /// See IntegerStateBase::operator^=
  DerefState operator^=(const DerefState &R) {
    DerefBytesState ^= R.DerefBytesState;
    GlobalState ^= R.GlobalState;
    return *this;
  }

  /// See IntegerStateBase::operator+=
  DerefState operator+=(const DerefState &R) {
    DerefBytesState += R.DerefBytesState;
    GlobalState += R.GlobalState;
    return *this;
  }

  /// See IntegerStateBase::operator&=
  DerefState operator&=(const DerefState &R) {
    DerefBytesState &= R.DerefBytesState;
    GlobalState &= R.GlobalState;
    return *this;
  }

  /// See IntegerStateBase::operator|=
  DerefState operator|=(const DerefState &R) {
    DerefBytesState |= R.DerefBytesState;
    GlobalState |= R.GlobalState;
    return *this;
  }

protected:
  const AANonNull *NonNullAA = nullptr;
};

/// An abstract interface for all dereferenceable attribute.
struct AADereferenceable
    : public IRAttribute<Attribute::Dereferenceable,
                         StateWrapper<DerefState, AbstractAttribute>> {
  AADereferenceable(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// Return true if we assume that the underlying value is nonnull.
  bool isAssumedNonNull() const {
    return NonNullAA && NonNullAA->isAssumedNonNull();
  }

  /// Return true if we know that the underlying value is nonnull.
  bool isKnownNonNull() const {
    return NonNullAA && NonNullAA->isKnownNonNull();
  }

  /// Return true if we assume that underlying value is
  /// dereferenceable(_or_null) globally.
  bool isAssumedGlobal() const { return GlobalState.getAssumed(); }

  /// Return true if we know that underlying value is
  /// dereferenceable(_or_null) globally.
  bool isKnownGlobal() const { return GlobalState.getKnown(); }

  /// Return assumed dereferenceable bytes.
  uint32_t getAssumedDereferenceableBytes() const {
    return DerefBytesState.getAssumed();
  }

  /// Return known dereferenceable bytes.
  uint32_t getKnownDereferenceableBytes() const {
    return DerefBytesState.getKnown();
  }

  /// Create an abstract attribute view for the position \p IRP.
  static AADereferenceable &createForPosition(const IRPosition &IRP,
                                              Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AADereferenceable"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AADereferenceable
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

using AAAlignmentStateType =
    IncIntegerState<uint32_t, Value::MaximumAlignment, 1>;
/// An abstract interface for all align attributes.
struct AAAlign : public IRAttribute<
                     Attribute::Alignment,
                     StateWrapper<AAAlignmentStateType, AbstractAttribute>> {
  AAAlign(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// Return assumed alignment.
  unsigned getAssumedAlign() const { return getAssumed(); }

  /// Return known alignment.
  unsigned getKnownAlign() const { return getKnown(); }

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAAlign"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AAAlign
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Create an abstract attribute view for the position \p IRP.
  static AAAlign &createForPosition(const IRPosition &IRP, Attributor &A);

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract interface for all nocapture attributes.
struct AANoCapture
    : public IRAttribute<
          Attribute::NoCapture,
          StateWrapper<BitIntegerState<uint16_t, 7, 0>, AbstractAttribute>> {
  AANoCapture(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// State encoding bits. A set bit in the state means the property holds.
  /// NO_CAPTURE is the best possible state, 0 the worst possible state.
  enum {
    NOT_CAPTURED_IN_MEM = 1 << 0,
    NOT_CAPTURED_IN_INT = 1 << 1,
    NOT_CAPTURED_IN_RET = 1 << 2,

    /// If we do not capture the value in memory or through integers we can only
    /// communicate it back as a derived pointer.
    NO_CAPTURE_MAYBE_RETURNED = NOT_CAPTURED_IN_MEM | NOT_CAPTURED_IN_INT,

    /// If we do not capture the value in memory, through integers, or as a
    /// derived pointer we know it is not captured.
    NO_CAPTURE =
        NOT_CAPTURED_IN_MEM | NOT_CAPTURED_IN_INT | NOT_CAPTURED_IN_RET,
  };

  /// Return true if we know that the underlying value is not captured in its
  /// respective scope.
  bool isKnownNoCapture() const { return isKnown(NO_CAPTURE); }

  /// Return true if we assume that the underlying value is not captured in its
  /// respective scope.
  bool isAssumedNoCapture() const { return isAssumed(NO_CAPTURE); }

  /// Return true if we know that the underlying value is not captured in its
  /// respective scope but we allow it to escape through a "return".
  bool isKnownNoCaptureMaybeReturned() const {
    return isKnown(NO_CAPTURE_MAYBE_RETURNED);
  }

  /// Return true if we assume that the underlying value is not captured in its
  /// respective scope but we allow it to escape through a "return".
  bool isAssumedNoCaptureMaybeReturned() const {
    return isAssumed(NO_CAPTURE_MAYBE_RETURNED);
  }

  /// Create an abstract attribute view for the position \p IRP.
  static AANoCapture &createForPosition(const IRPosition &IRP, Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AANoCapture"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AANoCapture
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract interface for value simplify abstract attribute.
struct AAValueSimplify : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAValueSimplify(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Return an assumed simplified value if a single candidate is found. If
  /// there cannot be one, return original value. If it is not clear yet, return
  /// the Optional::NoneType.
  virtual Optional<Value *> getAssumedSimplifiedValue(Attributor &A) const = 0;

  /// Create an abstract attribute view for the position \p IRP.
  static AAValueSimplify &createForPosition(const IRPosition &IRP,
                                            Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAValueSimplify"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAValueSimplify
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

struct AAHeapToStack : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAHeapToStack(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Returns true if HeapToStack conversion is assumed to be possible.
  bool isAssumedHeapToStack() const { return getAssumed(); }

  /// Returns true if HeapToStack conversion is known to be possible.
  bool isKnownHeapToStack() const { return getKnown(); }

  /// Create an abstract attribute view for the position \p IRP.
  static AAHeapToStack &createForPosition(const IRPosition &IRP, Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAHeapToStack"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AAHeapToStack
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract interface for privatizability.
///
/// A pointer is privatizable if it can be replaced by a new, private one.
/// Privatizing pointer reduces the use count, interaction between unrelated
/// code parts.
///
/// In order for a pointer to be privatizable its value cannot be observed
/// (=nocapture), it is (for now) not written (=readonly & noalias), we know
/// what values are necessary to make the private copy look like the original
/// one, and the values we need can be loaded (=dereferenceable).
struct AAPrivatizablePtr
    : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAPrivatizablePtr(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Returns true if pointer privatization is assumed to be possible.
  bool isAssumedPrivatizablePtr() const { return getAssumed(); }

  /// Returns true if pointer privatization is known to be possible.
  bool isKnownPrivatizablePtr() const { return getKnown(); }

  /// Return the type we can choose for a private copy of the underlying
  /// value. None means it is not clear yet, nullptr means there is none.
  virtual Optional<Type *> getPrivatizableType() const = 0;

  /// Create an abstract attribute view for the position \p IRP.
  static AAPrivatizablePtr &createForPosition(const IRPosition &IRP,
                                              Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAPrivatizablePtr"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAPricatizablePtr
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract interface for memory access kind related attributes
/// (readnone/readonly/writeonly).
struct AAMemoryBehavior
    : public IRAttribute<
          Attribute::ReadNone,
          StateWrapper<BitIntegerState<uint8_t, 3>, AbstractAttribute>> {
  AAMemoryBehavior(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// State encoding bits. A set bit in the state means the property holds.
  /// BEST_STATE is the best possible state, 0 the worst possible state.
  enum {
    NO_READS = 1 << 0,
    NO_WRITES = 1 << 1,
    NO_ACCESSES = NO_READS | NO_WRITES,

    BEST_STATE = NO_ACCESSES,
  };
  static_assert(BEST_STATE == getBestState(), "Unexpected BEST_STATE value");

  /// Return true if we know that the underlying value is not read or accessed
  /// in its respective scope.
  bool isKnownReadNone() const { return isKnown(NO_ACCESSES); }

  /// Return true if we assume that the underlying value is not read or accessed
  /// in its respective scope.
  bool isAssumedReadNone() const { return isAssumed(NO_ACCESSES); }

  /// Return true if we know that the underlying value is not accessed
  /// (=written) in its respective scope.
  bool isKnownReadOnly() const { return isKnown(NO_WRITES); }

  /// Return true if we assume that the underlying value is not accessed
  /// (=written) in its respective scope.
  bool isAssumedReadOnly() const { return isAssumed(NO_WRITES); }

  /// Return true if we know that the underlying value is not read in its
  /// respective scope.
  bool isKnownWriteOnly() const { return isKnown(NO_READS); }

  /// Return true if we assume that the underlying value is not read in its
  /// respective scope.
  bool isAssumedWriteOnly() const { return isAssumed(NO_READS); }

  /// Create an abstract attribute view for the position \p IRP.
  static AAMemoryBehavior &createForPosition(const IRPosition &IRP,
                                             Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAMemoryBehavior"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAMemoryBehavior
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract interface for all memory location attributes
/// (readnone/argmemonly/inaccessiblememonly/inaccessibleorargmemonly).
struct AAMemoryLocation
    : public IRAttribute<
          Attribute::ReadNone,
          StateWrapper<BitIntegerState<uint32_t, 511>, AbstractAttribute>> {
  using MemoryLocationsKind = StateType::base_t;

  AAMemoryLocation(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// Encoding of different locations that could be accessed by a memory
  /// access.
  enum {
    ALL_LOCATIONS = 0,
    NO_LOCAL_MEM = 1 << 0,
    NO_CONST_MEM = 1 << 1,
    NO_GLOBAL_INTERNAL_MEM = 1 << 2,
    NO_GLOBAL_EXTERNAL_MEM = 1 << 3,
    NO_GLOBAL_MEM = NO_GLOBAL_INTERNAL_MEM | NO_GLOBAL_EXTERNAL_MEM,
    NO_ARGUMENT_MEM = 1 << 4,
    NO_INACCESSIBLE_MEM = 1 << 5,
    NO_MALLOCED_MEM = 1 << 6,
    NO_UNKOWN_MEM = 1 << 7,
    NO_LOCATIONS = NO_LOCAL_MEM | NO_CONST_MEM | NO_GLOBAL_INTERNAL_MEM |
                   NO_GLOBAL_EXTERNAL_MEM | NO_ARGUMENT_MEM |
                   NO_INACCESSIBLE_MEM | NO_MALLOCED_MEM | NO_UNKOWN_MEM,

    // Helper bit to track if we gave up or not.
    VALID_STATE = NO_LOCATIONS + 1,

    BEST_STATE = NO_LOCATIONS | VALID_STATE,
  };
  static_assert(BEST_STATE == getBestState(), "Unexpected BEST_STATE value");

  /// Return true if we know that the associated functions has no observable
  /// accesses.
  bool isKnownReadNone() const { return isKnown(NO_LOCATIONS); }

  /// Return true if we assume that the associated functions has no observable
  /// accesses.
  bool isAssumedReadNone() const {
    return isAssumed(NO_LOCATIONS) | isAssumedStackOnly();
  }

  /// Return true if we know that the associated functions has at most
  /// local/stack accesses.
  bool isKnowStackOnly() const {
    return isKnown(inverseLocation(NO_LOCAL_MEM, true, true));
  }

  /// Return true if we assume that the associated functions has at most
  /// local/stack accesses.
  bool isAssumedStackOnly() const {
    return isAssumed(inverseLocation(NO_LOCAL_MEM, true, true));
  }

  /// Return true if we know that the underlying value will only access
  /// inaccesible memory only (see Attribute::InaccessibleMemOnly).
  bool isKnownInaccessibleMemOnly() const {
    return isKnown(inverseLocation(NO_INACCESSIBLE_MEM, true, true));
  }

  /// Return true if we assume that the underlying value will only access
  /// inaccesible memory only (see Attribute::InaccessibleMemOnly).
  bool isAssumedInaccessibleMemOnly() const {
    return isAssumed(inverseLocation(NO_INACCESSIBLE_MEM, true, true));
  }

  /// Return true if we know that the underlying value will only access
  /// argument pointees (see Attribute::ArgMemOnly).
  bool isKnownArgMemOnly() const {
    return isKnown(inverseLocation(NO_ARGUMENT_MEM, true, true));
  }

  /// Return true if we assume that the underlying value will only access
  /// argument pointees (see Attribute::ArgMemOnly).
  bool isAssumedArgMemOnly() const {
    return isAssumed(inverseLocation(NO_ARGUMENT_MEM, true, true));
  }

  /// Return true if we know that the underlying value will only access
  /// inaccesible memory or argument pointees (see
  /// Attribute::InaccessibleOrArgMemOnly).
  bool isKnownInaccessibleOrArgMemOnly() const {
    return isKnown(
        inverseLocation(NO_INACCESSIBLE_MEM | NO_ARGUMENT_MEM, true, true));
  }

  /// Return true if we assume that the underlying value will only access
  /// inaccesible memory or argument pointees (see
  /// Attribute::InaccessibleOrArgMemOnly).
  bool isAssumedInaccessibleOrArgMemOnly() const {
    return isAssumed(
        inverseLocation(NO_INACCESSIBLE_MEM | NO_ARGUMENT_MEM, true, true));
  }

  /// Return true if the underlying value may access memory through arguement
  /// pointers of the associated function, if any.
  bool mayAccessArgMem() const { return !isAssumed(NO_ARGUMENT_MEM); }

  /// Return true if only the memory locations specififed by \p MLK are assumed
  /// to be accessed by the associated function.
  bool isAssumedSpecifiedMemOnly(MemoryLocationsKind MLK) const {
    return isAssumed(MLK);
  }

  /// Return the locations that are assumed to be not accessed by the associated
  /// function, if any.
  MemoryLocationsKind getAssumedNotAccessedLocation() const {
    return getAssumed();
  }

  /// Return the inverse of location \p Loc, thus for NO_XXX the return
  /// describes ONLY_XXX. The flags \p AndLocalMem and \p AndConstMem determine
  /// if local (=stack) and constant memory are allowed as well. Most of the
  /// time we do want them to be included, e.g., argmemonly allows accesses via
  /// argument pointers or local or constant memory accesses.
  static MemoryLocationsKind
  inverseLocation(MemoryLocationsKind Loc, bool AndLocalMem, bool AndConstMem) {
    return NO_LOCATIONS & ~(Loc | (AndLocalMem ? NO_LOCAL_MEM : 0) |
                            (AndConstMem ? NO_CONST_MEM : 0));
  };

  /// Return the locations encoded by \p MLK as a readable string.
  static std::string getMemoryLocationsAsStr(MemoryLocationsKind MLK);

  /// Simple enum to distinguish read/write/read-write accesses.
  enum AccessKind {
    NONE = 0,
    READ = 1 << 0,
    WRITE = 1 << 1,
    READ_WRITE = READ | WRITE,
  };

  /// Check \p Pred on all accesses to the memory kinds specified by \p MLK.
  ///
  /// This method will evaluate \p Pred on all accesses (access instruction +
  /// underlying accessed memory pointer) and it will return true if \p Pred
  /// holds every time.
  virtual bool checkForAllAccessesToMemoryKind(
      function_ref<bool(const Instruction *, const Value *, AccessKind,
                        MemoryLocationsKind)>
          Pred,
      MemoryLocationsKind MLK) const = 0;

  /// Create an abstract attribute view for the position \p IRP.
  static AAMemoryLocation &createForPosition(const IRPosition &IRP,
                                             Attributor &A);

  /// See AbstractState::getAsStr().
  const std::string getAsStr() const override {
    return getMemoryLocationsAsStr(getAssumedNotAccessedLocation());
  }

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAMemoryLocation"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAMemoryLocation
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract interface for range value analysis.
struct AAValueConstantRange
    : public StateWrapper<IntegerRangeState, AbstractAttribute, uint32_t> {
  using Base = StateWrapper<IntegerRangeState, AbstractAttribute, uint32_t>;
  AAValueConstantRange(const IRPosition &IRP, Attributor &A)
      : Base(IRP, IRP.getAssociatedType()->getIntegerBitWidth()) {}

  /// See AbstractAttribute::getState(...).
  IntegerRangeState &getState() override { return *this; }
  const IntegerRangeState &getState() const override { return *this; }

  /// Create an abstract attribute view for the position \p IRP.
  static AAValueConstantRange &createForPosition(const IRPosition &IRP,
                                                 Attributor &A);

  /// Return an assumed range for the assocaited value a program point \p CtxI.
  /// If \p I is nullptr, simply return an assumed range.
  virtual ConstantRange
  getAssumedConstantRange(Attributor &A,
                          const Instruction *CtxI = nullptr) const = 0;

  /// Return a known range for the assocaited value at a program point \p CtxI.
  /// If \p I is nullptr, simply return a known range.
  virtual ConstantRange
  getKnownConstantRange(Attributor &A,
                        const Instruction *CtxI = nullptr) const = 0;

  /// Return an assumed constant for the assocaited value a program point \p
  /// CtxI.
  Optional<ConstantInt *>
  getAssumedConstantInt(Attributor &A,
                        const Instruction *CtxI = nullptr) const {
    ConstantRange RangeV = getAssumedConstantRange(A, CtxI);
    if (auto *C = RangeV.getSingleElement())
      return cast<ConstantInt>(
          ConstantInt::get(getAssociatedValue().getType(), *C));
    if (RangeV.isEmptySet())
      return llvm::None;
    return nullptr;
  }

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAValueConstantRange"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAValueConstantRange
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// A class for a set state.
/// The assumed boolean state indicates whether the corresponding set is full
/// set or not. If the assumed state is false, this is the worst state. The
/// worst state (invalid state) of set of potential values is when the set
/// contains every possible value (i.e. we cannot in any way limit the value
/// that the target position can take). That never happens naturally, we only
/// force it. As for the conditions under which we force it, see
/// AAPotentialValues.
template <typename MemberTy, typename KeyInfo = DenseMapInfo<MemberTy>>
struct PotentialValuesState : AbstractState {
  using SetTy = DenseSet<MemberTy, KeyInfo>;

  PotentialValuesState() : IsValidState(true), UndefIsContained(false) {}

  PotentialValuesState(bool IsValid)
      : IsValidState(IsValid), UndefIsContained(false) {}

  /// See AbstractState::isValidState(...)
  bool isValidState() const override { return IsValidState.isValidState(); }

  /// See AbstractState::isAtFixpoint(...)
  bool isAtFixpoint() const override { return IsValidState.isAtFixpoint(); }

  /// See AbstractState::indicatePessimisticFixpoint(...)
  ChangeStatus indicatePessimisticFixpoint() override {
    return IsValidState.indicatePessimisticFixpoint();
  }

  /// See AbstractState::indicateOptimisticFixpoint(...)
  ChangeStatus indicateOptimisticFixpoint() override {
    return IsValidState.indicateOptimisticFixpoint();
  }

  /// Return the assumed state
  PotentialValuesState &getAssumed() { return *this; }
  const PotentialValuesState &getAssumed() const { return *this; }

  /// Return this set. We should check whether this set is valid or not by
  /// isValidState() before calling this function.
  const SetTy &getAssumedSet() const {
    assert(isValidState() && "This set shoud not be used when it is invalid!");
    return Set;
  }

  /// Returns whether this state contains an undef value or not.
  bool undefIsContained() const {
    assert(isValidState() && "This flag shoud not be used when it is invalid!");
    return UndefIsContained;
  }

  bool operator==(const PotentialValuesState &RHS) const {
    if (isValidState() != RHS.isValidState())
      return false;
    if (!isValidState() && !RHS.isValidState())
      return true;
    if (undefIsContained() != RHS.undefIsContained())
      return false;
    return Set == RHS.getAssumedSet();
  }

  /// Maximum number of potential values to be tracked.
  /// This is set by -attributor-max-potential-values command line option
  static unsigned MaxPotentialValues;

  /// Return empty set as the best state of potential values.
  static PotentialValuesState getBestState() {
    return PotentialValuesState(true);
  }

  static PotentialValuesState getBestState(PotentialValuesState &PVS) {
    return getBestState();
  }

  /// Return full set as the worst state of potential values.
  static PotentialValuesState getWorstState() {
    return PotentialValuesState(false);
  }

  /// Union assumed set with the passed value.
  void unionAssumed(const MemberTy &C) { insert(C); }

  /// Union assumed set with assumed set of the passed state \p PVS.
  void unionAssumed(const PotentialValuesState &PVS) { unionWith(PVS); }

  /// Union assumed set with an undef value.
  void unionAssumedWithUndef() { unionWithUndef(); }

  /// "Clamp" this state with \p PVS.
  PotentialValuesState operator^=(const PotentialValuesState &PVS) {
    IsValidState ^= PVS.IsValidState;
    unionAssumed(PVS);
    return *this;
  }

  PotentialValuesState operator&=(const PotentialValuesState &PVS) {
    IsValidState &= PVS.IsValidState;
    unionAssumed(PVS);
    return *this;
  }

private:
  /// Check the size of this set, and invalidate when the size is no
  /// less than \p MaxPotentialValues threshold.
  void checkAndInvalidate() {
    if (Set.size() >= MaxPotentialValues)
      indicatePessimisticFixpoint();
  }

  /// If this state contains both undef and not undef, we can reduce
  /// undef to the not undef value.
  void reduceUndefValue() { UndefIsContained = UndefIsContained & Set.empty(); }

  /// Insert an element into this set.
  void insert(const MemberTy &C) {
    if (!isValidState())
      return;
    Set.insert(C);
    checkAndInvalidate();
  }

  /// Take union with R.
  void unionWith(const PotentialValuesState &R) {
    /// If this is a full set, do nothing.;
    if (!isValidState())
      return;
    /// If R is full set, change L to a full set.
    if (!R.isValidState()) {
      indicatePessimisticFixpoint();
      return;
    }
    for (const MemberTy &C : R.Set)
      Set.insert(C);
    UndefIsContained |= R.undefIsContained();
    reduceUndefValue();
    checkAndInvalidate();
  }

  /// Take union with an undef value.
  void unionWithUndef() {
    UndefIsContained = true;
    reduceUndefValue();
  }

  /// Take intersection with R.
  void intersectWith(const PotentialValuesState &R) {
    /// If R is a full set, do nothing.
    if (!R.isValidState())
      return;
    /// If this is a full set, change this to R.
    if (!isValidState()) {
      *this = R;
      return;
    }
    SetTy IntersectSet;
    for (const MemberTy &C : Set) {
      if (R.Set.count(C))
        IntersectSet.insert(C);
    }
    Set = IntersectSet;
    UndefIsContained &= R.undefIsContained();
    reduceUndefValue();
  }

  /// A helper state which indicate whether this state is valid or not.
  BooleanState IsValidState;

  /// Container for potential values
  SetTy Set;

  /// Flag for undef value
  bool UndefIsContained;
};

using PotentialConstantIntValuesState = PotentialValuesState<APInt>;

raw_ostream &operator<<(raw_ostream &OS,
                        const PotentialConstantIntValuesState &R);

/// An abstract interface for potential values analysis.
///
/// This AA collects potential values for each IR position.
/// An assumed set of potential values is initialized with the empty set (the
/// best state) and it will grow monotonically as we find more potential values
/// for this position.
/// The set might be forced to the worst state, that is, to contain every
/// possible value for this position in 2 cases.
///   1. We surpassed the \p MaxPotentialValues threshold. This includes the
///      case that this position is affected (e.g. because of an operation) by a
///      Value that is in the worst state.
///   2. We tried to initialize on a Value that we cannot handle (e.g. an
///      operator we do not currently handle).
///
/// TODO: Support values other than constant integers.
struct AAPotentialValues
    : public StateWrapper<PotentialConstantIntValuesState, AbstractAttribute> {
  using Base = StateWrapper<PotentialConstantIntValuesState, AbstractAttribute>;
  AAPotentialValues(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// See AbstractAttribute::getState(...).
  PotentialConstantIntValuesState &getState() override { return *this; }
  const PotentialConstantIntValuesState &getState() const override {
    return *this;
  }

  /// Create an abstract attribute view for the position \p IRP.
  static AAPotentialValues &createForPosition(const IRPosition &IRP,
                                              Attributor &A);

  /// Return assumed constant for the associated value
  Optional<ConstantInt *>
  getAssumedConstantInt(Attributor &A,
                        const Instruction *CtxI = nullptr) const {
    if (!isValidState())
      return nullptr;
    if (getAssumedSet().size() == 1)
      return cast<ConstantInt>(ConstantInt::get(getAssociatedValue().getType(),
                                                *(getAssumedSet().begin())));
    if (getAssumedSet().size() == 0) {
      if (undefIsContained())
        return cast<ConstantInt>(
            ConstantInt::get(getAssociatedValue().getType(), 0));
      return llvm::None;
    }

    return nullptr;
  }

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAPotentialValues"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAPotentialValues
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract interface for all noundef attributes.
struct AANoUndef
    : public IRAttribute<Attribute::NoUndef,
                         StateWrapper<BooleanState, AbstractAttribute>> {
  AANoUndef(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  /// Return true if we assume that the underlying value is noundef.
  bool isAssumedNoUndef() const { return getAssumed(); }

  /// Return true if we know that underlying value is noundef.
  bool isKnownNoUndef() const { return getKnown(); }

  /// Create an abstract attribute view for the position \p IRP.
  static AANoUndef &createForPosition(const IRPosition &IRP, Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AANoUndef"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AANoUndef
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_ATTRIBUTOR_ATTRIBUTES_H
