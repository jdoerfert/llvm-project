//===- CallbackEncapsulateTest.cpp - Unit tests for callback encapsulate --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CallbackEncapsulate.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static std::unique_ptr<Module> parseIR(LLVMContext &C, StringRef IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("CallbackEncapsulate", errs());
  return Mod;
}

static void verifyModulePtr(std::unique_ptr<Module> &M) {
  ASSERT_NE(M, nullptr);
  ASSERT_FALSE(verifyModule(*M, &errs()));
}

static void verifNoReplCalls(std::unique_ptr<Module> &M) {
  for (Function &F : *M)
    for (Instruction &I : instructions(&F))
      if (auto *CI = dyn_cast<CallInst>(&I))
        EXPECT_FALSE(
            isDirectCallSiteReplacedByAbstractCallSite(ImmutableCallSite(CI)));
}

static CallBase *verifyEncapsulateCallSites(std::unique_ptr<Module> &M,
                                            AbstractCallSite ACS) {
  CallBase *CB = encapsulateAbstractCallSite(ACS);
  EXPECT_TRUE(CB);
  EXPECT_FALSE(verifyModule(*M, &errs()));
  return CB;
}

static void verifyEncapsulatedState(std::unique_ptr<Module> &M,
                                    Function *Caller, Use *CalledUse,
                                    Use *CalleeUse) {
  AbstractCallSite CalledACS(CalledUse);
  ASSERT_TRUE(CalledACS);
  Function *Called = CalledACS.getCalledFunction();
  ASSERT_NE(Called, nullptr);

  AbstractCallSite CalleeACS(CalleeUse);
  ASSERT_TRUE(CalleeACS);
  Function *Callee = CalleeACS.getCalledFunction();
  ASSERT_NE(Callee, nullptr);

  Function *BeforeWrapper =
      M->getFunction((Called->getName() + ".before_wrapper").str());
  ASSERT_NE(BeforeWrapper, nullptr);
  EXPECT_TRUE(BeforeWrapper->hasLocalLinkage());
  BeforeWrapper->setName(BeforeWrapper->getName() + ".seen");

  Function *AfterWrapper =
      M->getFunction((Callee->getName() + ".after_wrapper").str());
  ASSERT_NE(AfterWrapper, nullptr);
  EXPECT_TRUE(AfterWrapper->hasLocalLinkage());
  AfterWrapper->setName(AfterWrapper->getName() + ".seen");

  EXPECT_EQ(AfterWrapper->getFunctionType(), Callee->getFunctionType());

  // Verify we see the right number of function users (this is to some degree
  // specific to the test structure).
  EXPECT_EQ(Caller->getNumUses(), 0U);
  EXPECT_EQ(AfterWrapper->getNumUses(), BeforeWrapper->getNumUses() + 1);

  AbstractCallSite BeforeACS(&*BeforeWrapper->use_begin());
  bool FoundDirectAfterACS = false, FoundCallbackAfterACS = false;
  ASSERT_GE(AfterWrapper->getNumUses(), 1);
  AbstractCallSite AfterDirectACS(&*AfterWrapper->use_begin()),
      AfterCallbackACS(&*AfterWrapper->use_begin());
  for (const Use &U : AfterWrapper->uses()) {
    AbstractCallSite ACS(&U);
    ASSERT_TRUE(ACS);
    if (ACS.isDirectCall()) {
      ASSERT_FALSE(FoundDirectAfterACS);
      AfterDirectACS = ACS;
      FoundDirectAfterACS = true;
    } else if (!FoundCallbackAfterACS) {
      AfterCallbackACS = ACS;
      FoundCallbackAfterACS = true;
    }
  }
  ASSERT_TRUE(FoundDirectAfterACS);
  ASSERT_TRUE(FoundCallbackAfterACS);
  ASSERT_TRUE(AfterDirectACS);
  ASSERT_TRUE(AfterCallbackACS);

  // Verify the call sites are as expected.
  ASSERT_TRUE(CalleeACS);
  if (Called == Callee) {
    EXPECT_TRUE(CalleeACS.isDirectCall());
  } else {
    EXPECT_TRUE(CalleeACS.isCallbackCall());
  }
  ASSERT_TRUE(BeforeACS);
  EXPECT_TRUE(BeforeACS.isDirectCall());
  EXPECT_TRUE(AfterDirectACS.isDirectCall());
  EXPECT_TRUE(AfterCallbackACS.isCallbackCall());

  EXPECT_TRUE(
      isDirectCallSiteReplacedByAbstractCallSite(AfterDirectACS.getCallSite()));
  EXPECT_FALSE(isDirectCallSiteReplacedByAbstractCallSite(
      AfterCallbackACS.getCallSite()));

  EXPECT_FALSE(
      isDirectCallSiteReplacedByAbstractCallSite(CalleeACS.getCallSite()));
  EXPECT_FALSE(
      isDirectCallSiteReplacedByAbstractCallSite(BeforeACS.getCallSite()));

  ASSERT_TRUE(isa<CallInst>(CalleeACS.getInstruction()));
  ASSERT_TRUE(isa<CallInst>(BeforeACS.getInstruction()));
  auto *BeforeWrapperCI = cast<CallInst>(BeforeACS.getInstruction());
  ASSERT_TRUE(isa<CallInst>(AfterDirectACS.getInstruction()));
  auto *AfterWrapperCI = cast<CallInst>(AfterDirectACS.getInstruction());

  auto CalleeAttrs = Callee->getAttributes();
  auto BeforeAttrs = BeforeWrapper->getAttributes();
  auto AfterAttrs = AfterWrapper->getAttributes();
  EXPECT_EQ(CalleeAttrs.getRetAttributes(), BeforeAttrs.getRetAttributes());
  EXPECT_EQ(CalleeAttrs.getRetAttributes(), AfterAttrs.getRetAttributes());

  // Verify the arguments and their mapping is as expected.
  ASSERT_EQ(Callee->arg_size(), AfterWrapper->arg_size());
  if (Called == Callee) {
    ASSERT_EQ(Callee->arg_size() * 2 + 1, BeforeWrapper->arg_size());
  } else {
    ASSERT_LE(Callee->arg_size() + 1, BeforeWrapper->arg_size());
  }

  for (unsigned ArgNo = 0; ArgNo < CalleeACS.getNumArgOperands(); ++ArgNo) {
    int OpIdx = CalleeACS.getCallArgOperandNo(ArgNo);
    if (OpIdx < 0)
      continue;

    auto CalleeAI = Callee->arg_begin() + ArgNo;
    auto AfterAI = AfterWrapper->arg_begin() + ArgNo;
    auto BeforeAI = BeforeWrapper->arg_begin() + OpIdx;
    EXPECT_EQ(CalleeAI->getType(), AfterAI->getType());
    EXPECT_EQ(CalleeAI->getType(), BeforeAI->getType());

    EXPECT_EQ(CalleeAttrs.getAttributes(AttributeList::FirstArgIndex + ArgNo),
              BeforeAttrs.getAttributes(AttributeList::FirstArgIndex + OpIdx));
    EXPECT_EQ(CalleeAttrs.getAttributes(AttributeList::FirstArgIndex + ArgNo),
              AfterAttrs.getAttributes(AttributeList::FirstArgIndex + ArgNo));

    EXPECT_EQ(CalleeAI->getNumUses(), 1U);
    EXPECT_EQ(BeforeAI->getNumUses(), 1U);
    EXPECT_EQ(CalleeAI->user_back(), AfterWrapperCI);
    ASSERT_EQ(BeforeAI->user_back(), CalledACS.getInstruction());
  }

  // Second part of the before wrapper arguments is dependent on the call(back)
  // but we know it starts with the after wrapper.
  auto BeforeAI = BeforeWrapper->arg_begin() + Called->arg_size();
  EXPECT_EQ(BeforeWrapperCI->getArgOperand(BeforeAI->getArgNo()), AfterWrapper);
  ++BeforeAI;

  for (unsigned ArgNo = 0; ArgNo < CalleeACS.getNumArgOperands(); ++ArgNo) {
    int OpIdx = CalleeACS.getCallArgOperandNo(ArgNo);
    if (OpIdx < 0)
      continue;

    unsigned BeforeArgNo = BeforeAI->getArgNo();

    auto CalleeAI = Callee->arg_begin() + ArgNo;
    EXPECT_EQ(CalleeAI->getType(), BeforeAI->getType());

    EXPECT_EQ(
        CalleeAttrs.getAttributes(AttributeList::FirstArgIndex + ArgNo),
        BeforeAttrs.getAttributes(AttributeList::FirstArgIndex + BeforeArgNo));

    EXPECT_EQ(BeforeAI->getNumUses(), 0U);

    ++BeforeAI;
  }
}

class CallbackEncapsulateTest : public ::testing::Test {
protected:
  LLVMContext C;
};

TEST_F(CallbackEncapsulateTest, CallbackEncapsulateDirectCall0) {

  StringRef ModuleAssembly = R"(
define noalias double* @callee() {
entry:
  ret double* null;
}

define double* @caller(i32 %unused) {
entry:
  %call = call double* @callee()
  ret double* %call
}
  )";

  std::unique_ptr<Module> M = parseIR(C, ModuleAssembly);
  verifyModulePtr(M);
  verifNoReplCalls(M);

  Function *Caller = M->getFunction("caller");
  Function *Callee = M->getFunction("callee");
  Function *Called = Callee;
  Use *CalleeUse = &*Callee->use_begin();
  Use *CalledUse = &*Called->use_begin();
  verifyEncapsulateCallSites(M, AbstractCallSite(CalleeUse));
  verifyEncapsulatedState(M, Caller, CalledUse, CalleeUse);
}

TEST_F(CallbackEncapsulateTest, CallbackEncapsulateDirectCall1) {

  StringRef ModuleAssembly = R"(
define double @callee(i32 %i0, double %d0, i16 signext %s0, i16 signext %s1, double %d1, i32 %i1) {
entry:
  %conv = sext i32 %i0 to i64
  call void @use(i64 %conv)
  %conv1 = fptosi double %d0 to i64
  call void @use(i64 %conv1)
  %conv2 = sext i16 %s0 to i64
  call void @use(i64 %conv2)
  %conv3 = sext i32 %i1 to i64
  call void @use(i64 %conv3)
  %conv4 = fptosi double %d1 to i64
  call void @use(i64 %conv4)
  %conv5 = sext i16 %s1 to i64
  call void @use(i64 %conv5)
  %add = fadd double %d0, %d1
  ret double %add
}

declare void @use(i64)

define double @caller(i32 %i, double %d) {
entry:
  %conv = sitofp i32 %i to double
  %mul = fmul double %conv, %d
  %conv1 = fptosi double %mul to i16
  %call = call double @callee(i32 %i, double %d, i16 signext %conv1, i16 signext %conv1, double %d, i32 %i)
  ret double %call
}
  )";

  std::unique_ptr<Module> M = parseIR(C, ModuleAssembly);
  verifyModulePtr(M);
  verifNoReplCalls(M);

  Function *Caller = M->getFunction("caller");
  Function *Callee = M->getFunction("callee");
  Function *Called = Callee;
  Use *CalleeUse = &*Callee->use_begin();
  Use *CalledUse = &*Called->use_begin();
  verifyEncapsulateCallSites(M, AbstractCallSite(CalleeUse));
  verifyEncapsulatedState(M, Caller, CalledUse, CalleeUse);
}

TEST_F(CallbackEncapsulateTest, CallbackEncapsulateDirectCalls0) {

  StringRef ModuleAssembly = R"(
define double* @callee() {
entry:
  ret double* null;
}

define double* @caller(i32 %unused) {
entry:
  %call0 = call double* @callee()
  %call1 = call double* @callee()
  ret double* %call0
}
  )";

  std::unique_ptr<Module> M = parseIR(C, ModuleAssembly);
  verifyModulePtr(M);
  verifNoReplCalls(M);

  Function *Caller = M->getFunction("caller");
  Function *Callee = M->getFunction("callee");
  Function *Called = Callee;
  Use *CalledUse = &*Called->use_begin();
  Use *CalleeUse0 = &*Callee->use_begin();
  Use *CalleeUse1 = &*(++Callee->use_begin());
  verifyEncapsulateCallSites(M, AbstractCallSite(CalleeUse0));
  verifyEncapsulatedState(M, Caller, CalledUse, CalleeUse0);
  verifyEncapsulateCallSites(M, AbstractCallSite(CalleeUse1));
  verifyEncapsulatedState(M, Caller, CalledUse, CalleeUse1);
}

TEST_F(CallbackEncapsulateTest, CallbackEncapsulateTransitiveCall0) {

  StringRef ModuleAssembly = R"(
%union.pthread_attr_t = type { i64, [48 x i8] }

define dso_local i32 @caller(i8* %arg) {
entry:
  %thread = alloca i64, align 8
  store i8 0, i8* %arg
  %call = call i32 @pthread_create(i64* nonnull %thread, %union.pthread_attr_t* null, i8* (i8*)* nonnull @callee, i8* %arg)
  ret i32 0
}

declare !callback !0 dso_local i32 @pthread_create(i64*, %union.pthread_attr_t*, i8* (i8*)*, i8*)

define internal i8* @callee(i8* %arg) {
entry:
  %l = load i8, i8* %arg
  %add = add i8 %l, 1
  store i8 %add, i8* %arg
  ret i8* %arg
}

!1 = !{i64 2, i64 3, i1 false}
!0 = !{!1}
  )";

  std::unique_ptr<Module> M = parseIR(C, ModuleAssembly);
  verifyModulePtr(M);
  verifNoReplCalls(M);

  Function *Caller = M->getFunction("caller");
  Function *Called = M->getFunction("pthread_create");
  Function *Callee = M->getFunction("callee");
  Use *CalledUse = &*Called->use_begin();
  verifyEncapsulateCallSites(M, AbstractCallSite(&*Callee->use_begin()));
  verifyEncapsulatedState(M, Caller, CalledUse, &*Callee->use_begin());
}

TEST_F(CallbackEncapsulateTest, CallbackEncapsulateTransitiveCalls0) {

  StringRef ModuleAssembly = R"(
%union.pthread_attr_t = type { i64, [48 x i8] }

define dso_local i32 @caller(i8* %arg) {
entry:
  %thread = alloca i64, align 8
  store i8 0, i8* %arg
  %call0 = call i32 @pthread_create(i64* nonnull %thread, %union.pthread_attr_t* null, i8* (i8*)* nonnull @callee, i8* %arg)
  %call1 = call i32 @pthread_create(i64* nonnull %thread, %union.pthread_attr_t* null, i8* (i8*)* nonnull @callee, i8* %arg)
  %call2 = call i32 @pthread_create(i64* nonnull %thread, %union.pthread_attr_t* null, i8* (i8*)* nonnull @callee, i8* %arg)
  ret i32 0
}

declare !callback !0 dso_local i32 @pthread_create(i64*, %union.pthread_attr_t*, i8* (i8*)*, i8*)

define internal i8* @callee(i8* %arg) {
entry:
  %l = load i8, i8* %arg
  %add = add i8 %l, 1
  store i8 %add, i8* %arg
  ret i8* %arg
}

!1 = !{i64 2, i64 3, i1 false}
!0 = !{!1}
  )";

  std::unique_ptr<Module> M = parseIR(C, ModuleAssembly);
  verifyModulePtr(M);
  verifNoReplCalls(M);

  Function *Caller = M->getFunction("caller");
  Function *Called = M->getFunction("pthread_create");
  Function *Callee = M->getFunction("callee");

  EXPECT_EQ(Callee->getNumUses(), 3U);
  EXPECT_EQ(Called->getNumUses(), 3U);
  auto CalledUI = Called->use_begin();
  Use *CalledUse = &*(CalledUI++);
  Instruction *EntryIt = Caller->getEntryBlock().getFirstNonPHI();
  CallBase *Call0 = cast<CallBase>(EntryIt->getNextNode()->getNextNode());
  CallBase *Call1 = cast<CallBase>(Call0->getNextNode());
  CallBase *Call2 = cast<CallBase>(Call1->getNextNode());
  Use *CalledUse0 = &Call0->getCalledOperandUse();
  Call0 =
      verifyEncapsulateCallSites(M, AbstractCallSite(&Call0->getOperandUse(2)));
  ASSERT_TRUE(Call0);
  verifyEncapsulatedState(M, Caller, CalledUse0, &Call0->getOperandUse(2));
  Use *CalledUse1 = &Call1->getCalledOperandUse();
  Call1 =
      verifyEncapsulateCallSites(M, AbstractCallSite(&Call1->getOperandUse(2)));
  ASSERT_TRUE(Call1);
  verifyEncapsulatedState(M, Caller, CalledUse1, &Call1->getOperandUse(2));
  Use *CalledUse2 = &Call2->getCalledOperandUse();
  Call2 =
      verifyEncapsulateCallSites(M, AbstractCallSite(&Call2->getOperandUse(2)));
  ASSERT_TRUE(Call2);
  verifyEncapsulatedState(M, Caller, CalledUse2, &Call2->getOperandUse(2));
  EXPECT_EQ(Callee->getNumUses(), 3U);
}

TEST_F(CallbackEncapsulateTest, CallbackEncapsulateTransitiveCalls1) {

  StringRef ModuleAssembly = R"(
%union.pthread_attr_t = type { i64, [48 x i8] }

define dso_local i32 @caller(i8* %arg) {
entry:
  %thread = alloca i64, align 8
  store i8 0, i8* %arg
  %call0 = call i32 @pthread_create(i64* nonnull %thread, %union.pthread_attr_t* null, i8* (i8*)* nonnull @callee0, i8* %arg)
  %call1 = call i32 @pthread_create(i64* nonnull %thread, %union.pthread_attr_t* null, i8* (i8*)* nonnull @callee1, i8* %arg)
  %call2 = call i32 @pthread_create(i64* nonnull %thread, %union.pthread_attr_t* null, i8* (i8*)* nonnull @callee2, i8* %arg)
  ret i32 0
}

declare !callback !0 dso_local i32 @pthread_create(i64*, %union.pthread_attr_t*, i8* (i8*)*, i8*)

define internal i8* @callee0(i8* %arg) {
entry:
  %l = load i8, i8* %arg
  %add = add i8 %l, 1
  store i8 %add, i8* %arg
  ret i8* %arg
}
define internal i8* @callee1(i8* %arg) {
entry:
  %l = load i8, i8* %arg
  %add = add i8 %l, 1
  store i8 %add, i8* %arg
  ret i8* %arg
}
define internal i8* @callee2(i8* %arg) {
entry:
  %l = load i8, i8* %arg
  %add = add i8 %l, 1
  store i8 %add, i8* %arg
  ret i8* %arg
}

!1 = !{i64 2, i64 3, i1 false}
!0 = !{!1}
  )";

  std::unique_ptr<Module> M = parseIR(C, ModuleAssembly);
  verifyModulePtr(M);
  verifNoReplCalls(M);

  Function *Caller = M->getFunction("caller");
  Function *Called = M->getFunction("pthread_create");
  Function *Callee0 = M->getFunction("callee0");
  Function *Callee1 = M->getFunction("callee1");
  Function *Callee2 = M->getFunction("callee2");

  EXPECT_EQ(Callee0->getNumUses(), 1U);
  EXPECT_EQ(Callee1->getNumUses(), 1U);
  EXPECT_EQ(Callee2->getNumUses(), 1U);
  EXPECT_EQ(Called->getNumUses(), 3U);
  auto CalledUI = Called->use_begin();
  Use *CalledUse = &*(CalledUI++);
  Instruction *EntryIt = Caller->getEntryBlock().getFirstNonPHI();
  CallBase *Call0 = cast<CallBase>(EntryIt->getNextNode()->getNextNode());
  CallBase *Call1 = cast<CallBase>(Call0->getNextNode());
  CallBase *Call2 = cast<CallBase>(Call1->getNextNode());
  Use *CalledUse0 = &Call0->getCalledOperandUse();
  Call0 =
      verifyEncapsulateCallSites(M, AbstractCallSite(&Call0->getOperandUse(2)));
  ASSERT_TRUE(Call0);
  verifyEncapsulatedState(M, Caller, CalledUse0, &Call0->getOperandUse(2));
  Use *CalledUse1 = &Call1->getCalledOperandUse();
  Call1 =
      verifyEncapsulateCallSites(M, AbstractCallSite(&Call1->getOperandUse(2)));
  ASSERT_TRUE(Call1);
  verifyEncapsulatedState(M, Caller, CalledUse1, &Call1->getOperandUse(2));
  Use *CalledUse2 = &Call2->getCalledOperandUse();
  Call2 =
      verifyEncapsulateCallSites(M, AbstractCallSite(&Call2->getOperandUse(2)));
  ASSERT_TRUE(Call2);
  verifyEncapsulatedState(M, Caller, CalledUse2, &Call2->getOperandUse(2));
  EXPECT_EQ(Callee0->getNumUses(), 1U);
  EXPECT_EQ(Callee1->getNumUses(), 1U);
  EXPECT_EQ(Callee2->getNumUses(), 1U);
}

} // namespace
