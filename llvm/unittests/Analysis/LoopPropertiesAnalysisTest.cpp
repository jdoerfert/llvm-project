//===- LoopPropertiesAnalysisTest.cpp - LoopInfo unit tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopPropertiesAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;
namespace {

static std::unique_ptr<Module> makeLLVMModule(LLVMContext &Context,
                                              const char *ModuleStr) {
  SMDiagnostic Err;
  return parseAssemblyString(ModuleStr, Err, Context);
}

/// Build the loop info and scalar evolution for the function and run the Test.
static void runWithLoopInfo(
    Module &M, StringRef FuncName,
    function_ref<void(Function &F, LoopInfo &LI, ScalarEvolution &SE)> Test) {
  auto *F = M.getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Could not find " << FuncName;

  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(*F);
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  ScalarEvolution SE(*F, TLI, AC, DT, LI);
  Test(*F, LI, SE);
}

TEST(LoopPropertiesAnalysisTest, BasicTest) {
  const char *ModuleStr = R"IR(
define i32 @foo() {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret i32 0

for.body:                                         ; preds = %entry, %for.cond.cleanup3
  %i.016 = phi i32 [ 1, %entry ], [ %inc8, %for.cond.cleanup3 ]
  br label %for.body4

for.cond.cleanup3:                                ; preds = %for.inc
  %inc8 = add nuw nsw i32 %i.016, 1
  %exitcond17.not = icmp eq i32 %inc8, 4
  br i1 %exitcond17.not, label %for.cond.cleanup, label %for.body

for.body4:                                        ; preds = %for.body, %for.inc
  %j.015 = phi i32 [ 1, %for.body ], [ %inc, %for.inc ]
  %rem = and i32 %j.015, 1
  %cmp5.not = icmp eq i32 %rem, 0
  br i1 %cmp5.not, label %if.end, label %for.inc

if.end:                                           ; preds = %for.body4
  br label %for.inc

for.inc:                                          ; preds = %for.body4, %if.end
  %inc = add nuw nsw i32 %j.015, 1
  %exitcond.not = icmp eq i32 %inc, 8
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.body4
}
)IR";

  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfo(
      *M, "foo", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        for (BasicBlock &BB : F) {
          if (BB.getName() == "for.body") {
            Loop *L = LI.getLoopFor(&BB);
            LoopPropertiesInfo LPI =
                LoopPropertiesInfo::getLoopPropertiesInfo(L, &LI, &SE);
            EXPECT_FALSE(LPI.IsInnerMostLoop);
            EXPECT_EQ(LPI.LoopDepth, 1);
            EXPECT_TRUE(LPI.HasLoopPreheader);
            EXPECT_EQ(LPI.PreheaderBlocksize, 1);
            EXPECT_TRUE(LPI.IsCountableLoop);
            EXPECT_TRUE(LPI.IsLoopBackEdgeConstant);
            EXPECT_EQ(LPI.LoopBackEdgeCount, 2);
            EXPECT_EQ(LPI.BasicBlockCount, 2);
            EXPECT_EQ(LPI.LoopBlocksizes.count(2), 1);
            EXPECT_EQ(LPI.LoopBlocksizes[2], 1);
            EXPECT_EQ(LPI.LoopBlocksizes.count(3), 1);
            EXPECT_EQ(LPI.LoopBlocksizes[3], 1);
            EXPECT_EQ(LPI.LoopLatchCount, 1);
            EXPECT_EQ(LPI.LoadInstCount, 0);
            EXPECT_EQ(LPI.StoreInstCount, 0);
            EXPECT_EQ(LPI.BinaryInstCount, 1);
            EXPECT_EQ(LPI.LogicalInstCount, 0);
            EXPECT_EQ(LPI.CastInstCount, 0);
          }
          if (BB.getName() == "for.body4") {
            Loop *L = LI.getLoopFor(&BB);
            LoopPropertiesInfo LPI =
                LoopPropertiesInfo::getLoopPropertiesInfo(L, &LI, &SE);
            EXPECT_TRUE(LPI.IsInnerMostLoop);
            EXPECT_EQ(LPI.LoopDepth, 2);
            EXPECT_TRUE(LPI.HasLoopPreheader);
            EXPECT_EQ(LPI.PreheaderBlocksize, 2);
            EXPECT_TRUE(LPI.IsCountableLoop);
            EXPECT_TRUE(LPI.IsLoopBackEdgeConstant);
            EXPECT_EQ(LPI.LoopBackEdgeCount, 6);
            EXPECT_EQ(LPI.BasicBlockCount, 3);
            EXPECT_EQ(LPI.LoopBlocksizes.count(1), 1);
            EXPECT_EQ(LPI.LoopBlocksizes[1], 1);
            EXPECT_EQ(LPI.LoopBlocksizes.count(3), 1);
            EXPECT_EQ(LPI.LoopBlocksizes[3], 1);
            EXPECT_EQ(LPI.LoopBlocksizes.count(4), 1);
            EXPECT_EQ(LPI.LoopBlocksizes[4], 1);
            EXPECT_EQ(LPI.LoopLatchCount, 1);
            EXPECT_EQ(LPI.LoadInstCount, 0);
            EXPECT_EQ(LPI.StoreInstCount, 0);
            EXPECT_EQ(LPI.BinaryInstCount, 1);
            EXPECT_EQ(LPI.LogicalInstCount, 1);
            EXPECT_EQ(LPI.CastInstCount, 0);
          }
        }
      });
}

} // end anonymous namespace
