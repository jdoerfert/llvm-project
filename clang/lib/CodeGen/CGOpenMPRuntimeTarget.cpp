//===-- CGOpenMPRuntimeTarget.cpp - Common OpenMP target codegen ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the code generation interface for OpenMP target offloading
// though the Target Region (TRegion) interface.
//
// See the file comment in CGOpenMPRuntimeTarget.h for more information.
//
//===----------------------------------------------------------------------===//

#include "CGOpenMPRuntimeTarget.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Cuda.h"

using namespace clang;
using namespace CodeGen;

CGOpenMPRuntimeTarget::CGOpenMPRuntimeTarget(CodeGenModule &CGM)
    : CGOpenMPRuntime(CGM, "_", "$") {
  if (!CGM.getLangOpts().OpenMPIsDevice)
    llvm_unreachable("Target code generation does only handle device code!");
}

const ValueDecl *CGOpenMPRuntimeTarget::getUnderlyingVar(const Expr *E) {
  E = E->IgnoreParens();
  if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    const Expr *Base = ASE->getBase()->IgnoreParenImpCasts();
    while (const auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
      Base = TempASE->getBase()->IgnoreParenImpCasts();
    E = Base;
  } else if (auto *OASE = dyn_cast<OMPArraySectionExpr>(E)) {
    const Expr *Base = OASE->getBase()->IgnoreParenImpCasts();
    while (const auto *TempOASE = dyn_cast<OMPArraySectionExpr>(Base))
      Base = TempOASE->getBase()->IgnoreParenImpCasts();
    while (const auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
      Base = TempASE->getBase()->IgnoreParenImpCasts();
    E = Base;
  }
  E = E->IgnoreParenImpCasts();
  if (const auto *DE = dyn_cast<DeclRefExpr>(E))
    return cast<ValueDecl>(DE->getDecl()->getCanonicalDecl());
  const auto *ME = cast<MemberExpr>(E);
  return cast<ValueDecl>(ME->getMemberDecl()->getCanonicalDecl());
}

void CGOpenMPRuntimeTarget::createOffloadEntry(
    llvm::Constant *ID, llvm::Constant *Addr, uint64_t Size, int32_t,
    llvm::GlobalValue::LinkageTypes) {
  // TODO: Add support for global variables on the device after declare target
  // support.
  if (!isa<llvm::Function>(Addr))
    return;
  llvm::Module &M = CGM.getModule();
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();

  // Get "nvvm.annotations" metadata node
  llvm::NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");

  llvm::Metadata *MDVals[] = {
      llvm::ConstantAsMetadata::get(Addr), llvm::MDString::get(Ctx, "kernel"),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 1))};
  // Append metadata to nvvm.annotations
  MD->addOperand(llvm::MDNode::get(Ctx, MDVals));
}

void CGOpenMPRuntimeTarget::emitProcBindClause(
    CodeGenFunction &CGF, OpenMPProcBindClauseKind ProcBind,
    SourceLocation Loc) {
  // Do nothing in case of SPMD mode and L0 parallel.
  if (getExecutionMode() == CGOpenMPRuntimeTarget::EM_SPMD)
    return;

  CGOpenMPRuntime::emitProcBindClause(CGF, ProcBind, Loc);
}

void CGOpenMPRuntimeTarget::emitNumThreadsClause(CodeGenFunction &CGF,
                                                  llvm::Value *NumThreads,
                                                  SourceLocation Loc) {
  // Do nothing in case of SPMD mode and L0 parallel.
  if (getExecutionMode() == CGOpenMPRuntimeTarget::EM_SPMD)
    return;

  CGOpenMPRuntime::emitNumThreadsClause(CGF, NumThreads, Loc);
}

void CGOpenMPRuntimeTarget::emitNumTeamsClause(CodeGenFunction &CGF,
                                               const Expr *NumTeams,
                                               const Expr *ThreadLimit,
                                               SourceLocation Loc) {}

void CGOpenMPRuntimeTarget::getDefaultScheduleAndChunk(
    CodeGenFunction &CGF, const OMPLoopDirective &S,
    OpenMPScheduleClauseKind &ScheduleKind, const Expr *&ChunkExpr) const {
  ScheduleKind = OMPC_SCHEDULE_static;
  // Chunk size is 1 in this case.
  llvm::APInt ChunkSize(32, 1);
  ChunkExpr = IntegerLiteral::Create(
      CGF.getContext(), ChunkSize,
      CGF.getContext().getIntTypeForBitwidth(32, /* Signed */ 0),
      SourceLocation());
}

void CGOpenMPRuntimeTarget::emitTeamsCall(
    CodeGenFunction &CGF, const OMPExecutableDirective &D, SourceLocation Loc,
    llvm::Function *OutlinedFn, ArrayRef<llvm::Value *> CapturedVars) {
  if (!CGF.HaveInsertPoint())
    return;

  Address ThreadIDAddr = emitThreadIDAddress(CGF, Loc);
  Address ZeroAddr = CGF.CreateMemTemp(
      CGF.getContext().getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/1),
      /*Name*/ ".zero.addr");
  CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C*/ 0));
  llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
  OutlinedFnArgs.push_back(ThreadIDAddr.getPointer());
  OutlinedFnArgs.push_back(ZeroAddr.getPointer());
  OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());

  emitOutlinedFunctionCall(CGF, Loc, OutlinedFn, OutlinedFnArgs);
}

unsigned CGOpenMPRuntimeTarget::getDefaultFirstprivateAddressSpace() const {
  return CGM.getContext().getTargetAddressSpace(LangAS::cuda_constant);
}

// Get current CudaArch and ignore any unknown values
static CudaArch getCudaArch(CodeGenModule &CGM) {
  if (!CGM.getTarget().hasFeature("ptx"))
    return CudaArch::UNKNOWN;
  llvm::StringMap<bool> Features;
  CGM.getTarget().initFeatureMap(Features, CGM.getDiags(),
                                 CGM.getTarget().getTargetOpts().CPU,
                                 CGM.getTarget().getTargetOpts().Features);
  for (const auto &Feature : Features) {
    if (Feature.getValue()) {
      CudaArch Arch = StringToCudaArch(Feature.getKey());
      if (Arch != CudaArch::UNKNOWN)
        return Arch;
    }
  }
  return CudaArch::UNKNOWN;
}

/// Check to see if target architecture supports unified addressing which is
/// a restriction for OpenMP requires clause "unified_shared_memory".
void CGOpenMPRuntimeTarget::checkArchForUnifiedAddressing(
    CodeGenModule &CGM, const OMPRequiresDecl *D) const {
  for (const OMPClause *Clause : D->clauselists()) {
    if (Clause->getClauseKind() == OMPC_unified_shared_memory) {
      switch (getCudaArch(CGM)) {
      case CudaArch::SM_20:
      case CudaArch::SM_21:
      case CudaArch::SM_30:
      case CudaArch::SM_32:
      case CudaArch::SM_35:
      case CudaArch::SM_37:
      case CudaArch::SM_50:
      case CudaArch::SM_52:
      case CudaArch::SM_53:
      case CudaArch::SM_60:
      case CudaArch::SM_61:
      case CudaArch::SM_62:
        CGM.Error(Clause->getBeginLoc(),
                  "Target architecture does not support unified addressing");
        return;
      case CudaArch::SM_70:
      case CudaArch::SM_72:
      case CudaArch::SM_75:
      case CudaArch::GFX600:
      case CudaArch::GFX601:
      case CudaArch::GFX700:
      case CudaArch::GFX701:
      case CudaArch::GFX702:
      case CudaArch::GFX703:
      case CudaArch::GFX704:
      case CudaArch::GFX801:
      case CudaArch::GFX802:
      case CudaArch::GFX803:
      case CudaArch::GFX810:
      case CudaArch::GFX900:
      case CudaArch::GFX902:
      case CudaArch::GFX904:
      case CudaArch::GFX906:
      case CudaArch::GFX909:
      case CudaArch::UNKNOWN:
        break;
      case CudaArch::LAST:
        llvm_unreachable("Unexpected Cuda arch.");
      }
    }
  }
}
