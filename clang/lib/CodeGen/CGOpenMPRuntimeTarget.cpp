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

llvm::FunctionCallee CGOpenMPRuntimeTarget::createTargetRuntimeFunction(
    OpenMPRTLTargetFunctions ID) {
  llvm::FunctionCallee RTLFn = nullptr;
  switch (ID) {
  case OMPRTL_NVPTX__kmpc_kernel_init: {
    // Build void __kmpc_kernel_init(kmp_int32 thread_limit, int16_t
    // RequiresOMPRuntime);
    llvm::Type *TypeParams[] = {CGM.Int32Ty, CGM.Int16Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_init");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_deinit: {
    // Build void __kmpc_kernel_deinit(int16_t IsOMPRuntimeInitialized);
    llvm::Type *TypeParams[] = {CGM.Int16Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_deinit");
    break;
  }
  case OMPRTL_NVPTX__kmpc_spmd_kernel_init: {
    // Build void __kmpc_spmd_kernel_init(kmp_int32 thread_limit,
    // int16_t RequiresOMPRuntime, int16_t RequiresDataSharing);
    llvm::Type *TypeParams[] = {CGM.Int32Ty, CGM.Int16Ty, CGM.Int16Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_spmd_kernel_init");
    break;
  }
  case OMPRTL_NVPTX__kmpc_spmd_kernel_deinit_v2: {
    // Build void __kmpc_spmd_kernel_deinit_v2(int16_t RequiresOMPRuntime);
    llvm::Type *TypeParams[] = {CGM.Int16Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_spmd_kernel_deinit_v2");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_prepare_parallel: {
    /// Build void __kmpc_kernel_prepare_parallel(
    /// void *outlined_function, int16_t IsOMPRuntimeInitialized);
    llvm::Type *TypeParams[] = {CGM.Int8PtrTy, CGM.Int16Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_prepare_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_parallel: {
    /// Build bool __kmpc_kernel_parallel(void **outlined_function,
    /// int16_t IsOMPRuntimeInitialized);
    llvm::Type *TypeParams[] = {CGM.Int8PtrPtrTy, CGM.Int16Ty};
    llvm::Type *RetTy = CGM.getTypes().ConvertType(CGM.getContext().BoolTy);
    auto *FnTy =
        llvm::FunctionType::get(RetTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_end_parallel: {
    /// Build void __kmpc_kernel_end_parallel();
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, llvm::None, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_end_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_serialized_parallel: {
    // Build void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_serialized_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_end_serialized_parallel: {
    // Build void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_end_serialized_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_shuffle_int32: {
    // Build int32_t __kmpc_shuffle_int32(int32_t element,
    // int16_t lane_offset, int16_t warp_size);
    llvm::Type *TypeParams[] = {CGM.Int32Ty, CGM.Int16Ty, CGM.Int16Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_shuffle_int32");
    break;
  }
  case OMPRTL_NVPTX__kmpc_shuffle_int64: {
    // Build int64_t __kmpc_shuffle_int64(int64_t element,
    // int16_t lane_offset, int16_t warp_size);
    llvm::Type *TypeParams[] = {CGM.Int64Ty, CGM.Int16Ty, CGM.Int16Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.Int64Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_shuffle_int64");
    break;
  }
  case OMPRTL_NVPTX__kmpc_nvptx_parallel_reduce_nowait_v2: {
    // Build int32_t kmpc_nvptx_parallel_reduce_nowait_v2(ident_t *loc,
    // kmp_int32 global_tid, kmp_int32 num_vars, size_t reduce_size, void*
    // reduce_data, void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t
    // lane_id, int16_t lane_offset, int16_t Algorithm Version), void
    // (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num));
    llvm::Type *ShuffleReduceTypeParams[] = {CGM.VoidPtrTy, CGM.Int16Ty,
                                             CGM.Int16Ty, CGM.Int16Ty};
    auto *ShuffleReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, ShuffleReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *InterWarpCopyTypeParams[] = {CGM.VoidPtrTy, CGM.Int32Ty};
    auto *InterWarpCopyFnTy =
        llvm::FunctionType::get(CGM.VoidTy, InterWarpCopyTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(),
                                CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.SizeTy,
                                CGM.VoidPtrTy,
                                ShuffleReduceFnTy->getPointerTo(),
                                InterWarpCopyFnTy->getPointerTo()};
    auto *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_parallel_reduce_nowait_v2");
    break;
  }
  case OMPRTL_NVPTX__kmpc_end_reduce_nowait: {
    // Build __kmpc_end_reduce_nowait(kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {CGM.Int32Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_end_reduce_nowait");
    break;
  }
  case OMPRTL_NVPTX__kmpc_nvptx_teams_reduce_nowait_v2: {
    // Build int32_t __kmpc_nvptx_teams_reduce_nowait_v2(ident_t *loc, kmp_int32
    // global_tid, void *global_buffer, int32_t num_of_records, void*
    // reduce_data,
    // void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
    // lane_offset, int16_t shortCircuit),
    // void (*kmp_InterWarpCopyFctPtr)(void* src, int32_t warp_num), void
    // (*kmp_ListToGlobalCpyFctPtr)(void *buffer, int idx, void *reduce_data),
    // void (*kmp_GlobalToListCpyFctPtr)(void *buffer, int idx,
    // void *reduce_data), void (*kmp_GlobalToListCpyPtrsFctPtr)(void *buffer,
    // int idx, void *reduce_data), void (*kmp_GlobalToListRedFctPtr)(void
    // *buffer, int idx, void *reduce_data));
    llvm::Type *ShuffleReduceTypeParams[] = {CGM.VoidPtrTy, CGM.Int16Ty,
                                             CGM.Int16Ty, CGM.Int16Ty};
    auto *ShuffleReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, ShuffleReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *InterWarpCopyTypeParams[] = {CGM.VoidPtrTy, CGM.Int32Ty};
    auto *InterWarpCopyFnTy =
        llvm::FunctionType::get(CGM.VoidTy, InterWarpCopyTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *GlobalListTypeParams[] = {CGM.VoidPtrTy, CGM.IntTy,
                                          CGM.VoidPtrTy};
    auto *GlobalListFnTy =
        llvm::FunctionType::get(CGM.VoidTy, GlobalListTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(),
                                CGM.Int32Ty,
                                CGM.VoidPtrTy,
                                CGM.Int32Ty,
                                CGM.VoidPtrTy,
                                ShuffleReduceFnTy->getPointerTo(),
                                InterWarpCopyFnTy->getPointerTo(),
                                GlobalListFnTy->getPointerTo(),
                                GlobalListFnTy->getPointerTo(),
                                GlobalListFnTy->getPointerTo(),
                                GlobalListFnTy->getPointerTo()};
    auto *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_teams_reduce_nowait_v2");
    break;
  }
  case OMPRTL_NVPTX__kmpc_data_sharing_init_stack: {
    /// Build void __kmpc_data_sharing_init_stack();
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, llvm::None, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_data_sharing_init_stack");
    break;
  }
  case OMPRTL_NVPTX__kmpc_data_sharing_init_stack_spmd: {
    /// Build void __kmpc_data_sharing_init_stack_spmd();
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, llvm::None, /*isVarArg*/ false);
    RTLFn =
        CGM.CreateRuntimeFunction(FnTy, "__kmpc_data_sharing_init_stack_spmd");
    break;
  }
  case OMPRTL_NVPTX__kmpc_data_sharing_coalesced_push_stack: {
    // Build void *__kmpc_data_sharing_coalesced_push_stack(size_t size,
    // int16_t UseSharedMemory);
    llvm::Type *TypeParams[] = {CGM.SizeTy, CGM.Int16Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_data_sharing_coalesced_push_stack");
    break;
  }
  case OMPRTL_NVPTX__kmpc_data_sharing_pop_stack: {
    // Build void __kmpc_data_sharing_pop_stack(void *a);
    llvm::Type *TypeParams[] = {CGM.VoidPtrTy};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy,
                                      /*Name=*/"__kmpc_data_sharing_pop_stack");
    break;
  }
  case OMPRTL_NVPTX__kmpc_begin_sharing_variables: {
    /// Build void __kmpc_begin_sharing_variables(void ***args,
    /// size_t n_args);
    llvm::Type *TypeParams[] = {CGM.Int8PtrPtrTy->getPointerTo(), CGM.SizeTy};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_begin_sharing_variables");
    break;
  }
  case OMPRTL_NVPTX__kmpc_end_sharing_variables: {
    /// Build void __kmpc_end_sharing_variables();
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, llvm::None, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_end_sharing_variables");
    break;
  }
  case OMPRTL_NVPTX__kmpc_get_shared_variables: {
    /// Build void __kmpc_get_shared_variables(void ***GlobalArgs);
    llvm::Type *TypeParams[] = {CGM.Int8PtrPtrTy->getPointerTo()};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_get_shared_variables");
    break;
  }
  case OMPRTL_NVPTX__kmpc_parallel_level: {
    // Build uint16_t __kmpc_parallel_level(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.Int16Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_parallel_level");
    break;
  }
  case OMPRTL_NVPTX__kmpc_is_spmd_exec_mode: {
    // Build int8_t __kmpc_is_spmd_exec_mode();
    auto *FnTy = llvm::FunctionType::get(CGM.Int8Ty, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_is_spmd_exec_mode");
    break;
  }
  case OMPRTL_NVPTX__kmpc_get_team_static_memory: {
    // Build void __kmpc_get_team_static_memory(int16_t isSPMDExecutionMode,
    // const void *buf, size_t size, int16_t is_shared, const void **res);
    llvm::Type *TypeParams[] = {CGM.Int16Ty, CGM.VoidPtrTy, CGM.SizeTy,
                                CGM.Int16Ty, CGM.VoidPtrPtrTy};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_get_team_static_memory");
    break;
  }
  case OMPRTL_NVPTX__kmpc_restore_team_static_memory: {
    // Build void __kmpc_restore_team_static_memory(int16_t isSPMDExecutionMode,
    // int16_t is_shared);
    llvm::Type *TypeParams[] = {CGM.Int16Ty, CGM.Int16Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn =
        CGM.CreateRuntimeFunction(FnTy, "__kmpc_restore_team_static_memory");
    break;
  }
  case OMPRTL__kmpc_barrier: {
    // Build void __kmpc_barrier(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name*/ "__kmpc_barrier");
    cast<llvm::Function>(RTLFn.getCallee())
        ->addFnAttr(llvm::Attribute::Convergent);
    break;
  }
  case OMPRTL__kmpc_barrier_simple_spmd: {
    // Build void __kmpc_barrier_simple_spmd(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    auto *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn =
        CGM.CreateRuntimeFunction(FnTy, /*Name*/ "__kmpc_barrier_simple_spmd");
    cast<llvm::Function>(RTLFn.getCallee())
        ->addFnAttr(llvm::Attribute::Convergent);
    break;
  }
  }
  return RTLFn;
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
