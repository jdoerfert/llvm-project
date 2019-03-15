//===-- CGOpenMPRuntimeTarget.h --- Common OpenMP target codegen ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code common to all OpenMP target codegens.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMETARGET_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMETARGET_H

#include "CGOpenMPRuntime.h"

namespace clang {
namespace CodeGen {

struct CGOpenMPRuntimeTarget : public CGOpenMPRuntime {

  explicit CGOpenMPRuntimeTarget(CodeGenModule &CGM);

  /// Defines the execution mode.
  enum ExecutionMode {
    /// SPMD execution mode (all threads are worker threads).
    EM_SPMD,
    /// Non-SPMD execution mode (1 master thread, others are workers).
    EM_NonSPMD,
    /// Unknown execution mode (orphaned directive).
    EM_Unknown,
  };

  /// Return the execution mode, if not overloaded this is always Unknown.
  virtual ExecutionMode getExecutionMode() const { return EM_Unknown; }

  /// Return the value decleration encapsulated in the expression \p E.
  static const ValueDecl *getUnderlyingVar(const Expr *E);

  enum OpenMPRTLTargetFunctions {
    /// Call to void __kmpc_kernel_init(kmp_int32 thread_limit,
    /// int16_t RequiresOMPRuntime);
    OMPRTL_NVPTX__kmpc_kernel_init,
    /// Call to void __kmpc_kernel_deinit(int16_t IsOMPRuntimeInitialized);
    OMPRTL_NVPTX__kmpc_kernel_deinit,
    /// Call to void __kmpc_spmd_kernel_init(kmp_int32 thread_limit,
    /// int16_t RequiresOMPRuntime, int16_t RequiresDataSharing);
    OMPRTL_NVPTX__kmpc_spmd_kernel_init,
    /// Call to void __kmpc_spmd_kernel_deinit_v2(int16_t RequiresOMPRuntime);
    OMPRTL_NVPTX__kmpc_spmd_kernel_deinit_v2,
    /// Call to void __kmpc_kernel_prepare_parallel(void
    /// *outlined_function, int16_t
    /// IsOMPRuntimeInitialized);
    OMPRTL_NVPTX__kmpc_kernel_prepare_parallel,
    /// Call to bool __kmpc_kernel_parallel(void **outlined_function,
    /// int16_t IsOMPRuntimeInitialized);
    OMPRTL_NVPTX__kmpc_kernel_parallel,
    /// Call to void __kmpc_kernel_end_parallel();
    OMPRTL_NVPTX__kmpc_kernel_end_parallel,
    /// Call to void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
    /// global_tid);
    OMPRTL_NVPTX__kmpc_serialized_parallel,
    /// Call to void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
    /// global_tid);
    OMPRTL_NVPTX__kmpc_end_serialized_parallel,
    /// Call to int32_t __kmpc_shuffle_int32(int32_t element,
    /// int16_t lane_offset, int16_t warp_size);
    OMPRTL_NVPTX__kmpc_shuffle_int32,
    /// Call to int64_t __kmpc_shuffle_int64(int64_t element,
    /// int16_t lane_offset, int16_t warp_size);
    OMPRTL_NVPTX__kmpc_shuffle_int64,
    /// Call to __kmpc_nvptx_parallel_reduce_nowait_v2(ident_t *loc, kmp_int32
    /// global_tid, kmp_int32 num_vars, size_t reduce_size, void* reduce_data,
    /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
    /// lane_offset, int16_t shortCircuit),
    /// void (*kmp_InterWarpCopyFctPtr)(void* src, int32_t warp_num));
    OMPRTL_NVPTX__kmpc_nvptx_parallel_reduce_nowait_v2,
    /// Call to __kmpc_nvptx_teams_reduce_nowait_v2(ident_t *loc, kmp_int32
    /// global_tid, void *global_buffer, int32_t num_of_records, void*
    /// reduce_data,
    /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
    /// lane_offset, int16_t shortCircuit),
    /// void (*kmp_InterWarpCopyFctPtr)(void* src, int32_t warp_num), void
    /// (*kmp_ListToGlobalCpyFctPtr)(void *buffer, int idx, void *reduce_data),
    /// void (*kmp_GlobalToListCpyFctPtr)(void *buffer, int idx,
    /// void *reduce_data), void (*kmp_GlobalToListCpyPtrsFctPtr)(void *buffer,
    /// int idx, void *reduce_data), void (*kmp_GlobalToListRedFctPtr)(void
    /// *buffer, int idx, void *reduce_data));
    OMPRTL_NVPTX__kmpc_nvptx_teams_reduce_nowait_v2,
    /// Call to __kmpc_nvptx_end_reduce_nowait(int32_t global_tid);
    OMPRTL_NVPTX__kmpc_end_reduce_nowait,
    /// Call to void __kmpc_data_sharing_init_stack();
    OMPRTL_NVPTX__kmpc_data_sharing_init_stack,
    /// Call to void __kmpc_data_sharing_init_stack_spmd();
    OMPRTL_NVPTX__kmpc_data_sharing_init_stack_spmd,
    /// Call to void* __kmpc_data_sharing_coalesced_push_stack(size_t size,
    /// int16_t UseSharedMemory);
    OMPRTL_NVPTX__kmpc_data_sharing_coalesced_push_stack,
    /// Call to void __kmpc_data_sharing_pop_stack(void *a);
    OMPRTL_NVPTX__kmpc_data_sharing_pop_stack,
    /// Call to void __kmpc_begin_sharing_variables(void ***args,
    /// size_t n_args);
    OMPRTL_NVPTX__kmpc_begin_sharing_variables,
    /// Call to void __kmpc_end_sharing_variables();
    OMPRTL_NVPTX__kmpc_end_sharing_variables,
    /// Call to void __kmpc_get_shared_variables(void ***GlobalArgs)
    OMPRTL_NVPTX__kmpc_get_shared_variables,
    /// Call to uint16_t __kmpc_parallel_level(ident_t *loc, kmp_int32
    /// global_tid);
    OMPRTL_NVPTX__kmpc_parallel_level,
    /// Call to int8_t __kmpc_is_spmd_exec_mode();
    OMPRTL_NVPTX__kmpc_is_spmd_exec_mode,
    /// Call to void __kmpc_get_team_static_memory(int16_t isSPMDExecutionMode,
    /// const void *buf, size_t size, int16_t is_shared, const void **res);
    OMPRTL_NVPTX__kmpc_get_team_static_memory,
    /// Call to void __kmpc_restore_team_static_memory(int16_t
    /// isSPMDExecutionMode, int16_t is_shared);
    OMPRTL_NVPTX__kmpc_restore_team_static_memory,
    /// Call to void __kmpc_barrier(ident_t *loc, kmp_int32 global_tid);
    OMPRTL__kmpc_barrier,
    /// Call to void __kmpc_barrier_simple_spmd(ident_t *loc, kmp_int32
    /// global_tid);
    OMPRTL__kmpc_barrier_simple_spmd,

    /// Target Region (TREgion) Kernel interface
    ///
    ///{

    /// char __kmpc_target_region_kernel_init(ident_t *Ident,
    ///                                       bool UseSPMDMode,
    ///                                       bool UseStateMachine,
    ///                                       bool RequiresOMPRuntime,
    ///                                       bool RequiresDataSharing);
    OMPRTL__kmpc_target_region_kernel_init,

    /// void __kmpc_target_region_kernel_deinit(ident_t *Ident,
    ///                                         bool UseSPMDMode,
    ///                                         bool RequiredOMPRuntime);
    OMPRTL__kmpc_target_region_kernel_deinit,

    /// void __kmpc_target_region_kernel_parallel(ident_t *Ident,
    ///                                           uint16_t UseSPMDMode,
    ///                                           bool RequiredOMPRuntime,
    ///                                           ParallelWorkFnTy WorkFn,
    ///                                           void *SharedVars,
    ///                                           uint16_t SharedVarsBytes,
    ///                                           void *PrivateVars,
    ///                                           uint16_t PrivateVarsBytes,
    ///                                           bool SharedPointers);
    OMPRTL__kmpc_target_region_kernel_parallel,

    ///}
  };

  /// Returns the OpenMP runtime function identified by \p ID.
  llvm::FunctionCallee createTargetRuntimeFunction(OpenMPRTLTargetFunctions ID);

  //
  // Base class overrides.
  //

  /// Creates offloading entry for the provided entry ID \a ID,
  /// address \a Addr, size \a Size, and flags \a Flags.
  void createOffloadEntry(llvm::Constant *ID, llvm::Constant *Addr,
                          uint64_t Size, int32_t Flags,
                          llvm::GlobalValue::LinkageTypes Linkage) override;

  /// Emit call to void __kmpc_push_proc_bind(ident_t *loc, kmp_int32
  /// global_tid, int proc_bind) to generate code for 'proc_bind' clause.
  virtual void emitProcBindClause(CodeGenFunction &CGF,
                                  OpenMPProcBindClauseKind ProcBind,
                                  SourceLocation Loc) override;

  /// Emits call to void __kmpc_push_num_threads(ident_t *loc, kmp_int32
  /// global_tid, kmp_int32 num_threads) to generate code for 'num_threads'
  /// clause.
  /// \param NumThreads An integer value of threads.
  virtual void emitNumThreadsClause(CodeGenFunction &CGF,
                                    llvm::Value *NumThreads,
                                    SourceLocation Loc) override;

  /// Set the number of teams to \p NumTeams and the thread limit to
  /// \p ThreadLimit.
  ///
  /// \param NumTeams An integer expression of teams.
  /// \param ThreadLimit An integer expression of threads.
  void emitNumTeamsClause(CodeGenFunction &CGF, const Expr *NumTeams,
                          const Expr *ThreadLimit, SourceLocation Loc) override;

  /// Choose a default value for the schedule clause.
  void getDefaultScheduleAndChunk(CodeGenFunction &CGF,
                                  const OMPLoopDirective &S,
                                  OpenMPScheduleClauseKind &ScheduleKind,
                                  const Expr *&ChunkExpr) const override;

  /// Emits code for teams call of the \a OutlinedFn with
  /// variables captured in a record which address is stored in \a
  /// CapturedStruct.
  /// \param OutlinedFn Outlined function to be run by team masters. Type of
  /// this function is void(*)(kmp_int32 *, kmp_int32, struct context_vars*).
  /// \param CapturedVars A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  ///
  void emitTeamsCall(CodeGenFunction &CGF, const OMPExecutableDirective &D,
                     SourceLocation Loc, llvm::Function *OutlinedFn,
                     ArrayRef<llvm::Value *> CapturedVars) override;

  /// Returns default address space for the constant firstprivates, __constant__
  /// address space by default.
  unsigned getDefaultFirstprivateAddressSpace() const override;

  /// Perform check on requires decl to ensure that target architecture
  /// supports unified addressing
  void checkArchForUnifiedAddressing(CodeGenModule &CGM,
                                     const OMPRequiresDecl *D) const override;
};

} // namespace CodeGen
} // namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMETARGET_H
