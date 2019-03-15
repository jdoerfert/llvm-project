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
