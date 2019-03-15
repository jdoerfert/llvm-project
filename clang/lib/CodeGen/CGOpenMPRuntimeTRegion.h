//===-- CGOpenMPRuntimeTRegion.h --- OpenMP RT TRegion interface codegen --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code generation interface for OpenMP target offloading though the generic
// Target Region (TRegion) interface.
//
// See openmp/libomptarget/deviceRTLs/common/target_Region.h for further
// information on the interface functions and their intended use.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMETREGION_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMETREGION_H

#include "CGOpenMPRuntimeTarget.h"
#include "llvm/ADT/SmallBitVector.h"

namespace clang {
namespace CodeGen {

class CGOpenMPRuntimeTRegion : public CGOpenMPRuntimeTarget {
  // TODO: The target region interface only covers kernel codes for now. This
  //       therefore codegen implicitly assumes the target region kernel
  //       interface is targeted. Once a second target region interface is put
  //       in place, e.g., specialized to many-core offloading, we might need
  //       to make the target interface explicit.

  /// Create an outlined function for a target kernel.
  ///
  /// \param D Directive to emit.
  /// \param ParentName Name of the function that encloses the target region.
  /// \param OutlinedFn Outlined function value to be defined by this call.
  /// \param OutlinedFnID Outlined function ID value to be defined by this call.
  /// \param CodeGen Object containing the target statements.
  /// An outlined function may not be an entry if, e.g. the if clause always
  /// evaluates to false.
  void emitKernel(const OMPExecutableDirective &D, StringRef ParentName,
                  llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
                  const RegionCodeGenTy &CodeGen);

  /// Helper for generic kernel mode, target directive's entry function.
  void emitKernelHeader(CodeGenFunction &CGF, llvm::BasicBlock *&ExitBB);

  /// Signal termination of generic mode execution.
  void emitKernelFooter(CodeGenFunction &CGF, llvm::BasicBlock *ExitBB);

  //
  // Base class overrides.
  //

  /// Emit outlined function for 'target' directive.
  ///
  /// \param D Directive to emit.
  /// \param ParentName Name of the function that encloses the target region.
  /// \param OutlinedFn Outlined function value to be defined by this call.
  /// \param OutlinedFnID Outlined function ID value to be defined by this call.
  /// \param IsOffloadEntry True if the outlined function is an offload entry.
  /// An outlined function may not be an entry if, e.g. the if clause always
  /// evaluates to false.
  void emitTargetOutlinedFunction(const OMPExecutableDirective &D,
                                  StringRef ParentName,
                                  llvm::Function *&OutlinedFn,
                                  llvm::Constant *&OutlinedFnID,
                                  bool IsOffloadEntry,
                                  const RegionCodeGenTy &CodeGen) override;

protected:
  /// Get the function name of an outlined region, customized to the target.
  StringRef getOutlinedHelperName() const override { return ".omp_TRegion."; }

public:
  explicit CGOpenMPRuntimeTRegion(CodeGenModule &CGM);

  /// Emits inlined function for the specified OpenMP parallel directive.
  ///
  /// \a D. This outlined function has type void(*)(kmp_int32 *ThreadID,
  /// kmp_int32 BoundID, struct context_vars*).
  /// \param D OpenMP directive.
  /// \param ThreadIDVar Variable for thread id in the current OpenMP region.
  /// \param InnermostKind Kind of innermost directive (for simple directives it
  /// is a directive itself, for combined - its innermost directive).
  /// \param CodeGen Code generation sequence for the \a D directive.
  llvm::Function *
  emitParallelOutlinedFunction(const OMPExecutableDirective &D,
                               const VarDecl *ThreadIDVar,
                               OpenMPDirectiveKind InnermostKind,
                               const RegionCodeGenTy &CodeGen) override;

  /// Emits code for parallel or serial call of the \a OutlinedFn with
  /// variables captured in a record which address is stored in \a
  /// CapturedStruct.
  /// \param OutlinedFn Outlined function to be run in parallel threads. Type of
  /// this function is void(*)(kmp_int32 *, kmp_int32, struct context_vars*).
  /// \param CapturedVars A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  /// \param IfCond Condition in the associated 'if' clause, if it was
  /// specified, nullptr otherwise.
  void emitParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                        llvm::Function *OutlinedFn,
                        ArrayRef<llvm::Value *> CapturedVars,
                        const Expr *IfCond) override;

  /// Emits a critical region.
  /// \param CriticalName Name of the critical region.
  /// \param CriticalOpGen Generator for the statement associated with the given
  /// critical region.
  /// \param Hint Value of the 'hint' clause (optional).
  void emitCriticalRegion(CodeGenFunction &CGF, StringRef CriticalName,
                          const RegionCodeGenTy &CriticalOpGen,
                          SourceLocation Loc,
                          const Expr *Hint = nullptr) override;

  /// Emit a code for reduction clause.
  ///
  /// \param Privates List of private copies for original reduction arguments.
  /// \param LHSExprs List of LHS in \a ReductionOps reduction operations.
  /// \param RHSExprs List of RHS in \a ReductionOps reduction operations.
  /// \param ReductionOps List of reduction operations in form 'LHS binop RHS'
  /// or 'operator binop(LHS, RHS)'.
  /// \param Options List of options for reduction codegen:
  ///     WithNowait true if parent directive has also nowait clause, false
  ///     otherwise.
  ///     SimpleReduction Emit reduction operation only. Used for omp simd
  ///     directive on the host.
  ///     ReductionKind The kind of reduction to perform.
  virtual void emitReduction(CodeGenFunction &CGF, SourceLocation Loc,
                             ArrayRef<const Expr *> Privates,
                             ArrayRef<const Expr *> LHSExprs,
                             ArrayRef<const Expr *> RHSExprs,
                             ArrayRef<const Expr *> ReductionOps,
                             ReductionOptionsTy Options) override;

protected:

  /// Hook to allow derived classes to perform checks on the AST that justify
  /// execution without runtime support.
  virtual bool mayNeedRuntimeSupport() const { return true; }

  /// Hook to allow derived classes to perform checks on the AST that justify
  /// execution without data sharing support.
  virtual bool mayPerformDataSharing() const { return true; }

private:

  /// Helper to check if SPMD mode is enabled. Derived classes that perform
  /// checks on the AST to justify SPMD mode can overload the
  /// CGOpenMPRuntimeTarget::getExecutionMode().
  bool isKnownSPMDMode() const { return getExecutionMode() == EM_SPMD; }

  /// Simple container for a wrapper of an outlined parallel function and the
  /// layout of the passed variables (= captured variables, both shared and
  /// firstprivate).
  struct WrapperInfo {
    llvm::Function *WrapperFn = nullptr;
    llvm::StructType *SharedVarsStructTy = nullptr;
    llvm::StructType *PrivateVarsStructTy = nullptr;
    llvm::SmallBitVector CaptureIsPrivate;
  };

  /// Map an outlined function to its wrapper and shared struct type. The latter
  /// defines the layout of the payload and the wrapper will unpack that payload
  /// and pass the values to the outlined function.
  llvm::DenseMap<llvm::Function *, WrapperInfo> WrapperInfoMap;

  /// Emit function which wraps the outline parallel region
  /// and controls the parameters which are passed to this function.
  /// The wrapper ensures that the outlined function is called
  /// with the correct arguments when data is shared.
  void createParallelDataSharingWrapper(llvm::Function *OutlinedParallelFn,
                                        const OMPExecutableDirective &D);
};

} // namespace CodeGen
} // namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMEKERNEL_H
