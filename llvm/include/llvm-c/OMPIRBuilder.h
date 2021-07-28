//===--------- OpenMPIRBuilder.h - LLVM C API OpenMP-IR-Builder API -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the C API endpoints for generating OpenMP LLVM-IR.
///
/// Note: This interface is experimental. It is *NOT* stable, and may be
///       changed without warning.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_C_OPENMPIRBUILDER_H
#define LLVM_C_OPENMPIRBUILDER_H

#include "llvm-c/Core.h"
#include "llvm-c/ExternC.h"

LLVM_C_EXTERN_C_BEGIN

typedef struct LLVMOpaqueOpenMPIRBuilder *LLVMOpenMPIRBuilderRef;

/**
 * @see omp::Directive
 */
typedef enum {
  LLVMOMPD_allocate,
  LLVMOMPD_assumes,
  LLVMOMPD_atomic,
  LLVMOMPD_barrier,
  LLVMOMPD_begin_assumes,
  LLVMOMPD_begin_declare_target,
  LLVMOMPD_begin_declare_variant,
  LLVMOMPD_cancel,
  LLVMOMPD_cancellation_point,
  LLVMOMPD_critical,
  LLVMOMPD_declare_mapper,
  LLVMOMPD_declare_reduction,
  LLVMOMPD_declare_simd,
  LLVMOMPD_declare_target,
  LLVMOMPD_declare_variant,
  LLVMOMPD_depobj,
  LLVMOMPD_distribute,
  LLVMOMPD_distribute_parallel_do,
  LLVMOMPD_distribute_parallel_do_simd,
  LLVMOMPD_distribute_parallel_for,
  LLVMOMPD_distribute_parallel_for_simd,
  LLVMOMPD_distribute_simd,
  LLVMOMPD_do,
  LLVMOMPD_do_simd,
  LLVMOMPD_end_assumes,
  LLVMOMPD_end_declare_target,
  LLVMOMPD_end_declare_variant,
  LLVMOMPD_end_do,
  LLVMOMPD_end_do_simd,
  LLVMOMPD_end_sections,
  LLVMOMPD_end_single,
  LLVMOMPD_end_workshare,
  LLVMOMPD_flush,
  LLVMOMPD_for,
  LLVMOMPD_for_simd,
  LLVMOMPD_master,
  LLVMOMPD_master_taskloop,
  LLVMOMPD_master_taskloop_simd,
  LLVMOMPD_ordered,
  LLVMOMPD_parallel,
  LLVMOMPD_parallel_do,
  LLVMOMPD_parallel_do_simd,
  LLVMOMPD_parallel_for,
  LLVMOMPD_parallel_for_simd,
  LLVMOMPD_parallel_master,
  LLVMOMPD_parallel_master_taskloop,
  LLVMOMPD_parallel_master_taskloop_simd,
  LLVMOMPD_parallel_sections,
  LLVMOMPD_parallel_workshare,
  LLVMOMPD_requires,
  LLVMOMPD_scan,
  LLVMOMPD_section,
  LLVMOMPD_sections,
  LLVMOMPD_simd,
  LLVMOMPD_single,
  LLVMOMPD_target,
  LLVMOMPD_target_data,
  LLVMOMPD_target_enter_data,
  LLVMOMPD_target_exit_data,
  LLVMOMPD_target_parallel,
  LLVMOMPD_target_parallel_do,
  LLVMOMPD_target_parallel_do_simd,
  LLVMOMPD_target_parallel_for,
  LLVMOMPD_target_parallel_for_simd,
  LLVMOMPD_target_simd,
  LLVMOMPD_target_teams,
  LLVMOMPD_target_teams_distribute,
  LLVMOMPD_target_teams_distribute_parallel_do,
  LLVMOMPD_target_teams_distribute_parallel_do_simd,
  LLVMOMPD_target_teams_distribute_parallel_for,
  LLVMOMPD_target_teams_distribute_parallel_for_simd,
  LLVMOMPD_target_teams_distribute_simd,
  LLVMOMPD_target_update,
  LLVMOMPD_task,
  LLVMOMPD_taskgroup,
  LLVMOMPD_taskloop,
  LLVMOMPD_taskloop_simd,
  LLVMOMPD_taskwait,
  LLVMOMPD_taskyield,
  LLVMOMPD_teams,
  LLVMOMPD_teams_distribute,
  LLVMOMPD_teams_distribute_parallel_do,
  LLVMOMPD_teams_distribute_parallel_do_simd,
  LLVMOMPD_teams_distribute_parallel_for,
  LLVMOMPD_teams_distribute_parallel_for_simd,
  LLVMOMPD_teams_distribute_simd,
  LLVMOMPD_threadprivate,
  LLVMOMPD_tile,
  LLVMOMPD_unknown,
  LLVMOMPD_unroll,
  LLVMOMPD_workshare,
  LLVMOMPD_dispatch,
  LLVMOMPD_interop,
  LLVMOMPD_masked,
} LLVMOMPDirective;

/**
 * @see OpenMPIRBuilder::LocationDescription
 */
struct LLVMOpenMPIRBuilderLocationDescription {
  /// The insertion point, an llvm::Instruction, if null BB has to be set.
  LLVMValueRef IP;

  /// The llvm::BasicBlock to insert at the end, if null IP has to be set.
  LLVMBasicBlockRef BB;

  /// The debug location metadata, or null.
  LLVMMetadataRef DebugLoc;
};

/**
 * Obtain an OpenMP-IR-Builder for the module \p M.
 *
 * @see OpenMPIRBuilder::OpenMPIRBuilder(...)
 * @see OpenMPIRBuilder::initialize()
 */
LLVMOpenMPIRBuilderRef LLVMGetOpenMPIRBuilder(LLVMModuleRef M);

/**
 * Generator for `#omp barrier`.
 *
 * @see OpenMPIRBuilder::createBarrier(...)
 */
LLVMOpenMPIRBuilderLocationDescription LLVMOpenMPIRBuilderCreateBarrier(
    LLVMOpenMPIRBuilderRef OMPBuilder,
    LLVMOpenMPIRBuilderLocationDescription LocationDescription,
    LLVMOMPDirective Directive, LLVMBool ForceSimpleCall,
    LLVMBool CheckCancelFlag);

/**
 * Finalize the OpenMP-IR-Builder \p OMPBuilder.
 *
 * @see OpenMPIRBuilder::finalize()
 */
void LLVMFinalizeOpenMPIRBuilder(LLVMOpenMPIRBuilderRef OMPBuilder);

LLVM_C_EXTERN_C_END

#endif
