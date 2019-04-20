//===-- target_region.h --- Target region OpenMP devie runtime interface --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Target region interfaces are simple interfaces designed to allow middle-end
// (=LLVM) passes to analyze and transform the code. To achieve good performance
// it may be required to run the associated passes. However, implementations of
// this interface shall always provide a correct implementation as close to the
// user expected code as possible.
//
//===----------------------------------------------------------------------===//

#ifndef _DEVICERTL_COMMON_INTERFACES_H_
#define _DEVICERTL_COMMON_INTERFACES_H_

#ifndef EXTERN
#define EXTERN
#endif
#ifndef CALLBACK
#define CALLBACK(Callee, Payload0, Payload1)
#endif

/// Forward declaration of the source location identifier "ident".
typedef struct ident ident_t;

/// The target region _kernel_ interface for GPUs
///
/// This deliberatly simple interface provides the middle-end (=LLVM) with
/// easier means to reason about the semantic of the code and transform it as
/// well. The runtime calls are therefore also desiged to carry sufficient
/// information necessary for optimization.
///
///
/// Intended usage:
///
/// \code
/// void kernel(...) {
///
///   char ThreadKind = __kmpc_target_region_kernel_init(...);
///
///   if (ThreadKind == -1) {               //  actual worker thread
///     if (!UsedLibraryStateMachine)
///       user_state_machine();
///     goto exit;
///   } else if (ThreadKind == 0) {         // surplus worker thread
///     goto exit;
///   } else {                              //    team master thread
///     goto user_code;
///   }
///
/// user_code:
///
///   // User defined kernel code, parallel regions are replaced by
///   // by __kmpc_target_region_kernel_parallel(...) calls.
///
///   // Fallthrough to de-initialization
///
/// deinit:
///   __kmpc_target_region_kernel_deinit(...);
///
/// exit:
///   /* exit the kernel */
/// }
/// \endcode
///
///
///{

/// Initialization
///
///
/// In SPMD mode, all threads will execute their respective initialization
/// routines.
///
/// In non-SPMD mode, team masters will invoke the initialization routines while
/// the rest is considered a worker thread. Worker threads required for this
/// target region will be trapped inside the function if \p UseStateMachine is
/// true. Otherwise they will escape with a return value of -1
///
/// \param Ident               Source location identification, can be NULL.
/// \param UseSPMDMode         Flag to indicate if execution is performed in
///                            SPMD mode.
/// \param RequiresOMPRuntime  Flag to indicate if the runtime is required and
///                            needs to be initialized.
/// \param UseStateMachine     Flag to indicate if the runtime state machine
///                            should be used in non-SPMD mode.
/// \param RequiresDataSharing Flag to indicate if there might be inter-thread
///                            sharing which needs runtime support.
///
/// \return 1, always in SPMD mode, and in non-SPMD mode if the thread is the
///            team master.
///         0, in non-SPMD mode and the thread is a surplus worker that should
///            not execute anything in the target region.
///        -1, in non-SPMD mode and the thread is a required worker which:
///             - finished work and should be terminated if \p UseStateMachine
///               is true.
///             - has not performed work and should be put in a user provied
///               state machine (as defined above).
///
EXTERN int8_t __kmpc_target_region_kernel_init(ident_t *Ident, bool UseSPMDMode,
                                               bool RequiresOMPRuntime,
                                               bool UseStateMachine,
                                               bool RequiresDataSharing);

/// De-Initialization
///
///
/// In non-SPMD, this function releases the workers trapped in a state machine
/// and also any memory dynamically allocated by the runtime.
///
/// \param Ident              Source location identification, can be NULL.
/// \param UseSPMDMode        Flag to indicate if execution is performed in
///                           SPMD mode.
/// \param RequiredOMPRuntime Flag to indicate if the runtime was required and
///                           is therefore initialized.
///
EXTERN void __kmpc_target_region_kernel_deinit(ident_t *Ident, bool UseSPMDMode,
                                               bool RequiredOMPRuntime);

/// Generic type of a work function in the target region kernel interface. The
/// two arguments are pointers to structures that contains the shared and
/// firstprivate variables respectively. Since the layout and size was known at
/// compile time, the front-end is expected to generate appropriate packing and
/// unpacking code.
typedef void (*ParallelWorkFnTy)(void * /* SharedValues */,
                                 void * /* PrivateValues */);

/// Enter a parallel region
///
///
/// The parallel region is defined by \p ParallelWorkFn. The shared variables,
/// \p SharedMemorySize bytes in total, start at \p SharedValues. The
/// firstprivate variables, \p PrivateValuesBytes bytes in total, start at
/// \p PrivateValues.
///
/// In SPMD mode, this function calls \p ParallelWorkFn with \p SharedValues and
/// \p PrivateValues as arguments before it returns.
///
/// In non-SPMD mode, \p ParallelWorkFn, \p SharedValues, and \p PrivateValues
/// are communicated to the workers before they are released from the state
/// machine to run the code defined by \p ParallelWorkFn in parallel. This
/// function will only return after all workers are finished.
///
/// \param Ident              Source location identification, can be NULL.
/// \param UseSPMDMode        Flag to indicate if execution is performed in
///                           SPMD mode with three potential values:
///                             -1, to indicate unknown mode, a runtime check
///                                 should then determine the current mode.
///                              0, to indicate no SPMD mode.
///                              1, to indicate SPMD mode.
/// \param RequiredOMPRuntime Flag to indicate if the runtime was required and
///                           is therefore initialized.
/// \param ParallelWorkFn     The outlined code that is executed in parallel by
///                           the threads in the team.
/// \param SharedValues       A pointer to the location of all shared values.
/// \param SharedValuesBytes  The total size of the shared values in bytes.
/// \param PrivateValues      A pointer to the location of all private values.
/// \param PrivateValuesBytes The total size of the private values in bytes.
/// \param SharedMemPointers  Flag to indicate that the pointer \p SharedValues
///                           and \p PrivateValues point into shared memory.
///                           If this flag is true, it also requires that all
///                           private values, if any, are stored directly after
///                           the shared values.
///
CALLBACK(ParallelWorkFn, SharedValues, PrivateValues)
EXTERN void __kmpc_target_region_kernel_parallel(
    ident_t *Ident, int16_t UseSPMDMode, bool RequiredOMPRuntime,
    ParallelWorkFnTy ParallelWorkFn, void *SharedValues,
    uint16_t SharedValuesBytes, void *PrivateValues,
    uint16_t PrivateValuesBytes, bool SharedMemPointers);

/// REDUCTION INTERFACE --- TODO
///
///{

#define REDUCTION_OPERATORS()                                                  \
  RO(RO_NOP, NOP)                                                              \
  RO(RO_ADD, ADD)                                                              \
  RO(RO_MUL, MUL)                                                              \
  RO(RO_MIN, MIN)                                                              \
  RO(RO_MAX, MAX)                                                              \
  RO(RO_XOR, XOR)                                                              \
  RO(RO_BOR, BOR)                                                              \
  RO(RO_BAND, BAND)

enum ReductionOperator {
#define RO(NAME, BIN) NAME,
  REDUCTION_OPERATORS()
#undef RO
};

#define REDUCTION_BASE_TYPES()                                                 \
  RBT(RBT_BOOL, BOOL_TY)                                                       \
  RBT(RBT_CHAR, CHAR_TY)                                                       \
  RBT(RBT_SHORT, SHORT_TY)                                                     \
  RBT(RBT_INT, INT_TY)                                                         \
  RBT(RBT_LONG, LONG_TY)                                                       \
  RBT(RBT_LONG_LONG, LONG_LONG_TY)                                             \
  RBT(RBT_HALF, HALF_FLOAT_TY)                                                 \
  RBT(RBT_FLOAT, FLOAT_TY)                                                     \
  RBT(RBT_DOUBLE, DOUBLE_FLOAT_TY)

enum ReductionBaseType {
#define RBT(NAME, TYPE) NAME,
  REDUCTION_BASE_TYPES()
#undef RBT
};

EXTERN void *__kmpc_target_region_kernel_reduction_init(
    ident_t *Ident, int16_t UseSPMDMode, bool RequiredOMPRuntime,
    int32_t GlobalTId, bool IsParallelReduction, bool IsTeamReduction,
    void *OriginalLocation, void *PrivateLocation,
    uint32_t NumReductionLocations, void *RHSPtr,
    enum ReductionBaseType BaseType);

EXTERN void __kmpc_target_region_kernel_reduction_finalize(
    ident_t *Ident, int16_t UseSPMDMode, bool RequiredOMPRuntime,
    int32_t GlobalTId, bool IsParallelReduction, bool IsTeamReduction,
    void *OriginalLocation, void *ReductionLocation,
    uint32_t NumReductionLocations, enum ReductionOperator RedOp,
    enum ReductionBaseType BaseType);
///}

///}

#endif
