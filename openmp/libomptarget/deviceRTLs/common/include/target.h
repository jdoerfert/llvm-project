//===-- target.h ---------- OpenMP device runtime target implementation ---===//
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

#ifndef LLVM_OPENMP_LIBOMPTARGET_DEVICERTLS_COMMON_TARGET_H
#define LLVM_OPENMP_LIBOMPTARGET_DEVICERTLS_COMMON_TARGET_H

#include <stdint.h>

extern "C" {

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
///   char ThreadKind = __kmpc_target_init(...);
///
///   // A custom state machine can be provided. If so, the
///   // UseGenericStateMachine argument to the init function should be set to
///   // `false` and the custom one should be executed if ThreaKind is -1.
///   if (ThreadKind == -1)
///     custom_state_machine();
///
///     // Check for (surplus) worker threads and end the kenel for them.
///   if (ThreadKind != 1)
///     return;
///
///   // User defined kernel code.
///
///   __kmpc_target_deinit(...);
/// }
/// \endcode
///
///
///{

/// Initialization
///
/// \param Ident               Source location identification, can be NULL.
///
int8_t __kmpc_target_init(ident_t *Ident, bool IsSPMD,
                          bool UseGenericStateMachine);

/// De-Initialization
///
/// In non-SPMD, this function releases the workers trapped in a state machine
/// and also any memory dynamically allocated by the runtime.
///
/// \param Ident Source location identification, can be NULL.
///
void __kmpc_target_deinit(ident_t *Ident, bool IsSPMD);

///}
}
#endif
