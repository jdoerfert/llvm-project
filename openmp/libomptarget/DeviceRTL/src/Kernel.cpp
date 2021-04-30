//===--- Kernel.cpp - OpenMP device kernel interface -------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the kernel entry points for the device.
//
//===----------------------------------------------------------------------===//

#include "Debug.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"

using namespace _OMP;

#pragma omp declare target

static void inititializeRuntime(bool IsSPMD) {
  synchronize::init(IsSPMD);
  mapping::init(IsSPMD);
  state::init(IsSPMD);
}

static void __kmpc_generic_kernel_init() {
  inititializeRuntime(/* IsSPMD */ false);
  // No need to wait since only the main threads will execute user
  // code and workers will run into a barrier right away.

  ASSERT(!mapping::isSPMDMode());
}

static void __kmpc_generic_kernel_deinit() {
  // Signal the workers to exit the state machine and exit the kernel.
  state::ParallelRegionFn = nullptr;
}

static void __kmpc_spmd_kernel_init() {
  inititializeRuntime(/* IsSPMD */ true);

  state::runAndCheckState(synchronize::threads);

  ASSERT(mapping::isSPMDMode());
}

static void __kmpc_spmd_kernel_deinit() {
  state::assumeInitialState(/* IsSPMD */ true);
}

extern "C" {

bool __kmpc_kernel_parallel(ParallelRegionFnTy *WorkFn);

/// Simple generic state machine for worker threads.
static void __kmpc_target_region_state_machine(IdentTy *Ident) {

  uint32_t TId = mapping::getThreadIdInBlock();

  do {
    ParallelRegionFnTy WorkFn = 0;

    // Wait for the signal that we have a new work function.
    synchronize::threads();

    // Retrieve the work function from the runtime.
    bool IsActive = __kmpc_kernel_parallel(&WorkFn);

    // If there is nothing more to do, break out of the state machine by
    // returning to the caller.
    if (!WorkFn)
      return;

    if (IsActive) {
      ASSERT(!mapping::isSPMDMode());
      ((void(*)(uint32_t,uint32_t))WorkFn)(0, TId);
      ASSERT(!mapping::isSPMDMode());
      state::resetStateForThread(TId);
    }

    synchronize::threads();

  } while (TId > 0);
}

/// Initialization
///
/// \param Ident               Source location identification, can be NULL.
///
int8_t __kmpc_target_init(IdentTy *Ident, bool IsSPMD,
                          bool UseGenericStateMachine) {
  if (IsSPMD)
    __kmpc_spmd_kernel_init();
  else
    __kmpc_generic_kernel_init();

  synchronize::threads();
  state::assumeInitialState(IsSPMD);

  uint32_t TId = mapping::getThreadIdInBlock();
  if (UseGenericStateMachine && !mapping::isMainThreadInGenericMode())
    __kmpc_target_region_state_machine(Ident);

  return TId;
}

/// De-Initialization
///
/// In non-SPMD, this function releases the workers trapped in a state machine
/// and also any memory dynamically allocated by the runtime.
///
/// \param Ident Source location identification, can be NULL.
///
void __kmpc_target_deinit(IdentTy *Ident, bool IsSPMD) {
  if (IsSPMD)
    __kmpc_spmd_kernel_deinit();
  else
    __kmpc_generic_kernel_deinit();
}

int8_t __kmpc_is_spmd_exec_mode() { return mapping::isSPMDMode(); }
}

#pragma omp end declare target
