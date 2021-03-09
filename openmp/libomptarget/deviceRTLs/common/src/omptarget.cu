//===--- omptarget.cu - OpenMP GPU initialization ---------------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the initialization code for the GPU
//
//===----------------------------------------------------------------------===//
#pragma omp declare target

#include "TeamState.h"
#include "ThreadState.h"
#include "common/omptarget.h"
#include "target_impl.h"
#include "target_interface.h"

////////////////////////////////////////////////////////////////////////////////
// init entry points
////////////////////////////////////////////////////////////////////////////////

static void initTeamState(bool IsSPMDExecutionMode) {
  int threadId = GetLogicalThreadIdInBlock(IsSPMDExecutionMode);

  if (threadId != 0)
    return;

  omp::TeamState.ICVState.nthreads_var = GetNumberOfThreadsInBlock();
  omp::TeamState.ICVState.levels_var = 0;
  omp::TeamState.ICVState.active_level = -1;
  omp::TeamState.ParallelTeamSize = -1;
  for (int i = 0; i < GetNumberOfThreadsInBlock() ;++i) {
    omp::ThreadStates[i] = 0;
  }
}

EXTERN void __kmpc_kernel_init(int, int16_t) {
  PRINT(LD_IO, "call to __kmpc_kernel_init with version %f\n",
        OMPTARGET_NVPTX_VERSION);
  setExecutionParameters(Generic, RuntimeInitialized);

  int threadIdInBlock = GetThreadIdInBlock();
  ASSERT0(LT_FUSSY, threadIdInBlock == GetMasterThreadID(),
          "__kmpc_kernel_init() must be called by team master warp only!");
  PRINT0(LD_IO, "call to __kmpc_kernel_init for master\n");

  // init thread private
  initTeamState(/* IsSPMDExecutionMode */ false);
  scratchpad.init();

  __kmpc_impl_target_init();

}

EXTERN void __kmpc_kernel_deinit(int16_t) {
  PRINT0(LD_IO, "call to __kmpc_kernel_deinit\n");

  // Done with work.  Kill the workers.
  omptarget_nvptx_workFn = 0;
}

EXTERN void __kmpc_spmd_kernel_init(int, int16_t) {
  PRINT0(LD_IO, "call to __kmpc_spmd_kernel_init\n");

  setExecutionParameters(Spmd, RuntimeInitialized);
  // init thread private
  initTeamState(/* IsSPMDExecutionMode */ true);
  scratchpad.init();
  __kmpc_impl_syncthreads();

  // init thread private from init value
  int threadId = GetThreadIdInBlock();
  PRINT(LD_PAR,
        "thread will execute parallel region with id %d in a team of "
        "%d threads\n",
        (int)threadId, (int)ThreadLimit);
}

EXTERN void __kmpc_spmd_kernel_deinit_v2(int16_t) {}

// Return true if the current target region is executed in SPMD mode.
EXTERN int8_t __kmpc_is_spmd_exec_mode() {
  PRINT0(LD_IO | LD_PAR, "call to __kmpc_is_spmd_exec_mode\n");
  return isSPMDMode();
}

#pragma omp end declare target
