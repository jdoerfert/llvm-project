//===---- parallel.cu - GPU OpenMP parallel implementation ------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Parallel implementation in the GPU. Here is the pattern:
//
//    while (not finished) {
//
//    if (master) {
//      sequential code, decide which par loop to do, or if finished
//     __kmpc_kernel_prepare_parallel() // exec by master only
//    }
//    syncthreads // A
//    __kmpc_kernel_parallel() // exec by all
//    if (this thread is included in the parallel) {
//      switch () for all parallel loops
//      __kmpc_kernel_end_parallel() // exec only by threads in parallel
//    }
//
//
//    The reason we don't exec end_parallel for the threads not included
//    in the parallel loop is that for each barrier in the parallel
//    region, these non-included threads will cycle through the
//    syncthread A. Thus they must preserve their current threadId that
//    is larger than thread in team.
//
//    To make a long story short...
//
//===----------------------------------------------------------------------===//

#include "ICVs.h"
#include "TeamState.h"
#include "ThreadState.h"
#include "interface.h"
#pragma omp declare target

#include "common/omptarget.h"
#include "target_impl.h"

using namespace omp;

////////////////////////////////////////////////////////////////////////////////
// support for parallel that goes parallel (1 static level only)
////////////////////////////////////////////////////////////////////////////////

INLINE static uint16_t determineNumberOfThreads() {
  int NThreadsICV = ICVStateTy::getICVForThread(&ICVStateTy::nthreads_var);

  uint16_t NumThreads = GetNumberOfWorkersInTeam();
  if (NThreadsICV != 0 && NThreadsICV < NumThreads) {
    NumThreads = NThreadsICV;
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  // On Volta and newer architectures we require that all lanes in
  // a warp participate in the parallel region.  Round down to a
  // multiple of WARPSIZE since it is legal to do so in OpenMP.
  if (NumThreads < WARPSIZE) {
    NumThreads = 1;
  } else {
    NumThreads = (NumThreads & ~((uint16_t)WARPSIZE - 1));
  }
#endif

  return NumThreads;
}

// This routine is always called by the team master..
EXTERN void __kmpc_kernel_prepare_parallel(void *WorkFn) {
  PRINT0(LD_IO, "call to __kmpc_kernel_prepare_parallel\n");

  omptarget_nvptx_workFn = WorkFn;

  uint16_t NumThreads = determineNumberOfThreads();
  TeamState.ParallelTeamSize = NumThreads;

  ASSERT0(LT_FUSSY, GetThreadIdInBlock() == GetMasterThreadID(),
          "only team master can create parallel");

  // We do *not* create a new data environment because all threads in the team
  // that are active are now running this parallel region. They share the
  // TeamState, which has an increase level-var and potentially active-level
  // set, but they do not have individual ThreadStates yet. If they ever
  // modify the ICVs beyond this point a ThreadStates will be allocated.
  auto Level = TeamState.ICVState.levels_var += 1;
  bool IsActiveParallelRegion = NumThreads > 1;
  if (IsActiveParallelRegion)
    TeamState.ICVState.active_level = Level;
}

// All workers call this function.  Deactivate those not needed.
// Fn - the outlined work function to execute.
// returns True if this thread is active, else False.
//
// Only the worker threads call this routine.
EXTERN bool __kmpc_kernel_parallel(void **WorkFn) {
  PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_parallel\n");

  // In case we have modified an ICV for this thread before a ThreadState was
  // created. We drop it now to not contaminate the next parallel region.
  int threadId = GetThreadIdInBlock();
  ThreadStateTy::dropForThread(threadId);

  // Work function and arguments for L1 parallel region.
  *WorkFn = omptarget_nvptx_workFn;

  // If this is the termination signal from the master, quit early.
  if (!*WorkFn) {
    PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_parallel finished\n");
    return false;
  }

  uint16_t NumThreads = determineNumberOfThreads();

  // Set to true for workers participating in the parallel region.
  bool ThreadIsActive = threadId < NumThreads;
  return ThreadIsActive;
}

EXTERN void __kmpc_kernel_end_parallel() {
  // pop stack
  PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_end_parallel\n");

  // We did *not* create a new data environment because all threads in the team
  // that were active were running the parallel region. We used the TeamState
  // which needs adjustment now.
  TeamState.ICVState.levels_var -= 1;
  bool IsActiveParallelRegion = omp_get_num_threads() > 1;
  if (IsActiveParallelRegion)
    TeamState.ICVState.active_level = 0;
}

////////////////////////////////////////////////////////////////////////////////
// support for parallel that goes sequential
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_serialized_parallel(kmp_Ident *, uint32_t) {
  PRINT0(LD_IO, "call to __kmpc_serialized_parallel\n");

  ThreadStateTy::enterDataEnvironment();
  ICVStateTy::incICVForThread(&ICVStateTy::levels_var, 1);
}

EXTERN void __kmpc_end_serialized_parallel(kmp_Ident *, uint32_t) {
  PRINT0(LD_IO, "call to __kmpc_end_serialized_parallel\n");

  ThreadStateTy::exitDataEnvironment();
  ICVStateTy::incICVForThread(&ICVStateTy::levels_var, -1);
}

EXTERN uint16_t __kmpc_parallel_level(kmp_Ident *, uint32_t) {
  PRINT0(LD_IO, "call to __kmpc_parallel_level\n");
  return omp_get_level();
}

// This kmpc call returns the thread id across all teams. It's value is
// cached by the compiler and used when calling the runtime. On nvptx
// it's cheap to recalculate this value so we never use the result
// of this call.
EXTERN int32_t __kmpc_global_thread_num(kmp_Ident *) {
  return omp_get_thread_num();
}

////////////////////////////////////////////////////////////////////////////////
// push params
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_push_num_threads(kmp_Ident *, int32_t, int32_t NumThreads) {
  ICVStateTy::setICVForThread(&ICVStateTy::nthreads_var, NumThreads);
}

// Do nothing. The host guarantees we started the requested number of
// teams and we only need inspection of gridDim.

EXTERN void __kmpc_push_num_teams(kmp_Ident *loc, int32_t tid,
                                  int32_t num_teams, int32_t thread_limit) {
  PRINT(LD_IO, "call kmpc_push_num_teams %d\n", (int)num_teams);
  ASSERT0(LT_FUSSY, 0, "should never have anything with new teams on device");
}

EXTERN void __kmpc_push_proc_bind(kmp_Ident *loc, uint32_t tid, int proc_bind) {
  PRINT(LD_IO, "call kmpc_push_proc_bind %d\n", (int)proc_bind);
}

#pragma omp end declare target
