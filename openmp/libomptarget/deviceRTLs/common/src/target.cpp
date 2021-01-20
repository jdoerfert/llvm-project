//===-- target.cpp -------- OpenMP device runtime target implementation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "target.h"


int32_t GetThreadIdInBlock();
int32_t GetMasterThreadID();
int32_t GetNumberOfThreadsInBlock();

extern "C" {

int32_t __kmpc_global_thread_num(ident_t *);
void __kmpc_spmd_kernel_init(int32_t, int16_t);
void __kmpc_kernel_init(int32_t, int16_t);
void __kmpc_kernel_deinit(int16_t);
void __kmpc_spmd_kernel_deinit_v2(int16_t);
void __kmpc_data_sharing_init_stack();
void __kmpc_data_sharing_init_stack_spmd();
bool __kmpc_kernel_parallel(void **);
void __kmpc_kernel_end_parallel();
void __kmpc_barrier_simple_spmd(ident_t *, int32_t);

#define WARPSIZE 32

using ParallelWorkFnTy = void (*)(int16_t, int32_t);

/// Simple generic state machine for worker threads.
static void __kmpc_target_region_state_machine(ident_t *Ident) {

  int32_t gtid = __kmpc_global_thread_num(Ident);

  do {
    void *WorkFn = 0;

    // Wait for the signal that we have a new work function.
    __kmpc_barrier_simple_spmd(Ident, gtid);

    // Retrieve the work function from the runtime.
    bool IsActive = __kmpc_kernel_parallel(&WorkFn);

    // If there is nothing more to do, break out of the state machine by
    // returning to the caller.
    if (!WorkFn)
      return;

    if (IsActive) {
      ((ParallelWorkFnTy)WorkFn)(0, gtid);

      __kmpc_kernel_end_parallel();
    }

    __kmpc_barrier_simple_spmd(Ident, gtid);

  } while (true);
}

/// Filter threads into main and worker. The return value indicates if the
/// thread is a main thread (1), a surplus thread (0), or a worker that is not
/// needed anymore (-1).
static int8_t __kmpc_target_region_thread_filter(ident_t *Ident,
                                                 uint32_t ThreadLimit,
                                                 bool UseGenericStateMachine) {

  uint32_t TId = GetThreadIdInBlock();
  bool IsWorker = TId < ThreadLimit;

  if (IsWorker) {
    if (UseGenericStateMachine)
      __kmpc_target_region_state_machine(Ident);
    return -1;
  }

  return TId == GetMasterThreadID();
}

int8_t __kmpc_target_init(ident_t *Ident, bool IsSPMD,
                          bool UseGenericStateMachine) {
  // TODO: This is not provided by clang or derived yet.
  const int16_t RequiresOMPRuntime = 1;

  int32_t NumThreads = GetNumberOfThreadsInBlock();

  // Handle the SPMD case first.
  if (IsSPMD) {
    __kmpc_spmd_kernel_init(NumThreads, RequiresOMPRuntime);
    __kmpc_data_sharing_init_stack_spmd();
    return 1;
  }

  // Reserve one WARP in non-SPMD mode for the main.
  int32_t ThreadLimit = NumThreads - WARPSIZE;
  int8_t FilterVal = __kmpc_target_region_thread_filter(Ident, ThreadLimit,
                                                        UseGenericStateMachine);

  // If the filter returns 1 the executing thread is a team main which will
  // initialize the kernel in the following, if not, we are done here.
  if (FilterVal != 1)
    return FilterVal;

  __kmpc_kernel_init(ThreadLimit, RequiresOMPRuntime);
  __kmpc_data_sharing_init_stack();
  return 1;
}

void __kmpc_target_deinit(ident_t *Ident, bool IsSPMD) {
  // TODO: This is not provided by clang or derived yet.
  const int16_t RequiresOMPRuntime = 1;

  // Handle the SPMD case first.
  if (IsSPMD) {
    __kmpc_spmd_kernel_deinit_v2(RequiresOMPRuntime);
    return;
  }

  __kmpc_kernel_deinit(RequiresOMPRuntime);

  // Barrier to terminate worker threads.
  __kmpc_barrier_simple_spmd(Ident, 0);
}
}
