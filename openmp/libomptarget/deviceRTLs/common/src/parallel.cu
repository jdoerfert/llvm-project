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

  // This routine is only called by the team master.  The team master is
  // the first thread of the last warp.  It always has the logical thread
  // id of 0 (since it is a shadow for the first worker thread).
  const int threadId = 0;
  omptarget_nvptx_TaskDescr *currTaskDescr =
      omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(threadId);
  ASSERT0(LT_FUSSY, currTaskDescr, "expected a top task descr");
  ASSERT0(LT_FUSSY, !currTaskDescr->InParallelRegion(),
          "cannot be called in a parallel region.");
  if (currTaskDescr->InParallelRegion()) {
    PRINT0(LD_PAR, "already in parallel: go seq\n");
    return;
  }

  uint16_t NumThreads = determineNumberOfThreads();
  TeamState.ParallelTeamSize = NumThreads;

  ASSERT0(LT_FUSSY, GetThreadIdInBlock() == GetMasterThreadID(),
          "only team master can create parallel");

  // Set number of threads on work descriptor.
  omptarget_nvptx_WorkDescr &workDescr = getMyWorkDescriptor();
  workDescr.WorkTaskDescr()->CopyToWorkDescr(currTaskDescr);
}

// All workers call this function.  Deactivate those not needed.
// Fn - the outlined work function to execute.
// returns True if this thread is active, else False.
//
// Only the worker threads call this routine.
EXTERN bool __kmpc_kernel_parallel(void **WorkFn) {
  PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_parallel\n");

  // Work function and arguments for L1 parallel region.
  *WorkFn = omptarget_nvptx_workFn;

  // If this is the termination signal from the master, quit early.
  if (!*WorkFn) {
    PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_parallel finished\n");
    return false;
  }

  uint16_t NumThreads = determineNumberOfThreads();

  // Only the worker threads call this routine and the master warp
  // never arrives here.  Therefore, use the nvptx thread id.
  int threadId = GetThreadIdInBlock();
  omptarget_nvptx_WorkDescr &workDescr = getMyWorkDescriptor();
  // Set to true for workers participating in the parallel region.
  bool ThreadIsActive = threadId < NumThreads;
  // Initialize state for active threads.
  if (ThreadIsActive) {
    // init work descriptor from workdesccr
    omptarget_nvptx_TaskDescr *newTaskDescr =
        omptarget_nvptx_threadPrivateContext->Level1TaskDescr(threadId);
    ASSERT0(LT_FUSSY, newTaskDescr, "expected a task descr");
    newTaskDescr->CopyFromWorkDescr(workDescr.WorkTaskDescr());
    // install new top descriptor
    omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(threadId,
                                                               newTaskDescr);
    // init private from int value
    PRINT(LD_PAR,
          "thread will execute parallel region with id %d in a team of "
          "%d threads\n",
          (int)newTaskDescr->ThreadId(), (int)nThreads);

    ThreadStateTy::enterDataEnvironment();
    unsigned Level = ICVStateTy::incICVForThread(&ICVStateTy::levels_var, 1);
    bool IsActiveParallelRegion = NumThreads > 1;
    if (IsActiveParallelRegion)
      ICVStateTy::setICVForThread(&ICVStateTy::active_level, Level);
  }

  return ThreadIsActive;
}

EXTERN void __kmpc_kernel_end_parallel() {
  // pop stack
  PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_end_parallel\n");
  ASSERT0(LT_FUSSY, isRuntimeInitialized(), "Expected initialized runtime.");

  // Only the worker threads call this routine and the master warp
  // never arrives here.  Therefore, use the nvptx thread id.
  int threadId = GetThreadIdInBlock();
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(threadId);
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(
      threadId, currTaskDescr->GetPrevTaskDescr());

  ThreadStateTy::exitDataEnvironment();
  ICVStateTy::incICVForThread(&ICVStateTy::levels_var, -1);
  bool IsActiveParallelRegion = omp_get_num_threads() > 1;
  if (IsActiveParallelRegion)
    ICVStateTy::setICVForThread(&ICVStateTy::active_level, 0);
}

////////////////////////////////////////////////////////////////////////////////
// support for parallel that goes sequential
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_serialized_parallel(kmp_Ident *loc, uint32_t global_tid) {
  PRINT0(LD_IO, "call to __kmpc_serialized_parallel\n");

  ThreadStateTy::enterDataEnvironment();
  ICVStateTy::incICVForThread(&ICVStateTy::levels_var, 1);

  if (checkRuntimeUninitialized(loc)) {
    ASSERT0(LT_FUSSY, checkSPMDMode(loc),
            "Expected SPMD mode with uninitialized runtime.");
    return;
  }

  // assume this is only called for nested parallel
  int threadId = GetLogicalThreadIdInBlock(checkSPMDMode(loc));

  // unlike actual parallel, threads in the same team do not share
  // the workTaskDescr in this case and num threads is fixed to 1

  // get current task
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(threadId);
  currTaskDescr->SaveLoopData();

  // allocate new task descriptor and copy value from current one, set prev to
  // it
  omptarget_nvptx_TaskDescr *newTaskDescr =
      (omptarget_nvptx_TaskDescr *)SafeMalloc(sizeof(omptarget_nvptx_TaskDescr),
                                              "new seq parallel task");
  newTaskDescr->CopyParent(currTaskDescr);

  // tweak values for serialized parallel case:
  // - each thread becomes ID 0 in its serialized parallel, and
  // - there is only one thread per team
  newTaskDescr->ThreadId() = 0;

  // set new task descriptor as top
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(threadId,
                                                             newTaskDescr);
}

EXTERN void __kmpc_end_serialized_parallel(kmp_Ident *loc,
                                           uint32_t global_tid) {
  PRINT0(LD_IO, "call to __kmpc_end_serialized_parallel\n");

  ThreadStateTy::exitDataEnvironment();
  ICVStateTy::incICVForThread(&ICVStateTy::levels_var, -1);

  if (checkRuntimeUninitialized(loc)) {
    ASSERT0(LT_FUSSY, checkSPMDMode(loc),
            "Expected SPMD mode with uninitialized runtime.");
    return;
  }

  // pop stack
  int threadId = GetLogicalThreadIdInBlock(checkSPMDMode(loc));
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(threadId);
  // set new top
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(
      threadId, currTaskDescr->GetPrevTaskDescr());
  // free
  SafeFree(currTaskDescr, "new seq parallel task");
  currTaskDescr = getMyTopTaskDescriptor(threadId);
  currTaskDescr->RestoreLoopData();
}

// This kmpc call returns the thread id across all teams. It's value is
// cached by the compiler and used when calling the runtime. On nvptx
// it's cheap to recalculate this value so we never use the result
// of this call.
EXTERN int32_t __kmpc_global_thread_num(kmp_Ident *loc) {
  return omp_get_thread_num();
}

////////////////////////////////////////////////////////////////////////////////
// push params
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_push_num_threads(kmp_Ident *Loc, int32_t TId,
                                    int32_t NumThreads) {
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
