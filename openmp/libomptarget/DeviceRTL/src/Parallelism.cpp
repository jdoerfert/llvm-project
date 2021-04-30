//===---- Parallelism.cpp - OpenMP GPU parallel implementation ---- C++ -*-===//
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

#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Types.h"

using namespace _OMP;

#pragma omp declare target

namespace {

uint32_t determineNumberOfThreads(int32_t NumThreadsClause) {
  uint32_t NThreadsICV = NumThreadsClause != -1 ? NumThreadsClause : icv::NThreads;
  uint32_t NumThreads = mapping::getBlockSize();

  if (NThreadsICV != 0 && NThreadsICV < NumThreads)
    NumThreads = NThreadsICV;

  // Round down to a multiple of WARPSIZE since it is legal to do so in OpenMP.
  if (NumThreads < mapping::getWarpSize())
    NumThreads = 1;
  else
    NumThreads = (NumThreads & ~((uint32_t)mapping::getWarpSize() - 1));

  return NumThreads;
}

// Invoke an outlined parallel function unwrapping arguments (up
// to 32).
void __kmp_invoke_microtask(kmp_int32 global_tid, kmp_int32 bound_tid, void *fn,
                            void **args, size_t nargs) {
  switch (nargs) {
#include "generated_microtask_cases.gen"
  default:
    printf("Too many arguments in kmp_invoke_microtask, aborting execution.\n");
    __builtin_trap();
  }
}

} // namespace

extern "C" {

void __kmpc_target_region_state_machine();

__attribute__((flatten, always_inline))
void __kmpc_parallel_51(IdentTy *ident, int32_t global_tid,
                                int32_t if_expr, int32_t num_threads,
                                int proc_bind, void *fn, void *wrapper_fn,
                                void **args, int64_t nargs) {
   //printf("enter parallelready fn %p, wrapper %p, #T %i, #A %li\n", fn, wrapper_fn, num_threads, nargs);
  //int32_t global_tid = mapping::getThreadIdInBlock();

  uint32_t NumThreads = determineNumberOfThreads(num_threads);
   if (mapping::isSPMDMode()) {
     if (global_tid < NumThreads) {
       //////printf("spmd mode invoke, fn %p, wrapper %p, %i\n", fn, wrapper_fn, NumThreads);
       __kmp_invoke_microtask(global_tid, 0, fn, args, nargs);
     }
     synchronize::threads();
     return;
   }


   // Handle the serialized case first, same for SPMD/non-SPMD.
   // TODO: Add UNLIKELY to optimize?
   if (!if_expr || icv::ActiveLevel) {
     __kmpc_serialized_parallel(ident, global_tid);
     __kmp_invoke_microtask(global_tid, 0, fn, args, nargs);
     __kmpc_end_serialized_parallel(ident, global_tid);
     return;
   }

   // Handle the num_threads clause.
   state::ParallelTeamSize = NumThreads;
   state::ParallelRegionFn = wrapper_fn;
   // We do *not* create a new data environment because all threads in the team
   // that are active are now running this parallel region. They share the
  // TeamState, which has an increase level-var and potentially active-level
  // set, but they do not have individual ThreadStates yet. If they ever
  // modify the ICVs beyond this point a ThreadStates will be allocated.
  int NewLevel = ++icv::Level;
  bool IsActiveParallelRegion = NumThreads > 1;
  if (!IsActiveParallelRegion) {
    __kmp_invoke_microtask(global_tid, 0, fn, args, nargs);
  } else {
    icv::ActiveLevel = NewLevel;

    void **GlobalArgs = nullptr;
    if (nargs) {
      __kmpc_begin_sharing_variables(&GlobalArgs, nargs);
      //printf("alloc global args %p %u\n", GlobalArgs, mapping::activemask());
      // TODO: faster memcpy?
      for (int I = 0; I < nargs; I++)
        GlobalArgs[I] = args[I];
    }

    // Master signals work to activate workers.
    //printf("main (%i) ready fn %p, wrapper %p, %i, %u\n", global_tid, fn, wrapper_fn, NumThreads, mapping::activemask());
    __kmpc_target_region_state_machine();
    //state::runAndCheckState(synchronize::threads);

    ////printf("main invoke fn %p, wrapper %p, %i\n", fn, wrapper_fn, NumThreads);
    //__kmp_invoke_microtask(global_tid, 0, fn, args, nargs);

    //// OpenMP [2.5, Parallel Construct, p.49]
    //// There is an implied barrier at the end of a parallel region. After the
    //// end of a parallel region, only the master thread of the team resumes
    //// execution of the enclosing task region.
    ////
    //// The master waits at this barrier until all workers are done.
    //printf("main done fn %p, wrapper %p, %i, %u\n", fn, wrapper_fn, NumThreads, mapping::activemask());
    //state::runAndCheckState(synchronize::threads);
    //printf("main alone fn %p, wrapper %p, %i\n", fn, wrapper_fn, NumThreads);


    if (nargs) {
      //printf("frees global args %p %u\n", GlobalArgs, mapping::activemask());
      memory::freeShared(GlobalArgs, "global args free shared");
    }
    //__kmpc_end_sharing_variables(GlobalArgs);
    //printf("main adjust fn %p, wrapper %p, %i, %u\n", fn, wrapper_fn, NumThreads, mapping::activemask());

    icv::ActiveLevel = 0;
  }

  --icv::Level;

  state::ParallelTeamSize = 1;

   // TODO: proc_bind is a noop?
   // if (proc_bind != proc_bind_default)
   //  __kmpc_push_proc_bind(ident, global_tid, proc_bind);
   //printf("exit parallelready fn %p, wrapper %p, %i\n", fn, wrapper_fn, num_threads);
}

bool __kmpc_kernel_parallel(ParallelRegionFnTy *WorkFn) {
  // Work function and arguments for L1 parallel region.
  *WorkFn = state::ParallelRegionFn;

  // If this is the termination signal from the master, quit early.
  if (!*WorkFn) {
    return false;
  }

  // Set to true for workers participating in the parallel region.
  uint32_t TId = mapping::getThreadIdInBlock();
  bool ThreadIsActive = TId < state::ParallelTeamSize;
  return ThreadIsActive;
}

void __kmpc_kernel_end_parallel() __attribute__((used)) {
  // In case we have modified an ICV for this thread before a ThreadState was
  // created. We drop it now to not contaminate the next parallel region.
  ASSERT(!mapping::isSPMDMode());
  uint32_t TId = mapping::getThreadIdInBlock();
  state::resetStateForThread(TId);
  ASSERT(!mapping::isSPMDMode());
}

void __kmpc_serialized_parallel(IdentTy *, uint32_t TId) {
  state::enterDataEnvironment();
  ++icv::Level;
}

void __kmpc_end_serialized_parallel(IdentTy *, uint32_t TId) {
  state::exitDataEnvironment();
  --icv::Level;
}

uint16_t __kmpc_parallel_level(IdentTy *, uint32_t) { return omp_get_level(); }

int32_t __kmpc_global_thread_num(IdentTy *) { return omp_get_thread_num(); }

void __kmpc_push_num_threads(IdentTy *, int32_t, int32_t NumThreads) {
  icv::NThreads = NumThreads;
}

void __kmpc_push_num_teams(IdentTy *loc, int32_t tid, int32_t num_teams,
                           int32_t thread_limit) {}

void __kmpc_push_proc_bind(IdentTy *loc, uint32_t tid, int proc_bind) {}
}

#pragma omp end declare target
