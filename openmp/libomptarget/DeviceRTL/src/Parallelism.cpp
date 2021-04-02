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

uint32_t determineNumberOfThreads() {
  uint32_t NThreadsICV = icv::NThreads;
  uint32_t NumThreads = mapping::getBlockSize();

  // In non-SPMD mode, we need to substract the warp for the master thread
  if (!mapping::isSPMDMode())
    NumThreads -= mapping::getWarpSize();

  if (NThreadsICV != 0 && NThreadsICV < NumThreads)
    NumThreads = NThreadsICV;

  // Round down to a multiple of WARPSIZE since it is legal to do so in OpenMP.
  if (NumThreads < mapping::getWarpSize())
    NumThreads = 1;
  else
    NumThreads = (NumThreads & ~((uint32_t)mapping::getWarpSize() - 1));

  return NumThreads;
}

} // namespace

extern "C" {

void __kmpc_kernel_prepare_parallel(ParallelRegionFnTy WorkFn) {

  uint32_t NumThreads = determineNumberOfThreads();
  state::ParallelTeamSize = NumThreads;
  state::ParallelRegionFn = WorkFn;

  // We do *not* create a new data environment because all threads in the team
  // that are active are now running this parallel region. They share the
  // TeamState, which has an increase level-var and potentially active-level
  // set, but they do not have individual ThreadStates yet. If they ever
  // modify the ICVs beyond this point a ThreadStates will be allocated.
  int NewLevel = ++icv::Level;
  bool IsActiveParallelRegion = NumThreads > 1;
  if (IsActiveParallelRegion)
    icv::ActiveLevel = NewLevel;
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

void __kmpc_kernel_end_parallel() {
  // We did *not* create a new data environment because all threads in the team
  // that were active were running the parallel region. We used the TeamState
  // which needs adjustment now.
  // --icv::Level;
  // bool IsActiveParallelRegion = state::ParallelTeamSize;
  // if (IsActiveParallelRegion)
  //   icv::ActiveLevel = 0;

  // state::ParallelTeamSize = 1;

  // In case we have modified an ICV for this thread before a ThreadState was
  // created. We drop it now to not contaminate the next parallel region.
  state::resetStateForThread();
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
