//===------------ libcall.cu - OpenMP GPU user calls ------------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenMP runtime functions that can be
// invoked by the user in an OpenMP region
//
//===----------------------------------------------------------------------===//

#include "ICVs.h"
#include "Mapping.h"
#include "TeamState.h"
#include "ThreadState.h"
#include "omptarget.h"
#include "support.h"
#include "target_interface.h"
#include <string.h>

#define ICV_DEBUG(...)

using namespace omp;

#pragma omp declare target

template <typename Ty> static Ty &getICVForThreadImpl(Ty ICVStateTy::*Var) {
  unsigned TId = getThreadIdForSharedMemArrayAccess();
  if (!ThreadStates[TId])
    return TeamState.ICVState.*Var;

  return ThreadStates[TId]->ICVState.*Var;
}

int &ICVStateTy::getICVForThread(int ICVStateTy::*Var) {
  return getICVForThreadImpl<int>(Var);
}

template <typename Ty>
static Ty incICVForThreadImpl(Ty ICVStateTy::*Var, Ty UpdateVal) {
  unsigned TId = getThreadIdForSharedMemArrayAccess();
  ICVStateTy::ensureICVStateForThread(TId);

  ThreadStates[TId]->ICVState.*Var += UpdateVal;

  return ThreadStates[TId]->ICVState.*Var;
}
int ICVStateTy::incICVForThread(int ICVStateTy::*Var, int UpdateVal) {
  return incICVForThreadImpl<int>(Var, UpdateVal);
}

template <typename Ty>
static Ty setICVForThreadImpl(Ty ICVStateTy::*Var, Ty UpdateVal) {
  unsigned TId = getThreadIdForSharedMemArrayAccess();
  ICVStateTy::ensureICVStateForThread(TId);

  ThreadStates[TId]->ICVState.*Var = UpdateVal;

  return ThreadStates[TId]->ICVState.*Var;
}
int ICVStateTy::setICVForThread(int ICVStateTy::*Var, int UpdateVal) {
  return setICVForThreadImpl<int>(Var, UpdateVal);
}

bool ICVStateTy::ensureICVStateForThread(unsigned TId) {
  if (ThreadStates[TId])
    return false;

  ThreadStates[TId] =
      static_cast<ThreadStateTy *>(malloc(sizeof(ThreadStateTy)));
  ThreadStates[TId]->init();
  return true;
}

void omp_set_dynamic(int V) { ICV_DEBUG("(%i); ignored", V); }
int omp_get_dynamic(void) {
  ICV_DEBUG("0; constant");
  return 0;
}

void omp_set_num_threads(int V) {
  if (isMainThreadInGenericMode()) {
    ICV_DEBUG("(%i); stored in team state", V);
    TeamState.ICVState.nthreads_var = V;
    return;
  }

  unsigned TId = getThreadIdForSharedMemArrayAccess();
  if (!ThreadStates[TId] && TeamState.ICVState.nthreads_var == V) {
    ICV_DEBUG("(%i); equal to team state setting, ignored", V);
    return;
  }

  ICVStateTy::ensureICVStateForThread(TId);

  ICV_DEBUG("(%i); set nthreads-var ICV for thread", V);
  ThreadStates[TId]->ICVState.nthreads_var = V;
}

int omp_get_max_threads(void) {
  return ICVStateTy::getICVForThread(&ICVStateTy::nthreads_var);
}

/// TODO not all functions below belong here.

int omp_get_level(void) {
  int LevelsVar = ICVStateTy::getICVForThread(&ICVStateTy::levels_var);
  __builtin_assume(LevelsVar >= 0);
  return LevelsVar;
}

int omp_get_active_level(void) {
  return !!ICVStateTy::getICVForThread(&ICVStateTy::active_level);
}

int omp_in_parallel(void) {
  return !!ICVStateTy::getICVForThread(&ICVStateTy::active_level);
}

void omp_get_schedule(omp_sched_t *ScheduleKind, int *ChunkSize) {
  uint64_t RunSchedVar =
      ICVStateTy::getICVForThread(&ICVStateTy::run_sched_var);
  ICVStateTy::RunSchedVarEncodingTy RunSchedVarEncoding;
  memcpy(&RunSchedVarEncoding, &RunSchedVar, sizeof(RunSchedVarEncoding));
  *ScheduleKind = RunSchedVarEncoding.ScheduleKind;
  *ChunkSize = RunSchedVarEncoding.ChunkSize;
}

void omp_set_schedule(omp_sched_t ScheduleKind, int ChunkSize) {
  ICVStateTy::RunSchedVarEncodingTy RunSchedVarEncoding;
  RunSchedVarEncoding.ScheduleKind = ScheduleKind;
  RunSchedVarEncoding.ChunkSize = ChunkSize;
  uint64_t RunSchedVar;
  memcpy(&RunSchedVar, &RunSchedVarEncoding, sizeof(RunSchedVarEncoding));
  ICVStateTy::setICVForThread(&ICVStateTy::run_sched_var, RunSchedVar);
}

static int returnValIfLevelIsActive(int Level, int Val, int DefaultVal,
                                    int OutOfBoundsVal = -1) {
  if (Level == 0)
    return DefaultVal;
  int LevelsVar = omp_get_level();
  if (Level < 0 || Level > LevelsVar)
    return OutOfBoundsVal;
  int ActiveLevel = ICVStateTy::getICVForThread(&ICVStateTy::active_level);
  if (Level != ActiveLevel)
    return DefaultVal;
  return Val;
}

int omp_get_ancestor_thread_num(int Level) {
  return returnValIfLevelIsActive(Level, getThreadIdForSharedMemArrayAccess(),
                                  0);
}

int omp_get_thread_num(void) {
  return omp_get_ancestor_thread_num(omp_get_level());
}

int omp_get_team_size(int Level) {
  return returnValIfLevelIsActive(Level, TeamState.ParallelTeamSize, 1);
}

int omp_get_num_threads(void) { return omp_get_team_size(omp_get_level()); }

void ThreadStateTy::enterDataEnvironment() {
  unsigned TId = getThreadIdForSharedMemArrayAccess();
  if (ICVStateTy::ensureICVStateForThread(TId))
    return;

  ThreadStateTy *NewThreadState =
      static_cast<ThreadStateTy *>(malloc(sizeof(ThreadStateTy)));
  NewThreadState->init(*ThreadStates[TId]);
  ThreadStates[TId] = NewThreadState;
}

void ThreadStateTy::exitDataEnvironment() {
  unsigned TId = getThreadIdForSharedMemArrayAccess();
  // assert(ThreadStates[TId] && "exptected thread state");
  free(ThreadStates[TId]);
  ThreadStates[TId] = ThreadStates[TId]->PreviousThreadState;
}

bool omp::isMainThreadInGenericMode() {
  if (isSPMDMode())
    return false;

  int TId = GetThreadIdInBlock();
  return TId == ((GetNumberOfThreadsInBlock() - 1) & ~(WARPSIZE - 1));
}

bool omp::isLeaderInSIMD() {
  __kmpc_impl_lanemask_t Active = __kmpc_impl_activemask();
  __kmpc_impl_lanemask_t LaneMaskLT = __kmpc_impl_lanemask_lt();
  unsigned int Position = __kmpc_impl_popc(Active & LaneMaskLT);
  return Position == 0;
}

unsigned omp::getNumberOfThreadsAccessingSharedMem() {
  return GetNumberOfThreadsInBlock();
}

unsigned omp::getThreadIdForSharedMemArrayAccess() {
  return GetThreadIdInBlock();
}

void ThreadStateTy::dropForThread(unsigned TId) {
  if (!ThreadStates[TId])
    return;

  // assert(!ThreadStates[TId]->PreviousThreadState && "leftover thread state");
  free(ThreadStates[TId]);
  ThreadStates[TId] = nullptr;
}

TeamStateTy SHARED(omp::TeamState);

[[clang::loader_uninitialized]] ThreadStateTy
    *omp::ThreadStates[MAX_THREADS_PER_TEAM];
#pragma omp allocate(omp::ThreadStates) allocator(omp_pteam_mem_alloc)

#pragma omp end declare target
