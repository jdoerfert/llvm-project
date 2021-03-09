//===--------- ICVs.h - OpenMP ICV handling ----------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_ICVS_H
#define OMPTARGET_ICVS_H

#include "Types.h"

struct ICVStateTy {
  int nthreads_var;

  int levels_var;

  /// The `active-level` describes which of the parallel levels counted with the
  /// `levels-var` is active. There can only be one.
  ///
  /// active-levels-var is 1, if active_level is not 0, otherweise it is 0.
  int active_level;

  /// Don't use UB type punning!
  ///
  ///{
  struct RunSchedVarEncodingTy {
    omp_sched_t ScheduleKind;
    int ChunkSize;
  };

  uint64_t run_sched_var;

  static_assert(
      sizeof(run_sched_var) == sizeof(RunSchedVarEncodingTy),
      "Schedule encoding is supposed to cover the entire run-sched-var ICV!");
  ///}

  static bool ensureICVStateForThread(unsigned TId);

  static int &getICVForThread(int ICVStateTy::*Var);
  static int incICVForThread(int ICVStateTy::*Var, int UpdateVal);
  static int setICVForThread(int ICVStateTy::*Var, int UpdateVal);

  static uint64_t &getICVForThread(uint64_t ICVStateTy::*Var);
  static uint64_t incICVForThread(uint64_t ICVStateTy::*Var,
                                  uint64_t UpdateVal);
  static uint64_t setICVForThread(uint64_t ICVStateTy::*Var,
                                  uint64_t UpdateVal);
};

#ifdef __cplusplus
extern "C" {
#endif

#pragma omp declare target

/// ICV: dyn-var, constant 0
///
/// setter: ignored.
/// getter: returns 0.
///
///{
void omp_set_dynamic(int);
int omp_get_dynamic(void);
///}

/// ICV: nthreads-var, integer
///
/// scope: data environment
///
/// setter: ignored.
/// getter: returns false.
///
/// implementation notes:
///
///
///{
void omp_set_num_threads(int);
int omp_get_max_threads(void);
///}

/// ICV: thread-limit-var, computed
///
/// getter: returns thread limited defined during launch.
///
///{
int omp_get_thread_limit(void);
///}

/// ICV: max-active-levels-var, constant 1
///
/// setter: ignored.
/// getter: returns 1.
///
///{
void omp_set_max_active_levels(int);
int omp_get_max_active_levels(void);
///}

/// ICV: places-partition-var
///
///
///{
///}

/// ICV: active-levels-var, 0 or 1
///
/// getter: returns 0 or 1.
///
///{
int omp_get_active_levels(void);
///}

/// ICV: levels-var
///
/// getter: returns parallel region nesting
///
///{
int omp_get_levels(void);
///}

/// ICV: run-sched-var
///
///
///{
void omp_set_schedule(omp_sched_t, int);
void omp_get_schedule(omp_sched_t *, int*);
///}

/// TODO this is incomplete.
int omp_get_num_threads(void);
int omp_get_thread_num(void);

#pragma omp end declare target

#ifdef __cplusplus
}
#endif

#endif
