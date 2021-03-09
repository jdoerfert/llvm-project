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
#pragma omp declare target

#include "common/omptarget.h"
#include "target_impl.h"

EXTERN double omp_get_wtick(void) {
  double rc = __kmpc_impl_get_wtick();
  PRINT(LD_IO, "omp_get_wtick() returns %g\n", rc);
  return rc;
}

EXTERN double omp_get_wtime(void) {
  double rc = __kmpc_impl_get_wtime();
  PRINT(LD_IO, "call omp_get_wtime() returns %g\n", rc);
  return rc;
}

EXTERN int omp_get_thread_limit(void) {
  if (isSPMDMode())
    return GetNumberOfThreadsInBlock();
  int rc = GetNumberOfWorkersInTeam();
  PRINT(LD_IO, "call omp_get_thread_limit() return %d\n", rc);
  return rc;
}

EXTERN int omp_get_num_procs(void) {
  int rc = GetNumberOfProcsInDevice(isSPMDMode());
  PRINT(LD_IO, "call omp_get_num_procs() returns %d\n", rc);
  return rc;
}

EXTERN int omp_in_final(void) {
  // treat all tasks as final... Specs may expect runtime to keep
  // track more precisely if a task was actively set by users... This
  // is not explicitly specified; will treat as if runtime can
  // actively decide to put a non-final task into a final one.
  int rc = 1;
  PRINT(LD_IO, "call omp_in_final() returns %d\n", rc);
  return rc;
}

EXTERN void omp_set_nested(int flag) {
  PRINT(LD_IO, "call omp_set_nested(%d) is ignored (no nested support)\n",
        flag);
}

EXTERN int omp_get_nested(void) {
  int rc = 0;
  PRINT(LD_IO, "call omp_get_nested() returns %d\n", rc);
  return rc;
}

EXTERN void omp_set_max_active_levels(int level) {
  PRINT(LD_IO,
        "call omp_set_max_active_levels(%d) is ignored (no nested support)\n",
        level);
}

EXTERN int omp_get_max_active_levels(void) {
  int rc = 1;
  PRINT(LD_IO, "call omp_get_max_active_levels() returns %d\n", rc);
  return rc;
}

EXTERN omp_proc_bind_t omp_get_proc_bind(void) {
  PRINT0(LD_IO, "call omp_get_proc_bin() is true, regardless on state\n");
  return omp_proc_bind_true;
}

EXTERN int omp_get_num_places(void) {
  PRINT0(LD_IO, "call omp_get_num_places() returns 0\n");
  return 0;
}

EXTERN int omp_get_place_num_procs(int place_num) {
  PRINT0(LD_IO, "call omp_get_place_num_procs() returns 0\n");
  return 0;
}

EXTERN void omp_get_place_proc_ids(int place_num, int *ids) {
  PRINT0(LD_IO, "call to omp_get_place_proc_ids()\n");
}

EXTERN int omp_get_place_num(void) {
  PRINT0(LD_IO, "call to omp_get_place_num() returns 0\n");
  return 0;
}

EXTERN int omp_get_partition_num_places(void) {
  PRINT0(LD_IO, "call to omp_get_partition_num_places() returns 0\n");
  return 0;
}

EXTERN void omp_get_partition_place_nums(int *place_nums) {
  PRINT0(LD_IO, "call to omp_get_partition_place_nums()\n");
}

EXTERN int omp_get_cancellation(void) {
  int rc = 0;
  PRINT(LD_IO, "call omp_get_cancellation() returns %d\n", rc);
  return rc;
}

EXTERN void omp_set_default_device(int deviceId) {
  PRINT0(LD_IO, "call omp_get_default_device() is undef on device\n");
}

EXTERN int omp_get_default_device(void) {
  PRINT0(LD_IO,
         "call omp_get_default_device() is undef on device, returns 0\n");
  return 0;
}

EXTERN int omp_get_num_devices(void) {
  PRINT0(LD_IO, "call omp_get_num_devices() is undef on device, returns 0\n");
  return 0;
}

EXTERN int omp_get_num_teams(void) {
  int rc = GetNumberOfOmpTeams();
  PRINT(LD_IO, "call omp_get_num_teams() returns %d\n", rc);
  return rc;
}

EXTERN int omp_get_team_num() {
  int rc = GetOmpTeamId();
  PRINT(LD_IO, "call omp_get_team_num() returns %d\n", rc);
  return rc;
}

// Unspecified on the device.
EXTERN int omp_get_initial_device(void) {
  PRINT0(LD_IO, "call omp_get_initial_device() returns 0\n");
  return 0;
}

// Unused for now.
EXTERN int omp_get_max_task_priority(void) {
  PRINT0(LD_IO, "call omp_get_max_task_priority() returns 0\n");
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// locks
////////////////////////////////////////////////////////////////////////////////

EXTERN void omp_init_lock(omp_lock_t *lock) {
  __kmpc_impl_init_lock(lock);
  PRINT0(LD_IO, "call omp_init_lock()\n");
}

EXTERN void omp_destroy_lock(omp_lock_t *lock) {
  __kmpc_impl_destroy_lock(lock);
  PRINT0(LD_IO, "call omp_destroy_lock()\n");
}

EXTERN void omp_set_lock(omp_lock_t *lock) {
  __kmpc_impl_set_lock(lock);
  PRINT0(LD_IO, "call omp_set_lock()\n");
}

EXTERN void omp_unset_lock(omp_lock_t *lock) {
  __kmpc_impl_unset_lock(lock);
  PRINT0(LD_IO, "call omp_unset_lock()\n");
}

EXTERN int omp_test_lock(omp_lock_t *lock) {
  int rc = __kmpc_impl_test_lock(lock);
  PRINT(LD_IO, "call omp_test_lock() return %d\n", rc);
  return rc;
}

#pragma omp end declare target
