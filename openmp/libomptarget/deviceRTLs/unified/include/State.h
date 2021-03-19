//===-------- State.h - OpenMP State & ICV interface ------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_STATE_H
#define OMPTARGET_STATE_H

#include "Types.h"

namespace _OMP {

namespace state {

#pragma omp declare target

// TODO: Expose this via CMAKE.
inline constexpr uint32_t SharedScratchpadSize = 2048;

/// Initialize the state machinery. Must be called by all threads.
void init(bool IsSPMD);

/// TODO
enum ValueKind {
  VK_NThreads,
  VK_Level,
  VK_ActiveLevel,
  VK_MaxActiveLevels,
  VK_RunSched,
  // ---
  VK_RunSchedChunk,
  VK_ParallelRegionFn,
  VK_ParallelTeamSize,
};

/// TODO
void enterDataEnvironment();

/// TODO
void exitDataEnvironment();

/// TODO
struct DateEnvironmentRAII {
  DateEnvironmentRAII() { enterDataEnvironment(); }
  ~DateEnvironmentRAII() { exitDataEnvironment(); }
};

/// TODO
void resetStateForThread();

uint32_t &lookup32(ValueKind VK, bool IsReadonly);
void* &lookupPtr(ValueKind VK, bool IsReadonly);

/// A mookup class without actual state used to provide
/// a nice interface to lookup and update ICV values
/// we can declare in global scope.
template <typename Ty, ValueKind Kind> struct Value {
  operator Ty() { return lookup(/* IsReadonly */ true); }

  Value &operator=(const Ty &Other) {
    set(Other);
    return *this;
  }

  Value &operator++() {
    inc(1);
    return *this;
  }

  Value &operator--() {
    inc(-1);
    return *this;
  }

private:

  Ty &lookup(bool IsReadonly) {
    return lookup32(Kind, IsReadonly);
  }

  Ty &inc(int UpdateVal) {
    return (lookup(/* IsReadonly */ false) += UpdateVal);
  }

  Ty &set(Ty UpdateVal) {
    return (lookup(/* IsReadonly */ false) = UpdateVal);
  }
};
/// A mookup class without actual state used to provide
/// a nice interface to lookup and update ICV values
/// we can declare in global scope.
template <typename Ty, ValueKind Kind> struct PtrValue {
  operator Ty() { return lookup(/* IsReadonly */ true); }

  PtrValue &operator=(const Ty Other) {
    set(Other);
    return *this;
  }

private:

  Ty &lookup(bool IsReadonly) {
    return lookupPtr(Kind, IsReadonly);
  }

  Ty &set(Ty UpdateVal) {
    return (lookup(/* IsReadonly */ false) = UpdateVal);
  }
};

/// TODO
inline state::Value<uint32_t, state::VK_RunSchedChunk> RunSchedChunk;

/// TODO
inline state::Value<uint32_t, state::VK_ParallelTeamSize> ParallelTeamSize;

/// TODO
inline state::PtrValue<ParallelRegionFnTy, state::VK_ParallelRegionFn>
    ParallelRegionFn;
#pragma omp end declare target

} // namespace state

namespace icv {

#pragma omp declare target

/// TODO
inline state::Value<uint32_t, state::VK_NThreads> NThreads;

/// TODO
inline state::Value<uint32_t, state::VK_Level> Level;

/// The `active-level` describes which of the parallel level counted with the
/// `level-var` is active. There can only be one.
///
/// active-level-var is 1, if ActiveLevelVar is not 0, otherweise it is 0.
inline state::Value<uint32_t, state::VK_ActiveLevel> ActiveLevel;

/// TODO
inline state::Value<uint32_t, state::VK_MaxActiveLevels> MaxActiveLevels;

/// TODO
inline state::Value<uint32_t, state::VK_RunSched> RunSched;

#pragma omp end declare target

} // namespace icv

namespace memory {

/// Alloca \p Size bytes in shared memory, if possible, for \p Reason.
///
/// Note: See the restrictions on __kmpc_alloc_shared for proper usage.
void *allocShared(uint64_t Size, const char *Reason);

/// Free \p Ptr, alloated via allocShared, for \p Reason.
///
/// Note: See the restrictions on __kmpc_free_shared for proper usage.
void freeShared(void *Ptr, const char *Reason);

/// Alloca \p Size bytes in global memory, if possible, for \p Reason.
void *allocGlobal(uint64_t Size, const char *Reason);

/// Free \p Ptr, alloated via allocGlobal, for \p Reason.
void freeGlobal(void *Ptr, const char *Reason);

} // namespace memory

} // namespace _OMP

/// External API
///
///{

extern "C" {

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

/// ICV: max-active-level-var, constant 1
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

/// ICV: active-level-var, 0 or 1
///
/// getter: returns 0 or 1.
///
///{
int omp_get_active_level(void);
///}

/// ICV: level-var
///
/// getter: returns parallel region nesting
///
///{
int omp_get_level(void);
///}

/// ICV: run-sched-var
///
///
///{
void omp_set_schedule(omp_sched_t, int);
void omp_get_schedule(omp_sched_t *, int *);
///}

/// TODO this is incomplete.
int omp_get_num_threads(void);
int omp_get_thread_num(void);
void omp_set_nested(int);

int omp_get_nested(void);

void omp_set_max_active_levels(int Level);

int omp_get_max_active_levels(void);

omp_proc_bind_t omp_get_proc_bind(void);

int omp_get_num_places(void);

int omp_get_place_num_procs(int place_num);

void omp_get_place_proc_ids(int place_num, int *ids);

int omp_get_place_num(void);

int omp_get_partition_num_places(void);

void omp_get_partition_place_nums(int *place_nums);

int omp_get_cancellation(void);

void omp_set_default_device(int deviceId);

int omp_get_default_device(void);

int omp_get_num_devices(void);

int omp_get_num_teams(void);

int omp_get_team_num();

int omp_get_initial_device(void);

/// Allocate \p Bytes in "shareable" memory and return the address. Needs to be
/// called balanced with __kmpc_free_shared like a stack (push/pop). Can be
/// called by any thread, allocation happens *per thread*.
void *__kmpc_alloc_shared(uint64_t Bytes);

/// Deallocate \p Ptr. Needs to be called balanced with __kmpc_alloc_shared like
/// a stack (push/pop). Can be called by any thread. \p Ptr has to be the
/// allocated by __kmpc_alloc_shared by the same thread.
void __kmpc_free_shared(void *Ptr);

/// Allocate sufficient space for \p NumArgs sequential `void*` and store the
/// allocation address in \p GlobalArgs.
///
/// Called by the main thread prior to a parallel region.
///
/// We also remember it in GlobalArgsPtr to ensure the worker threads and
/// deallocation function know the allocation address too.
void __kmpc_begin_sharing_variables(void ***GlobalArgs, uint64_t NumArgs);

/// Deallocate the memory allocated by __kmpc_begin_sharing_variables.
///
/// Called by the main thread after a parallel region.
///
/// TODO: This should really take the address, or number of bytes at least.
void __kmpc_end_sharing_variables();

/// Store the allocation address obtained via __kmpc_begin_sharing_variables in
/// \p GlobalArgs.
///
/// Called by the worker threads in the parallel region (function).
void __kmpc_get_shared_variables(void ***GlobalArgs);
}

/// Macros for allocating variables in different address spaces.
///{

// Follows the pattern in interface.h
typedef enum omp_allocator_handle_t {
  omp_null_allocator = 0,
  omp_default_mem_alloc = 1,
  omp_large_cap_mem_alloc = 2,
  omp_const_mem_alloc = 3,
  omp_high_bw_mem_alloc = 4,
  omp_low_lat_mem_alloc = 5,
  omp_cgroup_mem_alloc = 6,
  omp_pteam_mem_alloc = 7,
  omp_thread_mem_alloc = 8,
  KMP_ALLOCATOR_MAX_HANDLE = ~(0U)
} omp_allocator_handle_t;

#define __PRAGMA(STR) _Pragma(#STR)
#define OMP_PRAGMA(STR) __PRAGMA(omp STR)

#define SHARED(NAME)                                                           \
  NAME [[clang::loader_uninitialized]];                                        \
  OMP_PRAGMA(allocate(NAME) allocator(omp_pteam_mem_alloc))

// TODO: clang should use address space 5 for omp_thread_mem_alloc, but right
//       now that's not the case.
#define THREAD_LOCAL(NAME)                                                     \
  NAME [[clang::loader_uninitialized, clang::address_space(5)]]

// TODO: clang should use address space 4 for omp_const_mem_alloc, maybe it
//       does?
#define CONSTANT(NAME)                                                         \
  NAME [[clang::loader_uninitialized, clang::address_space(4)]]

///}

#endif
