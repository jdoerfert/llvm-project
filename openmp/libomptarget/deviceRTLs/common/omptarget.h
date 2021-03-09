//===---- omptarget.h - OpenMP GPU initialization ---------------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of all library macros, types,
// and functions.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_H
#define OMPTARGET_H

#include "common/allocator.h"
#include "common/debug.h" // debug
#include "common/state-queue.h"
#include "common/support.h"
#include "interface.h" // interfaces with omp, compiler, and user
#include "target_impl.h"

#include "Mapping.h"

#define OMPTARGET_NVPTX_VERSION 1.1

// used by the library for the interface with the app
#define DISPATCH_FINISHED 0
#define DISPATCH_NOTFINISHED 1

// used by dynamic scheduling
#define FINISHED 0
#define NOT_FINISHED 1
#define LAST_CHUNK 2

#define BARRIER_COUNTER 0
#define ORDERED_COUNTER 1

////////////////////////////////////////////////////////////////////////////////
// thread private data (struct of arrays for better coalescing)
// tid refers here to the global thread id
// do not support multiple concurrent kernel a this time

struct _Scratchpad {
  void init();

  void* alloc(size_t Bytes);
  void free(void *Ptr);


  // Add worst-case padding to DataSize so that future stack allocations are
  // correctly aligned.
  static constexpr size_t Alignment = 8;
  static constexpr uint32_t SHARED_STORAGE = 2048;
  static constexpr uint32_t MAIN_THREAD_USAGE = (SHARED_STORAGE / 4);

  char Data[SHARED_STORAGE] ALIGN(8);

  uint32_t warpStorage() {
    if (omp::isMainThreadInGenericMode())
      return MAIN_THREAD_USAGE;
    uint32_t NumWarps = GetNumberOfThreadsInBlock() / WARPSIZE;
    uint32_t WarpStorage = (SHARED_STORAGE - MAIN_THREAD_USAGE) / NumWarps;
    uint32_t WarpStorageAligned =  (WarpStorage / Alignment) * Alignment;
    return WarpStorageAligned;
  }

  char *warpBegin() {
    if (omp::isMainThreadInGenericMode())
      return &Data[0];
    return &Data[MAIN_THREAD_USAGE + warpStorage() * GetWarpId()];
  }
  char *warpData() {
    uint32_t *warpCounter = warpStorageTracker();
    return warpBegin() + (*warpCounter);
  }
  uint32_t *warpStorageTracker() {
    return ((uint32_t*)warpBegin());
  }
};
extern _Scratchpad EXTERN_SHARED(scratchpad);

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// global data tables
////////////////////////////////////////////////////////////////////////////////

extern uint32_t EXTERN_SHARED(execution_param);
extern void *EXTERN_SHARED(ReductionScratchpadPtr);

////////////////////////////////////////////////////////////////////////////////
// work function (outlined parallel/simd functions) and arguments.
// needed for L1 parallelism only.
////////////////////////////////////////////////////////////////////////////////

typedef void *omptarget_nvptx_WorkFn;
extern volatile omptarget_nvptx_WorkFn EXTERN_SHARED(omptarget_nvptx_workFn);

////////////////////////////////////////////////////////////////////////////////
// inlined implementation
////////////////////////////////////////////////////////////////////////////////

INLINE uint32_t __kmpc_impl_ffs(uint32_t x) { return __builtin_ffs(x); }
INLINE uint32_t __kmpc_impl_popc(uint32_t x) { return __builtin_popcount(x); }
INLINE uint32_t __kmpc_impl_ffs(uint64_t x) { return __builtin_ffsl(x); }
INLINE uint32_t __kmpc_impl_popc(uint64_t x) { return __builtin_popcountl(x); }

#include "common/omptargeti.h"

#endif
