//===----- data_sharing.cu - OpenMP GPU data sharing ------------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of data sharing environments
//
//===----------------------------------------------------------------------===//
#pragma omp declare target

#include "common/omptarget.h"
#include "target_impl.h"

// Return true if this is the master thread.
INLINE static bool IsMasterThread(bool isSPMDExecutionMode) {
  return !isSPMDExecutionMode && GetMasterThreadID() == GetThreadIdInBlock();
}

////////////////////////////////////////////////////////////////////////////////
// Runtime functions for trunk data sharing scheme.
////////////////////////////////////////////////////////////////////////////////
static const uint32_t Adjustment = 8;
void _Scratchpad::init() {
  int threadId = GetThreadIdInBlock();
  uint32_t *warpCounter = warpStorageTracker();
  *warpCounter = Adjustment;
}


void *_Scratchpad::alloc(size_t BytesPerLane) {
  BytesPerLane = (BytesPerLane + (Alignment - 1)) / Alignment * Alignment;
  void *Ptr;

  __kmpc_impl_lanemask_t Active = __kmpc_impl_activemask();
  uint32_t Leader = __kmpc_impl_ffs(Active) - 1;
  uint32_t NumActive = __kmpc_impl_popc(Active);
  uint32_t BytesTotal = BytesPerLane * NumActive;

  __kmpc_impl_lanemask_t LaneMaskLT = __kmpc_impl_lanemask_lt();
  unsigned int Position = __kmpc_impl_popc(Active & LaneMaskLT);
  if (Position == 0) {
    uint32_t *WarpCounter = warpStorageTracker();
      /*if (*WarpCounter % Alignment != 0)*/
        /*printf("r1");*/
    uint32_t BytesTotalAdjusted = BytesTotal + Adjustment;
    uint32_t BytesTotalAdjustedAligned = (BytesTotalAdjusted + (Alignment - 1)) / Alignment * Alignment;
    uint32_t OldUsed = *WarpCounter;
    if (OldUsed + BytesTotalAdjustedAligned > warpStorage()) {
      Ptr = (void *)SafeMalloc(BytesTotal, "Alloc Shared");
      /*Ptr = 0;*/
    } else {
      Ptr = warpData();
      *WarpCounter += BytesTotalAdjustedAligned;
      /*if (*WarpCounter % Alignment != 0)*/
        /*printf("r2");*/
      *((uint64_t*)Ptr) = BytesTotalAdjustedAligned;
      Ptr = (char*)Ptr + Adjustment;
    }
  }

  // Get address from lane Leader.
  int *FP = (int *)&Ptr;
  FP[0] = __kmpc_impl_shfl_sync(Active, FP[0], Leader);
  if (sizeof(Ptr) == 8)
    FP[1] = __kmpc_impl_shfl_sync(Active, FP[1], Leader);

  return (char*)Ptr + (BytesPerLane * Position);
}
void _Scratchpad::free(void *Ptr) {
  __kmpc_impl_lanemask_t Active = __kmpc_impl_activemask();
  __kmpc_impl_lanemask_t LaneMaskLT = __kmpc_impl_lanemask_lt();
  unsigned int Position = __kmpc_impl_popc(Active & LaneMaskLT);
  if (Position)
    return;
  if (Ptr < &Data[0] || Ptr > &Data[SHARED_STORAGE]) {
    SafeFree(Ptr, "Free Shared");
  } else {
    Ptr = (char*)Ptr - Adjustment;
    uint64_t BytesTotalAdjustedAligned = *((uint64_t*)Ptr);
    uint32_t *WarpCounter = warpStorageTracker();
      /*if (*WarpCounter % Alignment != 0)*/
        /*printf("r3");*/
    *WarpCounter -= BytesTotalAdjustedAligned;
  }
}

DEVICE void** SHARED(GlobalArgsPtr);
DEVICE _Scratchpad SHARED(scratchpad);

// Called at the time of the kernel initialization. This is used to initilize
// the list of references to shared variables and to pre-allocate global storage
// for holding the globalized variables.
//
// By default the globalized variables are stored in global memory. If the
// UseSharedMemory is set to true, the runtime will attempt to use shared memory
// as long as the size requested fits the pre-allocated size.
EXTERN void *__kmpc_alloc_shared(size_t DataSize) {
  return scratchpad.alloc(DataSize);
}

EXTERN void __kmpc_free_shared(void *FrameStart) {
  scratchpad.free(FrameStart);
}

// Begin a data sharing context. Maintain a list of references to shared
// variables. This list of references to shared variables will be passed
// to one or more threads.
// In L0 data sharing this is called by master thread.
// In L1 data sharing this is called by active warp master thread.
EXTERN void __kmpc_begin_sharing_variables(void ***GlobalArgs, size_t nArgs) {
  *GlobalArgs = GlobalArgsPtr = static_cast<decltype(GlobalArgsPtr)>(__kmpc_alloc_shared(nArgs * sizeof(GlobalArgsPtr[0])));
}

// End a data sharing context. There is no need to have a list of refs
// to shared variables because the context in which those variables were
// shared has now ended. This should clean-up the list of references only
// without affecting the actual global storage of the variables.
// In L0 data sharing this is called by master thread.
// In L1 data sharing this is called by active warp master thread.
EXTERN void __kmpc_end_sharing_variables() {
  __kmpc_free_shared(GlobalArgsPtr);
}

// This function will return a list of references to global variables. This
// is how the workers will get a reference to the globalized variable. The
// members of this list will be passed to the outlined parallel function
// preserving the order.
// Called by all workers.
EXTERN void __kmpc_get_shared_variables(void ***GlobalArgs) {
  *GlobalArgs = GlobalArgsPtr;
}

#pragma omp end declare target
