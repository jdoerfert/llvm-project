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

// Called at the time of the kernel initialization. This is used to initilize
// the list of references to shared variables and to pre-allocate global storage
// for holding the globalized variables.
//
// By default the globalized variables are stored in global memory. If the
// UseSharedMemory is set to true, the runtime will attempt to use shared memory
// as long as the size requested fits the pre-allocated size.
EXTERN void *__kmpc_alloc_shared(size_t DataSize) {
  return (void *)SafeMalloc(DataSize, "Alloc Shared");
}

EXTERN void __kmpc_free_shared(void *FrameStart) {
  SafeFree(FrameStart, "Free Shared");
}

// Begin a data sharing context. Maintain a list of references to shared
// variables. This list of references to shared variables will be passed
// to one or more threads.
// In L0 data sharing this is called by master thread.
// In L1 data sharing this is called by active warp master thread.
EXTERN void __kmpc_begin_sharing_variables(void ***GlobalArgs, size_t nArgs) {
  omptarget_nvptx_globalArgs.EnsureSize(nArgs);
  *GlobalArgs = omptarget_nvptx_globalArgs.GetArgs();
}

// End a data sharing context. There is no need to have a list of refs
// to shared variables because the context in which those variables were
// shared has now ended. This should clean-up the list of references only
// without affecting the actual global storage of the variables.
// In L0 data sharing this is called by master thread.
// In L1 data sharing this is called by active warp master thread.
EXTERN void __kmpc_end_sharing_variables() {
  omptarget_nvptx_globalArgs.DeInit();
}

// This function will return a list of references to global variables. This
// is how the workers will get a reference to the globalized variable. The
// members of this list will be passed to the outlined parallel function
// preserving the order.
// Called by all workers.
EXTERN void __kmpc_get_shared_variables(void ***GlobalArgs) {
  *GlobalArgs = omptarget_nvptx_globalArgs.GetArgs();
}

// This function is used to init static memory manager. This manager is used to
// manage statically allocated global memory. This memory is allocated by the
// compiler and used to correctly implement globalization of the variables in
// target, teams and distribute regions.
EXTERN void __kmpc_get_team_static_memory(int16_t isSPMDExecutionMode,
                                          const void *buf, size_t size,
                                          int16_t is_shared,
                                          const void **frame) {
  if (is_shared) {
    *frame = buf;
    return;
  }
  if (isSPMDExecutionMode) {
    if (GetThreadIdInBlock() == 0) {
      *frame = omptarget_nvptx_simpleMemoryManager.Acquire(buf, size);
    }
    __kmpc_impl_syncthreads();
    return;
  }
  ASSERT0(LT_FUSSY, GetThreadIdInBlock() == GetMasterThreadID(),
          "Must be called only in the target master thread.");
  *frame = omptarget_nvptx_simpleMemoryManager.Acquire(buf, size);
  __kmpc_impl_threadfence();
}

EXTERN void __kmpc_restore_team_static_memory(int16_t isSPMDExecutionMode,
                                              int16_t is_shared) {
  if (is_shared)
    return;
  if (isSPMDExecutionMode) {
    __kmpc_impl_syncthreads();
    if (GetThreadIdInBlock() == 0) {
      omptarget_nvptx_simpleMemoryManager.Release();
    }
    return;
  }
  __kmpc_impl_threadfence();
  ASSERT0(LT_FUSSY, GetThreadIdInBlock() == GetMasterThreadID(),
          "Must be called only in the target master thread.");
  omptarget_nvptx_simpleMemoryManager.Release();
}

// Deprecated globalization code
EXTERN void __kmpc_data_sharing_init_stack() {}
EXTERN void __kmpc_data_sharing_init_stack_spmd() {}

EXTERN void *__kmpc_data_sharing_coalesced_push_stack(size_t DataSize,
                                                      int16_t) {
  return (void *)SafeMalloc(DataSize, "Alloc Deprecated");
}

EXTERN void *__kmpc_data_sharing_push_stack(size_t DataSize, int16_t) {
  return (void *)SafeMalloc(DataSize, "Alloc Deprecated");
}

EXTERN void __kmpc_data_sharing_pop_stack(void *FrameStart) {
  SafeFree(FrameStart, "Free Shared");
}

#pragma omp end declare target
