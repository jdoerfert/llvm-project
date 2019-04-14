//===-- target_region.cu ---- CUDA impl. of the target region interface -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the common target region interface.
//
//===----------------------------------------------------------------------===//

// Include the native definitions first as certain defines might be needed in
// the common interface definition below.
#include "omptarget-nvptx.h"
#include "interface.h"

#include "../../common/target_region.h"

#include "cuda_fp16.h"

EXTERN void *__kmpc_target_region_kernel_get_shared_memory() {
  return _shared_bytes_buffer_memory.begin();
}
EXTERN void *__kmpc_target_region_kernel_get_private_memory() {
  return ((char *)_shared_bytes_buffer_memory.begin()) +
         _shared_bytes_buffer_memory.get_offset();
}

/// Simple generic state machine for worker threads.
INLINE static void
__kmpc_target_region_state_machine(ident_t *Ident,
                                   bool IsOMPRuntimeInitialized) {

  do {
    void *WorkFn = 0;

    // Wait for the signal that we have a new work function.
    __kmpc_barrier_simple_spmd(Ident, 0);

    // Retrieve the work function from the runtime.
    bool IsActive = __kmpc_kernel_parallel(&WorkFn, IsOMPRuntimeInitialized);

    // If there is nothing more to do, break out of the state machine by
    // returning to the caller.
    if (!WorkFn)
      return;

    if (IsActive) {
      void *SharedVars = __kmpc_target_region_kernel_get_shared_memory();
      void *PrivateVars = __kmpc_target_region_kernel_get_private_memory();

      ((ParallelWorkFnTy)WorkFn)(SharedVars, PrivateVars);

      __kmpc_kernel_end_parallel();
    }

    __kmpc_barrier_simple_spmd(Ident, 0);

  } while (true);
}

/// Filter threads into masters and workers. If \p UseStateMachine is true,
/// required workers will enter a state machine through and be trapped there.
/// Master and surplus worker threads will return from this function immediately
/// while required workers will only return once there is no more work. The
/// return value indicates if the thread is a master (1), a surplus worker (0),
/// or a finished required worker released from the state machine (-1).
INLINE static int8_t
__kmpc_target_region_thread_filter(ident_t *Ident, unsigned ThreadLimit,
                                   bool UseStateMachine,
                                   bool IsOMPRuntimeInitialized) {

  unsigned TId = GetThreadIdInBlock();
  bool IsWorker = TId < ThreadLimit;

  if (IsWorker) {
    if (UseStateMachine)
      __kmpc_target_region_state_machine(Ident, IsOMPRuntimeInitialized);
    return -1;
  }

  return TId == GetMasterThreadID();
}

EXTERN int8_t __kmpc_target_region_kernel_init(ident_t *Ident, bool UseSPMDMode,
                                               bool RequiresOMPRuntime,
                                               bool UseStateMachine,
                                               bool RequiresDataSharing) {
  unsigned NumThreads = GetNumberOfThreadsInBlock();

  // Handle the SPMD case first.
  if (UseSPMDMode) {

    __kmpc_spmd_kernel_init(NumThreads, RequiresOMPRuntime,
                            RequiresDataSharing);

    if (RequiresDataSharing)
      __kmpc_data_sharing_init_stack_spmd();

    return 1;
  }

  // Reserve one WARP in non-SPMD mode for the masters.
  unsigned ThreadLimit = NumThreads - WARPSIZE;
  int8_t FilterVal = __kmpc_target_region_thread_filter(
      Ident, ThreadLimit, UseStateMachine, RequiresOMPRuntime);

  // If the filter returns 1 the executing thread is a team master which will
  // initialize the kernel in the following.
  if (FilterVal == 1) {
    __kmpc_kernel_init(ThreadLimit, RequiresOMPRuntime);
    __kmpc_data_sharing_init_stack();
    _shared_bytes_buffer_memory.init();
  }

  return FilterVal;
}

EXTERN void __kmpc_target_region_kernel_deinit(ident_t *Ident, bool UseSPMDMode,
                                               bool RequiredOMPRuntime) {
  // Handle the SPMD case first.
  if (UseSPMDMode) {
    __kmpc_spmd_kernel_deinit_v2(RequiredOMPRuntime);
    return;
  }

  __kmpc_kernel_deinit(RequiredOMPRuntime);

  // Barrier to terminate worker threads.
  __kmpc_barrier_simple_spmd(Ident, 0);

  // Release any dynamically allocated memory used for sharing.
  _shared_bytes_buffer_memory.release();
}

EXTERN void __kmpc_target_region_kernel_parallel(
    ident_t *Ident, int16_t UseSPMDMode, bool RequiredOMPRuntime,
    ParallelWorkFnTy ParallelWorkFn, void *SharedVars, uint16_t SharedVarsBytes,
    void *PrivateVars, uint16_t PrivateVarsBytes, bool SharedMemPointers) {

  // If the mode is unknown we check it at runtime
  if (UseSPMDMode == -1)
    UseSPMDMode = __kmpc_is_spmd_exec_mode();

  // In SPMD mode, we simply call the work function with the provided values.
  if (UseSPMDMode) {
    ParallelWorkFn(SharedVars, PrivateVars);
    return;
  }

  if (SharedMemPointers) {
    // If shared memory pointers are used, the user guarantees that all private
    // variables, if any, are stored directly after the shared ones in memory.
    // Additionally, this memory can be accessed by all the threads. In that
    // case, we do not need to copy memory around but simply use the provided
    // locations. However, we still need to inform the buffer of these
    // locations as the worker threads might use the
    //   __kmpc_target_region_kernel_get_shared_memory()
    // and
    //   __kmpc_target_region_kernel_get_private_memory()
    // functions to get the respective pointers.

    _shared_bytes_buffer_memory.set(SharedVars, SharedVarsBytes);

  } else {

    size_t BytesToCopy = SharedVarsBytes + PrivateVarsBytes;
    if (BytesToCopy) {
      // Resize the shared memory to be able to hold the data which is required
      // to be in shared memory. Also set the offset to the beginning to the
      // private variables.
      _shared_bytes_buffer_memory.resize(BytesToCopy, SharedVarsBytes);

      // Copy the shared and private variables into shared memory.
      void *SVMemory = __kmpc_target_region_kernel_get_shared_memory();
      void *PVMemory = __kmpc_target_region_kernel_get_private_memory();
      memcpy(SVMemory, SharedVars, SharedVarsBytes);
      memcpy(PVMemory, PrivateVars, PrivateVarsBytes);
    }
  }

  // TODO: It seems we could store the work function in the same shared space
  // as the rest of the variables above.
  //
  // Initialize the parallel work, e.g., make sure the work function is known.
  __kmpc_kernel_prepare_parallel((void *)ParallelWorkFn, RequiredOMPRuntime);

  // TODO: It is odd that we call the *_spmd version in non-SPMD mode here.
  //
  // Activate workers. This barrier is used by the master to signal
  // work for the workers.
  __kmpc_barrier_simple_spmd(Ident, 0);

  // OpenMP [2.5, Parallel Construct, p.49]
  // There is an implied barrier at the end of a parallel region. After the
  // end of a parallel region, only the master thread of the team resumes
  // execution of the enclosing task region.
  //
  // The master waits at this barrier until all workers are done.
  __kmpc_barrier_simple_spmd(Ident, 0);

  // Update the shared variables if necessary, that is if we did not use user
  // memory in the first .
  if (!SharedMemPointers && SharedVarsBytes)
    memcpy(SharedVars, __kmpc_target_region_kernel_get_shared_memory(),
           SharedVarsBytes);

  // We could set (or reset) the _shared_bytes_buffer_memory pointer to NULL (or
  // the old value) if we used user provided memory. This is not necessary as
  // long as the buffer knows not to free the explicitly "set" pointer.
}

// Introduce the cuda spellings for types:
#define BOOL_TY bool
#define CHAR_TY char
#define SHORT_TY short
#define INT_TY int
#define LONG_TY long
#define LONG_LONG_TY long long
// volatile in combination with __half is broken.
#define HALF_FLOAT_TY float
#define FLOAT_TY float
#define DOUBLE_FLOAT_TY double

#define NOP(l, r) (r)
#define ADD(l, r) ((l) + (r))
#define MUL(l, r) ((l) * (r))
#define MIN(l, r) ((l) < (r) ? (l) : (r))
#define MAX(l, r) ((l) < (r) ? (r) : (l))
#define XOR(l, r) (((long long)(l)) ^ (((long long)r)))
#define BOR(l, r) (((long long)(l)) | (((long long)r)))
#define BAND(l, r) (((long long)(l)) & (((long long)r)))

typedef void (*ReduceFnTy)(void *, void *, enum ReductionOperator);

template <typename data_t>
INLINE static void reduce(void *SrcPtr, void *DestPtr,
                          enum ReductionOperator RedOp) {
  switch (RedOp) {
#define RO(NAME, BIN)                                                \
  case NAME:                                                                   \
    *((data_t *)DestPtr) = BIN(*((data_t *)DestPtr), *((volatile gata_t *)SrcPtr));     \
    break;

    REDUCTION_OPERATORS()
#undef RO
  };
}

template<typename data_t>
INLINE static void shuffleAndStore(void *SrcPtr, void *DestPtr, int16_t DestOffset) {
  size_t LeftoverSize = sizeof(data_t);
  for (size_t ShuffleSize = 8; ShuffleSize >= 1; ShuffleSize /= 2) {
    while (ShuffleSize <= LeftoverSize) {
      if (ShuffleSize <= 4) {
        int32_t *SrcPtr32 = (int32_t *)SrcPtr;
        int32_t *DestPtr32 = (int32_t *)DestPtr;
        int32_t Res =
            __kmpc_shuffle_int32(*SrcPtr32, (int32_t)DestOffset, WARPSIZE);
        *DestPtr32 = Res;
        SrcPtr = (SrcPtr32 + 1);
        DestPtr = (DestPtr32 + 1);
      } else {
        int64_t *SrcPtr64 = (int64_t *)SrcPtr;
        int64_t *DestPtr64 = (int64_t *)DestPtr;
        int64_t Res =
            __kmpc_shuffle_int64(*SrcPtr64, (int64_t)DestOffset, WARPSIZE);
        *DestPtr64 = Res;
        SrcPtr = (SrcPtr64 + 1);
        DestPtr = (DestPtr64 + 1);
      }
      LeftoverSize -= ShuffleSize;
    }
  }
}

template <typename data_t, enum ReductionOperator RedOp>
INLINE static void shuffleAndReduce(void *LocalItem, int16_t LaneId,
                                    int16_t Offset, int16_t AlgoVer) {
  __align__(64) data_t RemoteItem = 42;

  shuffleAndStore<data_t>(LocalItem, (void*)&RemoteItem, Offset);

  if ((AlgoVer == 0) |
      ((AlgoVer == 1) & ((uint16_t)LaneId < (uint16_t)Offset)) |
      ((AlgoVer == 2) & (LaneId & (int16_t)1) & (Offset > (int16_t)0)))
    reduce<data_t>((void*)&RemoteItem, LocalItem, RedOp);
  else
    reduce<data_t>((void*)&RemoteItem, LocalItem, RO_NOP);
}

template<typename data_t>
INLINE static void interWarpCopy(void *Ptr, int32_t WarpNum) {
  __shared__ __align__(256) data_t Buffer[WARPSIZE];
  uint32_t GlobalTId = __kmpc_global_thread_num(0);
  uint32_t TId =  GetThreadIdInBlock();
  uint32_t LaneId = TId % WARPSIZE;

  __kmpc_barrier(0, GlobalTId);

  bool IsMasterThread = (LaneId == 0);
  if (IsMasterThread) {
    uint32_t WarpId = TId / WARPSIZE;
    Buffer[WarpId] = *((data_t*)Ptr);
  }

  __kmpc_barrier(0, GlobalTId);

  bool IsActiveThread = (TId < (uint32_t)WarpNum);
  if (IsActiveThread)
    *((data_t*)Ptr) = Buffer[TId];

}

INLINE static
void globalToBufferCopy(void *buffer, int idx, void *reduce_data) {
}

INLINE static
void globalToBufferReduce(void *buffer, int idx, void *reduce_data) {
}

INLINE static
void bufferToGlobalCopy(void *buffer, int idx, void *reduce_data) {
}

INLINE static
void bufferToGlobalReduce(void *buffer, int idx, void *reduce_data) {
}

template<typename data_t>
INLINE static
void initialize(void *LocPtr, void *DataPtr) {
  *((data_t *)LocPtr) = *((data_t *)DataPtr);
}

EXTERN void *__kmpc_target_region_kernel_reduction_init(
    ident_t *Ident, int16_t UseSPMDMode, bool RequiredOMPRuntime,
    int32_t GlobalTId, bool IsParallelReduction, bool IsTeamReduction,
    void *OriginalLocation, void *PrivateLocation,
    uint32_t ReductionLocationSize, void *RHSPtr,
    enum ReductionBaseType BaseType) {

  switch (BaseType) {
#define RBT(NAME, TYPE)                                                        \
  case NAME:                                                                   \
    initialize<TYPE>(PrivateLocation, RHSPtr);                                 \
    break;

    REDUCTION_BASE_TYPES()
#undef RBT
  }

  return PrivateLocation;
}

template <typename data_t>
INLINE static kmp_ShuffleReductFctPtr
getShuffleAndReduceFn(enum ReductionOperator RedOp) {

  switch (RedOp) {
#define RO(NAME, BIN)                                                \
  case NAME:                                                                   \
    return &shuffleAndReduce<data_t, NAME>;

    REDUCTION_OPERATORS()
#undef RO
  }

  // Unreachable
  return 0;
}

EXTERN void __kmpc_target_region_kernel_reduction_finalize(
    ident_t *Ident, int16_t UseSPMDMode, bool RequiredOMPRuntime,
    int32_t GlobalTId, bool IsParallelReduction, bool IsTeamReduction,
    void *OriginalLocation, void *ReductionLocation,
    uint32_t NumReductionLocations, enum ReductionOperator RedOp,
    enum ReductionBaseType BaseType) {

  // If the mode is unknown we check it at runtime
  if (UseSPMDMode == -1)
    UseSPMDMode = __kmpc_is_spmd_exec_mode();

  ReduceFnTy Reduce;
  size_t ReduceSize = 0;
  kmp_ShuffleReductFctPtr ShuffleAndReduce;
  kmp_InterWarpCopyFctPtr InterWarpCopy;
  switch (BaseType) {
#define RBT(NAME, TYPE)                                                        \
  case NAME:                                                                   \
    ShuffleAndReduce = getShuffleAndReduceFn<TYPE>(RedOp);                     \
    InterWarpCopy = &interWarpCopy<TYPE>;                                      \
    ReduceSize = sizeof(TYPE);                                                 \
    Reduce = &reduce<TYPE>;                                                    \
    break;

    REDUCTION_BASE_TYPES()
#undef RBT
  }

  int32_t res = 0;
  if (IsParallelReduction) {
    res = __kmpc_nvptx_parallel_reduce_nowait_v3(
        Ident, GlobalTId, /* NumVars */ 1, ReduceSize, ReductionLocation,
        ShuffleAndReduce, InterWarpCopy, UseSPMDMode, RequiredOMPRuntime);
  } else if (IsTeamReduction) {
//    res = __kmpc_nvptx_teams_reduce_nowait_v3(
//        Ident, GlobalTId, /* NumVars */ 1, NumReductionLocations,
//        ReductionLocation, ShuffleAndReduce, InterWarpCopy,
//        &globalToBufferCopy, &globalToBufferReduce, &bufferToGlobalCopy,
//        &bufferToGlobalReduce, UseSPMDMode, RequiredOMPRuntime);
  } else {
    // This should probably cause an abort, e.g., assert, as this should be
    // either parallel or team reduction.
  }

  if (res == 1) {
    __kmpc_nvptx_end_reduce_nowait(GlobalTId);
    Reduce(ReductionLocation, OriginalLocation, RedOp);
  }
};
