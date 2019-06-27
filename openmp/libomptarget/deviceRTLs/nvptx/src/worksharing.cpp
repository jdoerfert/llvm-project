//===--- worksharing.cpp --- OpenMP worksharing constructs ------+- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the OpenMP runtime calls for
// worksharing constructs that use the same interface.
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"

/// Compute the lower bound (\p LB), upper bound (\p UB), stride (\p Stride),
/// and last iteration flag (\p IsLast) for a loop with static scheduling and
/// chunk size \p ChunkSize. The \p EntityId describes the location of the
/// executing thread in the team and \p NumberOfEntities is the number of
/// threads in the team.
///
/// IVTy is type of the loop iteration variable and SIVTy is the signed version
/// of IVTy.
template <typename IVTy, typename SIVTy>
INLINE static void ForStaticChunk(int &last, IVTy &LB, IVTy &UB, SIVTy &Stride,
                                  SIVTy ChunkSize, IVTy EntityId,
                                  IVTy NumberOfEntities) {
  // Each thread executes multiple chunks all of the same size.

  // Distance between two successive chunks
  Stride = NumberOfEntities * ChunkSize;
  LB = LB + EntityId * ChunkSize;

  // Adjust the upper bound by one to match the less-than (<=) comparison
  // Clang uses, e.g., the loop condition will be i <= UB not i < UB.
  IVTy InputUb = UB;
  UB = LB + ChunkSize - 1;

  // Say UB' is the begining of the last chunk. Then who ever has a
  // lower bound plus a multiple of the increment equal to UB' is
  // the last one.
  IVTy beginingLastChunk = InputUb - (InputUb % ChunkSize);
  last = ((beginingLastChunk - LB) % Stride) == 0;
}

/// Compute the lower bound (\p LB), upper bound (\p UB), stride (\p Stride),
/// and last iteration flag (\p IsLast) for a loop with static scheduling and
/// chunk size \p ChunkSize. The \p EntityId describes the location of the
/// executing thread in the team and \p NumberOfEntities is the number of
/// threads in the team.
///
/// IVTy is type of the loop iteration variable and SIVTy is the signed version
/// of IVTy.
template <typename IVTy, typename SIVTy>
INLINE static void ForStaticNoChunk(int &IsLast, IVTy &LB, IVTy &UB,
                                    SIVTy &Stride, SIVTy &ChunkSize,
                                    IVTy EntityId, IVTy NumberOfEntities) {
  // No chunk size specified. Each thread gets at most one chunk; chunks are
  // all almost of equal size
  IVTy loopSize = UB - LB + 1;

  ChunkSize = loopSize / NumberOfEntities;
  IVTy leftOver = loopSize - ChunkSize * NumberOfEntities;

  if (EntityId < leftOver) {
    ChunkSize++;
    LB = LB + EntityId * ChunkSize;
  } else {
    LB = LB + EntityId * ChunkSize + leftOver;
  }

  // Adjust the upper bound by one to match the less-than (<=) comparison
  // Clang uses, e.g., the loop condition will be i <= UB not i < UB.
  IVTy InputUb = UB;
  UB = LB + ChunkSize - 1;

  IsLast = LB <= InputUb && InputUb <= UB;

  // Make sure we only do 1 chunk per warp.
  Stride = loopSize;
}

/// Calculate initial bounds for static loop and stride
///
/// \param[in] GlobalTID global thread id
/// \param[in] ScheduleType type of scheduling (see omptarget-nvptx.h)
/// \param[in] IsLastIterPtr pointer to last iteration
/// \param[in,out] LowerBoundPtr to loop lower bound. it will contain value of
/// lower bound of first chunk
/// \param[in,out] UpperBoundPtr to loop upper bound. It will contain value of
/// upper bound of first chunk
/// \param[in,out] StridePtr to loop stride. It will contain value of stride
/// between two successive chunks executed by the same thread
/// \param[in] ChunkSize
/// \param[in] IsSPMDExecutionMode flag to indicate SPMD-mode
///
/// IVTy is type of the loop iteration variable and SIVTy is the signed version
/// of IVTy.
template <typename IVTy, typename SIVTy>
INLINE static void for_static_init(int32_t GlobalTID, int32_t ScheduleType,
                                   int32_t *IsLastIterPtr, IVTy *LowerBoundPtr,
                                   IVTy *UpperBoundPtr, SIVTy *StridePtr,
                                   SIVTy ChunkSize, bool IsSPMDExecutionMode) {
  // When IsSPMDExecutionMode is true, we assume that the caller is
  // in an L0 parallel region and that all worker threads participate.

  // Assume we are in teams region or that we use a single block
  // per target region
  SIVTy numberOfActiveOMPThreads = GetNumberOfOmpThreads(IsSPMDExecutionMode);

  // All warps that are in excess of the maximum requested, do
  // not execute the loop
  PRINT(LD_LOOP,
        "OMP Thread %d: schedule type %d, chunk size = %lld, mytid "
        "%d, num tids %d\n",
        (int)GlobalTID, (int)ScheduleType, (long long)ChunkSize, (int)GlobalTID,
        (int)numberOfActiveOMPThreads);
  ASSERT0(LT_FUSSY, GlobalTID < numberOfActiveOMPThreads,
          "current thread is not needed here; error");

  // copy
  int IsLastIter = 0;
  IVTy LB = *LowerBoundPtr;
  IVTy UB = *UpperBoundPtr;
  SIVTy Stride = *StridePtr;
  // init
  switch (SCHEDULE_WITHOUT_MODIFIERS(ScheduleType)) {
  case kmp_sched_static_chunk: {
    if (ChunkSize > 0) {
      ForStaticChunk(IsLastIter, LB, UB, Stride, ChunkSize, GlobalTID,
                     numberOfActiveOMPThreads);
      break;
    }
  } // note: if ChunkSize <=0, use nochunk
  case kmp_sched_static_balanced_chunk: {
    if (ChunkSize > 0) {
      // round up to make sure the chunk is enough to cover all iterations
      IVTy tripCount = UB - LB + 1; // +1 because UB is inclusive
      IVTy span =
          (tripCount + numberOfActiveOMPThreads - 1) / numberOfActiveOMPThreads;
      // perform chunk adjustment
      ChunkSize = (span + ChunkSize - 1) & ~(ChunkSize - 1);

      ASSERT0(LT_FUSSY, UB >= LB, "ub must be >= lb.");
      IVTy oldUb = UB;
      ForStaticChunk(IsLastIter, LB, UB, Stride, ChunkSize, GlobalTID,
                     numberOfActiveOMPThreads);
      if (UB > oldUb)
        UB = oldUb;
      break;
    }
  } // note: if ChunkSize <=0, use nochunk
  case kmp_sched_static_nochunk: {
    ForStaticNoChunk(IsLastIter, LB, UB, Stride, ChunkSize, GlobalTID,
                     numberOfActiveOMPThreads);
    break;
  }
  case kmp_sched_distr_static_chunk: {
    if (ChunkSize > 0) {
      ForStaticChunk(IsLastIter, LB, UB, Stride, ChunkSize, GetOmpTeamId(),
                     GetNumberOfOmpTeams());
      break;
    } // note: if ChunkSize <=0, use nochunk
  }
  case kmp_sched_distr_static_nochunk: {
    ForStaticNoChunk(IsLastIter, LB, UB, Stride, ChunkSize, GetOmpTeamId(),
                     GetNumberOfOmpTeams());
    break;
  }
  case kmp_sched_distr_static_chunk_sched_static_chunkone: {
    ForStaticChunk(IsLastIter, LB, UB, Stride, ChunkSize,
                   numberOfActiveOMPThreads * GetOmpTeamId() + GlobalTID,
                   GetNumberOfOmpTeams() * numberOfActiveOMPThreads);
    break;
  }
  default: {
    ASSERT(LT_FUSSY, FALSE, "unknown ScheduleType %d", (int)ScheduleType);
    PRINT(LD_LOOP, "unknown ScheduleType %d, revert back to static chunk\n",
          (int)ScheduleType);
    ForStaticChunk(IsLastIter, LB, UB, Stride, ChunkSize, GlobalTID,
                   numberOfActiveOMPThreads);
    break;
  }
  }
  // copy back
  *IsLastIterPtr = IsLastIter;
  *LowerBoundPtr = LB;
  *UpperBoundPtr = UB;
  *StridePtr = Stride;
  PRINT(LD_LOOP,
        "Got sched: Active %d, total %d: lb %lld, ub %lld, stride %lld, last "
        "%d\n",
        (int)numberOfActiveOMPThreads, (int)GetNumberOfWorkersInTeam(),
        (long long)(*LowerBoundPtr), (long long)(*UpperBoundPtr),
        (long long)(*StridePtr), (int)IsLastIter);
}

#define FOR_STATIC_GEN(SUFFIX, IVTy, SIVTy, TIDTy, SPMD)                       \
  EXTERN void __kmpc_for_static_init##SUFFIX(                                  \
      kmp_Ident *Loc, TIDTy TId, TIDTy ScheduleType, TIDTy *IsLast, IVTy *LB,  \
      IVTy *UB, SIVTy *Stride, SIVTy Incr, SIVTy ChunkSize) {                  \
    PRINT0(LD_IO, "call kmpc_for_static_init" #SUFFIX "\n");                   \
    for_static_init<IVTy, SIVTy>(TId, ScheduleType, IsLast, LB, UB, Stride,    \
                                 ChunkSize, SPMD);                             \
  }

FOR_STATIC_GEN(_4, int32_t, int32_t, int32_t, checkSPMDMode(Loc))
FOR_STATIC_GEN(_4u, uint32_t, int32_t, int32_t, checkSPMDMode(Loc))
FOR_STATIC_GEN(_8, int64_t, int64_t, int32_t, checkSPMDMode(Loc))
FOR_STATIC_GEN(_8u, uint64_t, int64_t, int32_t, checkSPMDMode(Loc))
FOR_STATIC_GEN(_4_simple_spmd, int32_t, int32_t, int32_t, true)
FOR_STATIC_GEN(_4u_simple_spmd, uint32_t, int32_t, int32_t, true)
FOR_STATIC_GEN(_8_simple_spmd, int64_t, int64_t, int32_t, true)
FOR_STATIC_GEN(_8u_simple_spmd, uint64_t, int64_t, int32_t, true)
FOR_STATIC_GEN(_4_simple_generic, int32_t, int32_t, int32_t, false)
FOR_STATIC_GEN(_4u_simple_generic, uint32_t, int32_t, int32_t, false)
FOR_STATIC_GEN(_8_simple_generic, int64_t, int64_t, int32_t, false)
FOR_STATIC_GEN(_8u_simple_generic, uint64_t, int64_t, int32_t, false)
#undef FOR_STATIC_GEN

EXTERN void __kmpc_for_static_fini(kmp_Ident *Loc, int32_t global_tid) {
  PRINT0(LD_IO, "call kmpc_for_static_fini\n");
}

/// Return true if \p Schedule guarantees an order for the loop iterations.
INLINE static int isOrderedSchedule(kmp_sched_t Schedule) {
  return Schedule >= kmp_sched_ordered_first &&
         Schedule <= kmp_sched_ordered_last;
}

///
/// IVTy is type of the loop iteration variable and SIVTy is the signed version
/// of IVTy.
template <typename IVTy, typename SIVTy>
INLINE static void dispatch_init(kmp_Ident *Loc, int32_t threadId,
                                 kmp_sched_t Schedule, IVTy LB, IVTy UB,
                                 SIVTy st, SIVTy ChunkSize) {
  bool IsSPMD = checkSPMDMode(Loc);
  if (checkRuntimeUninitialized(Loc)) {
    // In SPMD mode no need to check parallelism level - dynamic scheduling
    // may appear only in L2 parallel regions with lightweight runtime.
    ASSERT0(LT_FUSSY, IsSPMD, "Expected non-SPMD mode.");
    return;
  }

  int tid = GetLogicalThreadIdInBlock(IsSPMD);
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(tid);
  IVTy tnum = GetNumberOfOmpThreads(IsSPMD);
  IVTy tripCount = UB - LB + 1; // +1 because UB is inclusive
  ASSERT0(LT_FUSSY, threadId < tnum,
          "current thread is not needed here; error");

  /* Currently just ignore the monotonic and non-monotonic modifiers
   * (the compiler isn't producing them * yet anyway).
   * When it is we'll want to look at them somewhere here and use that
   * information to add to our schedule choice. We shouldn't need to pass
   * them on, they merely affect which schedule we can legally choose for
   * various dynamic cases. (In paritcular, whether or not a stealing scheme
   * is legal).
   */
  Schedule = SCHEDULE_WITHOUT_MODIFIERS(Schedule);

  // Process schedule.
  if (tnum == 1 || tripCount <= 1 || isOrderedSchedule(Schedule)) {
    if (isOrderedSchedule(Schedule))
      __kmpc_barrier(Loc, threadId);
    PRINT(LD_LOOP,
          "go sequential as tnum=%ld, trip count %lld, ordered sched=%d\n",
          (long)tnum, (long long)tripCount, (int)Schedule);
    Schedule = kmp_sched_static_chunk;
    ChunkSize = tripCount; // one thread gets the whole loop
  } else if (Schedule == kmp_sched_runtime) {
    // process runtime
    omp_sched_t rtSched = currTaskDescr->GetRuntimeSched();
    ChunkSize = currTaskDescr->RuntimeChunkSize();
    switch (rtSched) {
    case omp_sched_static: {
      if (ChunkSize > 0)
        Schedule = kmp_sched_static_chunk;
      else
        Schedule = kmp_sched_static_nochunk;
      break;
    }
    case omp_sched_auto: {
      Schedule = kmp_sched_static_chunk;
      ChunkSize = 1;
      break;
    }
    case omp_sched_dynamic:
    case omp_sched_guided: {
      Schedule = kmp_sched_dynamic;
      break;
    }
    }
    PRINT(LD_LOOP, "Runtime sched is %d with chunk %lld\n", (int)Schedule,
          (long long)ChunkSize);
  } else if (Schedule == kmp_sched_auto) {
    Schedule = kmp_sched_static_chunk;
    ChunkSize = 1;
    PRINT(LD_LOOP, "Auto sched is %d with chunk %lld\n", (int)Schedule,
          (long long)ChunkSize);
  } else {
    PRINT(LD_LOOP, "Dyn sched is %d with chunk %lld\n", (int)Schedule,
          (long long)ChunkSize);
    ASSERT(LT_FUSSY,
           Schedule == kmp_sched_dynamic || Schedule == kmp_sched_guided,
           "unknown schedule %d & chunk %lld\n", (int)Schedule,
           (long long)ChunkSize);
  }

  // init schedules
  if (Schedule == kmp_sched_static_chunk) {
    ASSERT0(LT_FUSSY, ChunkSize > 0, "bad chunk value");
    // save sched state
    omptarget_nvptx_threadPrivateContext->ScheduleType(tid) = Schedule;
    // save UB
    omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid) = UB;
    // compute static chunk
    SIVTy Stride;
    int IsLastIter = 0;
    ForStaticChunk(IsLastIter, LB, UB, Stride, ChunkSize, threadId, tnum);
    // save computed params
    omptarget_nvptx_threadPrivateContext->Chunk(tid) = ChunkSize;
    omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = LB;
    omptarget_nvptx_threadPrivateContext->Stride(tid) = Stride;
    PRINT(
        LD_LOOP,
        "dispatch init (static chunk) : num threads = %d, ub =  %" PRId64
        ", next lower bound = %llu, stride = %llu\n",
        (int)tnum, omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid),
        (unsigned long long)
            omptarget_nvptx_threadPrivateContext->NextLowerBound(tid),
        (unsigned long long)omptarget_nvptx_threadPrivateContext->Stride(tid));
  } else if (Schedule == kmp_sched_static_balanced_chunk) {
    ASSERT0(LT_FUSSY, ChunkSize > 0, "bad chunk value");
    // save sched state
    omptarget_nvptx_threadPrivateContext->ScheduleType(tid) = Schedule;
    // save UB
    omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid) = UB;
    // compute static chunk
    SIVTy Stride;
    int IsLastIter = 0;
    // round up to make sure the chunk is enough to cover all iterations
    IVTy span = (tripCount + tnum - 1) / tnum;
    // perform chunk adjustment
    ChunkSize = (span + ChunkSize - 1) & ~(ChunkSize - 1);

    IVTy oldUb = UB;
    ForStaticChunk(IsLastIter, LB, UB, Stride, ChunkSize, threadId, tnum);
    ASSERT0(LT_FUSSY, UB >= LB, "ub must be >= lb.");
    if (UB > oldUb)
      UB = oldUb;
    // save computed params
    omptarget_nvptx_threadPrivateContext->Chunk(tid) = ChunkSize;
    omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = LB;
    omptarget_nvptx_threadPrivateContext->Stride(tid) = Stride;
    PRINT(
        LD_LOOP,
        "dispatch init (static chunk) : num threads = %d, ub =  %" PRId64
        ", next lower bound = %llu, stride = %llu\n",
        (int)tnum, omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid),
        (unsigned long long)
            omptarget_nvptx_threadPrivateContext->NextLowerBound(tid),
        (unsigned long long)omptarget_nvptx_threadPrivateContext->Stride(tid));
  } else if (Schedule == kmp_sched_static_nochunk) {
    ASSERT0(LT_FUSSY, ChunkSize == 0, "bad chunk value");
    // save sched state
    omptarget_nvptx_threadPrivateContext->ScheduleType(tid) = Schedule;
    // save UB
    omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid) = UB;
    // compute static chunk
    SIVTy Stride;
    int IsLastIter = 0;
    ForStaticNoChunk(IsLastIter, LB, UB, Stride, ChunkSize, threadId, tnum);
    // save computed params
    omptarget_nvptx_threadPrivateContext->Chunk(tid) = ChunkSize;
    omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = LB;
    omptarget_nvptx_threadPrivateContext->Stride(tid) = Stride;
    PRINT(
        LD_LOOP,
        "dispatch init (static nochunk) : num threads = %d, ub = %" PRId64
        ", next lower bound = %llu, stride = %llu\n",
        (int)tnum, omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid),
        (unsigned long long)
            omptarget_nvptx_threadPrivateContext->NextLowerBound(tid),
        (unsigned long long)omptarget_nvptx_threadPrivateContext->Stride(tid));
  } else if (Schedule == kmp_sched_dynamic || Schedule == kmp_sched_guided) {
    // save data
    omptarget_nvptx_threadPrivateContext->ScheduleType(tid) = Schedule;
    if (ChunkSize < 1)
      ChunkSize = 1;
    omptarget_nvptx_threadPrivateContext->Chunk(tid) = ChunkSize;
    omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid) = UB;
    omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = LB;
    __kmpc_barrier(Loc, threadId);
    if (tid == 0) {
      omptarget_nvptx_threadPrivateContext->Cnt() = 0;
      __threadfence_block();
    }
    __kmpc_barrier(Loc, threadId);
    PRINT(LD_LOOP,
          "dispatch init (dyn) : num threads = %d, lb = %llu, ub = %" PRId64
          ", chunk %" PRIu64 "\n",
          (int)tnum,
          (unsigned long long)
              omptarget_nvptx_threadPrivateContext->NextLowerBound(tid),
          omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid),
          omptarget_nvptx_threadPrivateContext->Chunk(tid));
  }
}

////////////////////////////////////////////////////////////////////////////////
// Support for dispatch next

INLINE static int64_t shuffle(unsigned active, int64_t val, int leader) {
  int lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(val));
  hi = __SHFL_SYNC(active, hi, leader);
  lo = __SHFL_SYNC(active, lo, leader);
  asm volatile("mov.b64 %0, {%1,%2};" : "=l"(val) : "r"(lo), "r"(hi));
  return val;
}

INLINE static uint64_t nextIter() {
  unsigned int active = __ACTIVEMASK();
  int leader = __ffs(active) - 1;
  int change = __popc(active);
  unsigned lane_mask_lt;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
  unsigned int rank = __popc(active & lane_mask_lt);
  uint64_t warp_res;
  if (rank == 0) {
    warp_res = atomicAdd(
        (unsigned long long *)&omptarget_nvptx_threadPrivateContext->Cnt(),
        change);
  }
  warp_res = shuffle(active, warp_res, leader);
  return warp_res + rank;
}

INLINE static int DynamicNextChunk(IVTy &LB, IVTy &UB, IVTy chunkSize,
                                   IVTy loopLowerBound, IVTy loopUpperBound) {
  IVTy N = nextIter();
  LB = loopLowerBound + N * chunkSize;
  UB = LB + chunkSize - 1; // Clang uses i <= UB

  // 3 result cases:
  //  a. LB and UB < loopUpperBound --> NOT_FINISHED
  //  b. LB < loopUpperBound and UB >= loopUpperBound: last chunk -->
  //  NOT_FINISHED
  //  c. LB and UB >= loopUpperBound: empty chunk --> FINISHED
  // a.
  if (LB <= loopUpperBound && UB < loopUpperBound) {
    PRINT(LD_LOOPD, "lb %lld, ub %lld, loop ub %lld; not finished\n",
          (long long)LB, (long long)UB, (long long)loopUpperBound);
    return NOT_FINISHED;
  }
  // b.
  if (LB <= loopUpperBound) {
    PRINT(LD_LOOPD, "lb %lld, ub %lld, loop ub %lld; clip to loop ub\n",
          (long long)LB, (long long)UB, (long long)loopUpperBound);
    UB = loopUpperBound;
    return LAST_CHUNK;
  }
  // c. if we are here, we are in case 'c'
  LB = loopUpperBound + 2;
  UB = loopUpperBound + 1;
  PRINT(LD_LOOPD, "lb %lld, ub %lld, loop ub %lld; finished\n", (long long)LB,
        (long long)UB, (long long)loopUpperBound);
  return FINISHED;
}

INLINE static int dispatch_next(kmp_Ident *Loc, int32_t GlobalTID,
                                int32_t *IsLastIterPtr, IVTy *LowerBoundPtr,
                                IVTy *UpperBoundPtr, SIVTy *StridePtr) {
  bool IsSPMD = checkSPMDMode(Loc);
  if (checkRuntimeUninitialized(Loc)) {
    // In SPMD mode no need to check parallelism level - dynamic scheduling
    // may appear only in L2 parallel regions with lightweight runtime.
    ASSERT0(LT_FUSSY, IsSPMD, "Expected non-SPMD mode.");
    if (*IsLastIterPtr)
      return DISPATCH_FINISHED;
    *IsLastIterPtr = 1;
    return DISPATCH_NOTFINISHED;
  }
  // ID of a thread in its own warp

  // automatically selects thread or warp ID based on selected implementation
  int tid = GetLogicalThreadIdInBlock(IsSPMD);
  ASSERT0(LT_FUSSY, GlobalTID < GetNumberOfOmpThreads(IsSPMD),
          "current thread is not needed here; error");
  // retrieve schedule
  kmp_sched_t Schedule =
      omptarget_nvptx_threadPrivateContext->ScheduleType(tid);

  // xxx reduce to one
  if (Schedule == kmp_sched_static_chunk ||
      Schedule == kmp_sched_static_nochunk) {
    IVTy myLb = omptarget_nvptx_threadPrivateContext->NextLowerBound(tid);
    IVTy UB = omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid);
    // finished?
    if (myLb > UB) {
      PRINT(LD_LOOP, "static loop finished with myLb %lld, ub %lld\n",
            (long long)myLb, (long long)UB);
      return DISPATCH_FINISHED;
    }
    // not finished, save current bounds
    SIVTy ChunkSize = omptarget_nvptx_threadPrivateContext->Chunk(tid);
    *LowerBoundPtr = myLb;
    IVTy myUb = myLb + ChunkSize - 1; // Clang uses i <= ub
    if (myUb > UB)
      myUb = UB;
    *UpperBoundPtr = myUb;
    *IsLastIterPtr = (int32_t)(myUb == UB);

    // increment next lower bound by the stride
    SIVTy Stride = omptarget_nvptx_threadPrivateContext->Stride(tid);
    omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = myLb + Stride;
    PRINT(LD_LOOP, "static loop continues with myLb %lld, myUb %lld\n",
          (long long)*LowerBoundPtr, (long long)*UpperBoundPtr);
    return DISPATCH_NOTFINISHED;
  }
  ASSERT0(LT_FUSSY,
          Schedule == kmp_sched_dynamic || Schedule == kmp_sched_guided,
          "bad sched");
  IVTy myLb, myUb;
  int finished = DynamicNextChunk(
      myLb, myUb, omptarget_nvptx_threadPrivateContext->Chunk(tid),
      omptarget_nvptx_threadPrivateContext->NextLowerBound(tid),
      omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid));

  if (finished == FINISHED)
    return DISPATCH_FINISHED;

  // not finished (either not finished or last chunk)
  *IsLastIterPtr = (int32_t)(finished == LAST_CHUNK);
  *LowerBoundPtr = myLb;
  *UpperBoundPtr = myUb;
  *StridePtr = 1;

  PRINT(LD_LOOP,
        "Got sched: active %d, total %d: lb %lld, ub %lld, stride = %lld, "
        "last %d\n",
        (int)GetNumberOfOmpThreads(isSPMDMode()),
        (int)GetNumberOfWorkersInTeam(), (long long)*LowerBoundPtr,
        (long long)*UpperBoundPtr, (long long)*StridePtr, (int)*IsLastIterPtr);
  return DISPATCH_NOTFINISHED;
}

#define DISPATCH_GEN(SUFFIX, IVTy, SIVTy, TIDTy)                               \
  EXTERN void __kmpc_dispatch_init##SUFFIX(kmp_Ident *Loc, TIDTy TID,          \
                                           TIDTy Schedule, IVTy LB, IVTy UB,   \
                                           SIVTy Stride, SIVTy ChunkSize) {    \
    PRINT0(LD_IO, "call kmpc_dispatch_init" #SUFFIX "\n");                     \
    dispatch_init<IVTy, SIVTy>(Loc, TID, (kmp_sched_t)Schedule, LB, UB,        \
                               Stride, ChunkSize);                             \
  }                                                                            \
  EXTERN void __kmpc_dispatch_next##SUFFIX(kmp_Ident *Loc, TIDTy TID,          \
                                           TIDTy *IsLast, IVTy LB, IVTy UB,    \
                                           SIVTy Stride) {                     \
    PRINT0(LD_IO, "call kmpc_dispatch_next" #SUFFIX "\n");                     \
    dispatch_next<IVTy, SIVTy>(Loc, TID, IsLast, LB, UB, Stride);              \
  }                                                                            \
  EXTERN void __kmpc_dispatch_fini##SUFFIX(kmp_Ident *Loc, TIDTy TID) {        \
    PRINT0(LD_IO, "call kmpc_dispatch_fini" #SUFFIX "\n");                     \
  }

DISPATCH_GEN(_4, int32_t, int32_t, int32_t)
DISPATCH_GEN(_4u, uint32_t, int32_t, int32_t)
DISPATCH_GEN(_8, int64_t, int64_t, int32_t)
DISPATCH_GEN(_8u, uint64_t, int64_t, int32_t)
#undef DISPATCH_GEN

static INLINE void syncWorkersInGenericMode(uint32_t NumThreads) {
  int NumWarps = ((NumThreads + WARPSIZE - 1) / WARPSIZE);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  // On Volta and newer architectures we require that all lanes in
  // a warp (at least, all present for the kernel launch) participate in the
  // barrier.  This is enforced when launching the parallel region.  An
  // exception is when there are < WARPSIZE workers.  In this case only 1 worker
  // is started, so we don't need a barrier.
  if (NumThreads > 1) {
#endif
    named_sync(L1_BARRIER, WARPSIZE * NumWarps);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  }
#endif
}

EXTERN void __kmpc_reduce_conditional_lastprivate(kmp_Ident *Loc,
                                                  int32_t GlobalTID,
                                                  int32_t varNum, void *array) {
  PRINT0(LD_IO, "call to __kmpc_reduce_conditional_lastprivate(...)\n");
  ASSERT0(LT_FUSSY, checkRuntimeInitialized(Loc),
          "Expected non-SPMD mode + initialized runtime.");

  omptarget_nvptx_TeamDescr &teamDescr = getMyTeamDescriptor();
  uint32_t NumThreads = GetNumberOfOmpThreads(checkSPMDMode(Loc));
  uint64_t *Buffer = teamDescr.getLastprivateIterBuffer();
  for (unsigned i = 0; i < varNum; i++) {
    // Reset buffer.
    if (GlobalTID == 0)
      *Buffer = 0; // Reset to minimum loop iteration value.

    // Barrier.
    syncWorkersInGenericMode(NumThreads);

    // Atomic max of iterations.
    uint64_t *varArray = (uint64_t *)array;
    uint64_t elem = varArray[i];
    (void)atomicMax((unsigned long long int *)Buffer,
                    (unsigned long long int)elem);

    // Barrier.
    syncWorkersInGenericMode(NumThreads);

    // Read max value and update thread private array.
    varArray[i] = *Buffer;

    // Barrier.
    syncWorkersInGenericMode(NumThreads);
  }
}
