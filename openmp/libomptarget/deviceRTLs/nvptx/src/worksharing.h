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
/// chunk size \p Chunk. The \p EntityId describes the location of the executing
/// thread in the team and \p NumberOfEntities is the number of threads in the
/// team.
///
/// IVTy is type of the loop iteration variable and SIVTy is the signed version
/// of IVTy.
template <typename IVTy, typename SIVTy>
INLINE static void ForStaticChunk(int &last, IVTy &LB, IVTy &UB, SIVTy &stride,
                                  SIVTy Chunk, IVTy EntityId,
                                  IVTy NumberOfEntities) {
  // Each thread executes multiple chunks all of the same size.

  // Distance between two successive chunks
  stride = NumberOfEntities * Chunk;
  LB = LB + EntityId * Chunk;

  // Adjust the upper bound by one to match the less-than (<=) comparison
  // Clang uses, e.g., the loop condition will be i <= UB not i < UB.
  IVTy InputUb = UB;
  UB = LB + Chunk - 1;

  // Say UB' is the begining of the last chunk. Then who ever has a
  // lower bound plus a multiple of the increment equal to UB' is
  // the last one.
  IVTy beginingLastChunk = InputUb - (InputUb % Chunk);
  last = ((beginingLastChunk - LB) % stride) == 0;
}

/// Compute the lower bound (\p LB), upper bound (\p UB), stride (\p Stride),
/// and last iteration flag (\p IsLast) for a loop with static scheduling and
/// chunk size \p Chunk. The \p EntityId describes the location of the executing
/// thread in the team and \p NumberOfEntities is the number of threads in the
/// team.
///
/// IVTy is type of the loop iteration variable and SIVTy is the signed version
/// of IVTy.
template <typename IVTy, typename SIVTy>
INLINE static void ForStaticNoChunk(int &IsLast, IVTy &LB, IVTy &UB,
                                    SIVTy &stride, SIVTy &Chunk, IVTy EntityId,
                                    IVTy NumberOfEntities) {
  // No chunk size specified. Each thread gets at most one chunk; chunks are
  // all almost of equal size
  IVTy loopSize = UB - LB + 1;

  Chunk = loopSize / NumberOfEntities;
  IVTy leftOver = loopSize - Chunk * NumberOfEntities;

  if (EntityId < leftOver) {
    Chunk++;
    LB = LB + EntityId * Chunk;
  } else {
    LB = LB + EntityId * Chunk + leftOver;
  }

  // Adjust the upper bound by one to match the less-than (<=) comparison
  // Clang uses, e.g., the loop condition will be i <= UB not i < UB.
  IVTy InputUb = UB;
  UB = LB + Chunk - 1;

  IsLast = LB <= InputUb && InputUb <= UB;

  // Make sure we only do 1 chunk per warp.
  stride = loopSize;
}

/// Calculate initial bounds for static loop and stride
///
/// \param[in] Loc location in code of the call (not used here)
/// \param[in] global_tid global thread id
/// \param[in] schetype type of scheduling (see omptarget-nvptx.h)
/// \param[in] plastiter pointer to last iteration
/// \param[in,out] pointer to loop lower bound. it will contain value of
/// lower bound of first chunk
/// \param[in,out] pointer to loop upper bound. It will contain value of
/// upper bound of first chunk
/// \param[in,out] pointer to loop stride. It will contain value of stride
/// between two successive chunks executed by the same thread
/// \param[in] loop increment bump
/// \param[in] chunk size
///
/// IVTy is type of the loop iteration variable and SIVTy is the signed version
/// of IVTy.
template <typename IVTy, typename SIVTy>
INLINE static void for_static_init(int32_t gtid, int32_t schedtype,
                                   int32_t *plastiter, IVTy *plower,
                                   IVTy *pupper, SIVTy *pstride, SIVTy chunk,
                                   bool IsSPMDExecutionMode) {
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
        (int)gtid, (int)schedtype, (long long)chunk, (int)gtid,
        (int)numberOfActiveOMPThreads);
  ASSERT0(LT_FUSSY, gtid < numberOfActiveOMPThreads,
          "current thread is not needed here; error");

  // copy
  int lastiter = 0;
  IVTy lb = *plower;
  IVTy ub = *pupper;
  SIVTy stride = *pstride;
  // init
  switch (SCHEDULE_WITHOUT_MODIFIERS(schedtype)) {
  case kmp_sched_static_chunk: {
    if (chunk > 0) {
      ForStaticChunk(lastiter, lb, ub, stride, chunk, gtid,
                     numberOfActiveOMPThreads);
      break;
    }
  } // note: if chunk <=0, use nochunk
  case kmp_sched_static_balanced_chunk: {
    if (chunk > 0) {
      // round up to make sure the chunk is enough to cover all iterations
      IVTy tripCount = ub - lb + 1; // +1 because ub is inclusive
      IVTy span =
          (tripCount + numberOfActiveOMPThreads - 1) / numberOfActiveOMPThreads;
      // perform chunk adjustment
      chunk = (span + chunk - 1) & ~(chunk - 1);

      ASSERT0(LT_FUSSY, ub >= lb, "ub must be >= lb.");
      IVTy oldUb = ub;
      ForStaticChunk(lastiter, lb, ub, stride, chunk, gtid,
                     numberOfActiveOMPThreads);
      if (ub > oldUb)
        ub = oldUb;
      break;
    }
  } // note: if chunk <=0, use nochunk
  case kmp_sched_static_nochunk: {
    ForStaticNoChunk(lastiter, lb, ub, stride, chunk, gtid,
                     numberOfActiveOMPThreads);
    break;
  }
  case kmp_sched_distr_static_chunk: {
    if (chunk > 0) {
      ForStaticChunk(lastiter, lb, ub, stride, chunk, GetOmpTeamId(),
                     GetNumberOfOmpTeams());
      break;
    } // note: if chunk <=0, use nochunk
  }
  case kmp_sched_distr_static_nochunk: {
    ForStaticNoChunk(lastiter, lb, ub, stride, chunk, GetOmpTeamId(),
                     GetNumberOfOmpTeams());
    break;
  }
  case kmp_sched_distr_static_chunk_sched_static_chunkone: {
    ForStaticChunk(lastiter, lb, ub, stride, chunk,
                   numberOfActiveOMPThreads * GetOmpTeamId() + gtid,
                   GetNumberOfOmpTeams() * numberOfActiveOMPThreads);
    break;
  }
  default: {
    ASSERT(LT_FUSSY, FALSE, "unknown schedtype %d", (int)schedtype);
    PRINT(LD_LOOP, "unknown schedtype %d, revert back to static chunk\n",
          (int)schedtype);
    ForStaticChunk(lastiter, lb, ub, stride, chunk, gtid,
                   numberOfActiveOMPThreads);
    break;
  }
  }
  // copy back
  *plastiter = lastiter;
  *plower = lb;
  *pupper = ub;
  *pstride = stride;
  PRINT(LD_LOOP,
        "Got sched: Active %d, total %d: lb %lld, ub %lld, stride %lld, last "
        "%d\n",
        (int)numberOfActiveOMPThreads, (int)GetNumberOfWorkersInTeam(),
        (long long)(*plower), (long long)(*pupper), (long long)(*pstride),
        (int)lastiter);
}

#define FOR_STATIC_GEN(SUFFIX, IVTy, SIVTy, TIDTy, SPMD)                       \
  EXTERN void __kmpc_for_static_init##SUFFIX(                                  \
      kmp_Ident *Loc, TIDTy TId, TIDTy ScheduleType, TIDTy *IsLast, IVTy *LB,  \
      IVTy *UB, SIVTy *Stride, SIVTy Incr, SIVTy Chunk) {                      \
    PRINT0(LD_IO, "call kmpc_for_static_init" #SUFFIX "\n");                   \
    for_static_init<IVTy, SIVTy>(TId, ScheduleType, IsLast, LB, UB, Stride,    \
                                 Chunk, SPMD);                                 \
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
                                 kmp_sched_t Schedule, IVTy lb, IVTy ub,
                                 SIVTy st, SIVTy chunk) {
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
  IVTy tripCount = ub - lb + 1; // +1 because ub is inclusive
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
    chunk = tripCount; // one thread gets the whole loop
  } else if (Schedule == kmp_sched_runtime) {
    // process runtime
    omp_sched_t rtSched = currTaskDescr->GetRuntimeSched();
    chunk = currTaskDescr->RuntimeChunkSize();
    switch (rtSched) {
    case omp_sched_static: {
      if (chunk > 0)
        Schedule = kmp_sched_static_chunk;
      else
        Schedule = kmp_sched_static_nochunk;
      break;
    }
    case omp_sched_auto: {
      Schedule = kmp_sched_static_chunk;
      chunk = 1;
      break;
    }
    case omp_sched_dynamic:
    case omp_sched_guided: {
      Schedule = kmp_sched_dynamic;
      break;
    }
    }
    PRINT(LD_LOOP, "Runtime sched is %d with chunk %lld\n", (int)Schedule,
          (long long)chunk);
  } else if (Schedule == kmp_sched_auto) {
    Schedule = kmp_sched_static_chunk;
    chunk = 1;
    PRINT(LD_LOOP, "Auto sched is %d with chunk %lld\n", (int)Schedule,
          (long long)chunk);
  } else {
    PRINT(LD_LOOP, "Dyn sched is %d with chunk %lld\n", (int)Schedule,
          (long long)chunk);
    ASSERT(
        LT_FUSSY, Schedule == kmp_sched_dynamic || Schedule == kmp_sched_guided,
        "unknown schedule %d & chunk %lld\n", (int)Schedule, (long long)chunk);
  }

  // init schedules
  if (Schedule == kmp_sched_static_chunk) {
    ASSERT0(LT_FUSSY, chunk > 0, "bad chunk value");
    // save sched state
    omptarget_nvptx_threadPrivateContext->ScheduleType(tid) = Schedule;
    // save ub
    omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid) = ub;
    // compute static chunk
    SIVTy stride;
    int lastiter = 0;
    ForStaticChunk(lastiter, lb, ub, stride, chunk, threadId, tnum);
    // save computed params
    omptarget_nvptx_threadPrivateContext->Chunk(tid) = chunk;
    omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = lb;
    omptarget_nvptx_threadPrivateContext->Stride(tid) = stride;
    PRINT(
        LD_LOOP,
        "dispatch init (static chunk) : num threads = %d, ub =  %" PRId64
        ", next lower bound = %llu, stride = %llu\n",
        (int)tnum, omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid),
        (unsigned long long)
            omptarget_nvptx_threadPrivateContext->NextLowerBound(tid),
        (unsigned long long)omptarget_nvptx_threadPrivateContext->Stride(tid));
  } else if (Schedule == kmp_sched_static_balanced_chunk) {
    ASSERT0(LT_FUSSY, chunk > 0, "bad chunk value");
    // save sched state
    omptarget_nvptx_threadPrivateContext->ScheduleType(tid) = Schedule;
    // save ub
    omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid) = ub;
    // compute static chunk
    SIVTy stride;
    int lastiter = 0;
    // round up to make sure the chunk is enough to cover all iterations
    IVTy span = (tripCount + tnum - 1) / tnum;
    // perform chunk adjustment
    chunk = (span + chunk - 1) & ~(chunk - 1);

    IVTy oldUb = ub;
    ForStaticChunk(lastiter, lb, ub, stride, chunk, threadId, tnum);
    ASSERT0(LT_FUSSY, ub >= lb, "ub must be >= lb.");
    if (ub > oldUb)
      ub = oldUb;
    // save computed params
    omptarget_nvptx_threadPrivateContext->Chunk(tid) = chunk;
    omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = lb;
    omptarget_nvptx_threadPrivateContext->Stride(tid) = stride;
    PRINT(
        LD_LOOP,
        "dispatch init (static chunk) : num threads = %d, ub =  %" PRId64
        ", next lower bound = %llu, stride = %llu\n",
        (int)tnum, omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid),
        (unsigned long long)
            omptarget_nvptx_threadPrivateContext->NextLowerBound(tid),
        (unsigned long long)omptarget_nvptx_threadPrivateContext->Stride(tid));
  } else if (Schedule == kmp_sched_static_nochunk) {
    ASSERT0(LT_FUSSY, chunk == 0, "bad chunk value");
    // save sched state
    omptarget_nvptx_threadPrivateContext->ScheduleType(tid) = Schedule;
    // save ub
    omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid) = ub;
    // compute static chunk
    SIVTy stride;
    int lastiter = 0;
    ForStaticNoChunk(lastiter, lb, ub, stride, chunk, threadId, tnum);
    // save computed params
    omptarget_nvptx_threadPrivateContext->Chunk(tid) = chunk;
    omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = lb;
    omptarget_nvptx_threadPrivateContext->Stride(tid) = stride;
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
    if (chunk < 1)
      chunk = 1;
    omptarget_nvptx_threadPrivateContext->Chunk(tid) = chunk;
    omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid) = ub;
    omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = lb;
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

INLINE static int DynamicNextChunk(IVTy &lb, IVTy &ub, IVTy chunkSize,
                                   IVTy loopLowerBound, IVTy loopUpperBound) {
  IVTy N = nextIter();
  lb = loopLowerBound + N * chunkSize;
  ub = lb + chunkSize - 1; // Clang uses i <= ub

  // 3 result cases:
  //  a. lb and ub < loopUpperBound --> NOT_FINISHED
  //  b. lb < loopUpperBound and ub >= loopUpperBound: last chunk -->
  //  NOT_FINISHED
  //  c. lb and ub >= loopUpperBound: empty chunk --> FINISHED
  // a.
  if (lb <= loopUpperBound && ub < loopUpperBound) {
    PRINT(LD_LOOPD, "lb %lld, ub %lld, loop ub %lld; not finished\n",
          (long long)lb, (long long)ub, (long long)loopUpperBound);
    return NOT_FINISHED;
  }
  // b.
  if (lb <= loopUpperBound) {
    PRINT(LD_LOOPD, "lb %lld, ub %lld, loop ub %lld; clip to loop ub\n",
          (long long)lb, (long long)ub, (long long)loopUpperBound);
    ub = loopUpperBound;
    return LAST_CHUNK;
  }
  // c. if we are here, we are in case 'c'
  lb = loopUpperBound + 2;
  ub = loopUpperBound + 1;
  PRINT(LD_LOOPD, "lb %lld, ub %lld, loop ub %lld; finished\n", (long long)lb,
        (long long)ub, (long long)loopUpperBound);
  return FINISHED;
}

INLINE static int dispatch_next(kmp_Ident *Loc, int32_t gtid, int32_t *plast,
                                IVTy *plower, IVTy *pupper, SIVTy *pstride) {
  bool IsSPMD = checkSPMDMode(Loc);
  if (checkRuntimeUninitialized(Loc)) {
    // In SPMD mode no need to check parallelism level - dynamic scheduling
    // may appear only in L2 parallel regions with lightweight runtime.
    ASSERT0(LT_FUSSY, IsSPMD, "Expected non-SPMD mode.");
    if (*plast)
      return DISPATCH_FINISHED;
    *plast = 1;
    return DISPATCH_NOTFINISHED;
  }
  // ID of a thread in its own warp

  // automatically selects thread or warp ID based on selected implementation
  int tid = GetLogicalThreadIdInBlock(IsSPMD);
  ASSERT0(LT_FUSSY, gtid < GetNumberOfOmpThreads(IsSPMD),
          "current thread is not needed here; error");
  // retrieve schedule
  kmp_sched_t Schedule =
      omptarget_nvptx_threadPrivateContext->ScheduleType(tid);

  // xxx reduce to one
  if (Schedule == kmp_sched_static_chunk ||
      Schedule == kmp_sched_static_nochunk) {
    IVTy myLb = omptarget_nvptx_threadPrivateContext->NextLowerBound(tid);
    IVTy ub = omptarget_nvptx_threadPrivateContext->LoopUpperBound(tid);
    // finished?
    if (myLb > ub) {
      PRINT(LD_LOOP, "static loop finished with myLb %lld, ub %lld\n",
            (long long)myLb, (long long)ub);
      return DISPATCH_FINISHED;
    }
    // not finished, save current bounds
    SIVTy chunk = omptarget_nvptx_threadPrivateContext->Chunk(tid);
    *plower = myLb;
    IVTy myUb = myLb + chunk - 1; // Clang uses i <= ub
    if (myUb > ub)
      myUb = ub;
    *pupper = myUb;
    *plast = (int32_t)(myUb == ub);

    // increment next lower bound by the stride
    SIVTy stride = omptarget_nvptx_threadPrivateContext->Stride(tid);
    omptarget_nvptx_threadPrivateContext->NextLowerBound(tid) = myLb + stride;
    PRINT(LD_LOOP, "static loop continues with myLb %lld, myUb %lld\n",
          (long long)*plower, (long long)*pupper);
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
  *plast = (int32_t)(finished == LAST_CHUNK);
  *plower = myLb;
  *pupper = myUb;
  *pstride = 1;

  PRINT(LD_LOOP,
        "Got sched: active %d, total %d: lb %lld, ub %lld, stride = %lld, "
        "last %d\n",
        (int)GetNumberOfOmpThreads(isSPMDMode()),
        (int)GetNumberOfWorkersInTeam(), (long long)*plower, (long long)*pupper,
        (long long)*pstride, (int)*plast);
  return DISPATCH_NOTFINISHED;
}

#define DISPATCH_GEN(SUFFIX, IVTy, SIVTy, TIDTy)                               \
  EXTERN void __kmpc_dispatch_init##SUFFIX(kmp_Ident *Loc, TIDTy TID,          \
                                           TIDTy Schedule, IVTy LB, IVTy UB,   \
                                           SIVTy Stride, SIVTy Chunk) {        \
    PRINT0(LD_IO, "call kmpc_dispatch_init" #SUFFIX "\n");                     \
    dispatch_init<IVTy, SIVTy>(Loc, TID, (kmp_sched_t)Schedule, LB, UB,        \
                               Stride, Chunk);                                 \
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

EXTERN void __kmpc_reduce_conditional_lastprivate(kmp_Ident *Loc, int32_t gtid,
                                                  int32_t varNum, void *array) {
  PRINT0(LD_IO, "call to __kmpc_reduce_conditional_lastprivate(...)\n");
  ASSERT0(LT_FUSSY, checkRuntimeInitialized(Loc),
          "Expected non-SPMD mode + initialized runtime.");

  omptarget_nvptx_TeamDescr &teamDescr = getMyTeamDescriptor();
  uint32_t NumThreads = GetNumberOfOmpThreads(checkSPMDMode(Loc));
  uint64_t *Buffer = teamDescr.getLastprivateIterBuffer();
  for (unsigned i = 0; i < varNum; i++) {
    // Reset buffer.
    if (gtid == 0)
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
