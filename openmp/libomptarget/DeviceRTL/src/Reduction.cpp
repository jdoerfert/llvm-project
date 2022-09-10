//===---- Reduction.cpp - OpenMP device reduction implementation - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of reduction with KMPC interface.
//
//===----------------------------------------------------------------------===//

#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

using namespace _OMP;

namespace {

#pragma omp begin declare target device_type(nohost)

void gpu_regular_warp_reduce(void *reduce_data, ShuffleReductFnTy shflFct) {
  for (uint32_t mask = mapping::getWarpSize() / 2; mask > 0; mask /= 2) {
    shflFct(reduce_data, /*LaneId - not used= */ 0,
            /*Offset = */ mask, /*AlgoVersion=*/0);
  }
}

void gpu_irregular_warp_reduce(void *reduce_data, ShuffleReductFnTy shflFct,
                               uint32_t size, uint32_t tid) {
  uint32_t curr_size;
  uint32_t mask;
  curr_size = size;
  mask = curr_size / 2;
  while (mask > 0) {
    shflFct(reduce_data, /*LaneId = */ tid, /*Offset=*/mask, /*AlgoVersion=*/1);
    curr_size = (curr_size + 1) / 2;
    mask = curr_size / 2;
  }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700
static uint32_t gpu_irregular_simd_reduce(void *reduce_data,
                                          ShuffleReductFnTy shflFct) {
  uint32_t size, remote_id, physical_lane_id;
  physical_lane_id = mapping::getThreadIdInBlock() % mapping::getWarpSize();
  __kmpc_impl_lanemask_t lanemask_lt = mapping::lanemaskLT();
  __kmpc_impl_lanemask_t Liveness = mapping::activemask();
  uint32_t logical_lane_id = utils::popc(Liveness & lanemask_lt) * 2;
  __kmpc_impl_lanemask_t lanemask_gt = mapping::lanemaskGT();
  do {
    Liveness = mapping::activemask();
    remote_id = utils::ffs(Liveness & lanemask_gt);
    size = utils::popc(Liveness);
    logical_lane_id /= 2;
    shflFct(reduce_data, /*LaneId =*/logical_lane_id,
            /*Offset=*/remote_id - 1 - physical_lane_id, /*AlgoVersion=*/2);
  } while (logical_lane_id % 2 == 0 && size > 1);
  return (logical_lane_id == 0);
}
#endif

static int32_t nvptx_parallel_reduce_nowait(int32_t TId, int32_t num_vars,
                                            uint64_t reduce_size,
                                            void *reduce_data,
                                            ShuffleReductFnTy shflFct,
                                            InterWarpCopyFnTy cpyFct,
                                            bool isSPMDExecutionMode, bool) {
  uint32_t BlockThreadId = mapping::getThreadIdInBlock();
  if (mapping::isMainThreadInGenericMode(/* IsSPMD */ false))
    BlockThreadId = 0;
  uint32_t NumThreads = omp_get_num_threads();
  if (NumThreads == 1)
    return 1;
    /*
     * This reduce function handles reduction within a team. It handles
     * parallel regions in both L1 and L2 parallelism levels. It also
     * supports Generic, SPMD, and NoOMP modes.
     *
     * 1. Reduce within a warp.
     * 2. Warp master copies value to warp 0 via shared memory.
     * 3. Warp 0 reduces to a single value.
     * 4. The reduced value is available in the thread that returns 1.
     */

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  uint32_t WarpsNeeded =
      (NumThreads + mapping::getWarpSize() - 1) / mapping::getWarpSize();
  uint32_t WarpId = mapping::getWarpId();

  // Volta execution model:
  // For the Generic execution mode a parallel region either has 1 thread and
  // beyond that, always a multiple of 32. For the SPMD execution mode we may
  // have any number of threads.
  if ((NumThreads % mapping::getWarpSize() == 0) || (WarpId < WarpsNeeded - 1))
    gpu_regular_warp_reduce(reduce_data, shflFct);
  else if (NumThreads > 1) // Only SPMD execution mode comes thru this case.
    gpu_irregular_warp_reduce(reduce_data, shflFct,
                              /*LaneCount=*/NumThreads % mapping::getWarpSize(),
                              /*LaneId=*/mapping::getThreadIdInBlock() %
                                  mapping::getWarpSize());

  // When we have more than [mapping::getWarpSize()] number of threads
  // a block reduction is performed here.
  //
  // Only L1 parallel region can enter this if condition.
  if (NumThreads > mapping::getWarpSize()) {
    // Gather all the reduced values from each warp
    // to the first warp.
    cpyFct(reduce_data, WarpsNeeded);

    if (WarpId == 0)
      gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded,
                                BlockThreadId);
  }
  return BlockThreadId == 0;
#else
  __kmpc_impl_lanemask_t Liveness = mapping::activemask();
  if (Liveness == lanes::All) // Full warp
    gpu_regular_warp_reduce(reduce_data, shflFct);
  else if (!(Liveness & (Liveness + 1))) // Partial warp but contiguous lanes
    gpu_irregular_warp_reduce(reduce_data, shflFct,
                              /*LaneCount=*/utils::popc(Liveness),
                              /*LaneId=*/mapping::getThreadIdInBlock() %
                                  mapping::getWarpSize());
  else { // Dispersed lanes. Only threads in L2
         // parallel region may enter here; return
         // early.
    return gpu_irregular_simd_reduce(reduce_data, shflFct);
  }

  // When we have more than [mapping::getWarpSize()] number of threads
  // a block reduction is performed here.
  //
  // Only L1 parallel region can enter this if condition.
  if (NumThreads > mapping::getWarpSize()) {
    uint32_t WarpsNeeded =
        (NumThreads + mapping::getWarpSize() - 1) / mapping::getWarpSize();
    // Gather all the reduced values from each warp
    // to the first warp.
    cpyFct(reduce_data, WarpsNeeded);

    uint32_t WarpId = BlockThreadId / mapping::getWarpSize();
    if (WarpId == 0)
      gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded,
                                BlockThreadId);

    return BlockThreadId == 0;
  }

  // Get the OMP thread Id. This is different from BlockThreadId in the case of
  // an L2 parallel region.
  return TId == 0;
#endif // __CUDA_ARCH__ >= 700
}

uint32_t roundToWarpsize(uint32_t s) {
  if (s < mapping::getWarpSize())
    return 1;
  return (s & ~(unsigned)(mapping::getWarpSize() - 1));
}

uint32_t kmpcMin(uint32_t x, uint32_t y) { return x < y ? x : y; }

static uint32_t IterCnt = 0;
static uint32_t Cnt = 0;

} // namespace

extern "C" {
int32_t __kmpc_nvptx_parallel_reduce_nowait_v2(
    IdentTy *Loc, int32_t TId, int32_t num_vars, uint64_t reduce_size,
    void *reduce_data, ShuffleReductFnTy shflFct, InterWarpCopyFnTy cpyFct) {
  FunctionTracingRAII();
  return nvptx_parallel_reduce_nowait(TId, num_vars, reduce_size, reduce_data,
                                      shflFct, cpyFct, mapping::isSPMDMode(),
                                      false);
}

int32_t __kmpc_nvptx_teams_reduce_nowait_v2(
    IdentTy *Loc, int32_t TId, void *GlobalBuffer, uint32_t num_of_records,
    void *reduce_data, ShuffleReductFnTy shflFct, InterWarpCopyFnTy cpyFct,
    ListGlobalFnTy lgcpyFct, ListGlobalFnTy lgredFct, ListGlobalFnTy glcpyFct,
    ListGlobalFnTy glredFct) {
  FunctionTracingRAII();

  // Terminate all threads in non-SPMD mode except for the master thread.
  uint32_t ThreadId = mapping::getThreadIdInBlock();
  if (mapping::isGenericMode()) {
    if (!mapping::isMainThreadInGenericMode())
      return 0;
    ThreadId = 0;
  }

  // In non-generic mode all workers participate in the teams reduction.
  // In generic mode only the team master participates in the teams
  // reduction because the workers are waiting for parallel work.
  uint32_t NumThreads = omp_get_num_threads();
  uint32_t TeamId = omp_get_team_num();
  uint32_t NumTeams = omp_get_num_teams();
  static unsigned SHARED(Bound);
  static unsigned SHARED(ChunkTeamCount);

  // Block progress for teams greater than the current upper
  // limit. We always only allow a number of teams less or equal
  // to the number of slots in the buffer.
  bool IsMaster = (ThreadId == 0);
  while (IsMaster) {
    Bound = atomic::load(&IterCnt, __ATOMIC_SEQ_CST);
    if (TeamId < Bound + num_of_records)
      break;
  }

  if (IsMaster) {
    int ModBockId = TeamId % num_of_records;
    if (TeamId < num_of_records) {
      lgcpyFct(GlobalBuffer, ModBockId, reduce_data);
    } else
      lgredFct(GlobalBuffer, ModBockId, reduce_data);

    fence::system(__ATOMIC_SEQ_CST);

    // Increment team counter.
    // This counter is incremented by all teams in the current
    // BUFFER_SIZE chunk.
    ChunkTeamCount = atomic::inc(&Cnt, num_of_records - 1u, __ATOMIC_SEQ_CST);
  }
  // Synchronize
  if (mapping::isSPMDMode())
    __kmpc_barrier(Loc, TId);

  // reduce_data is global or shared so before being reduced within the
  // warp we need to bring it in local memory:
  // local_reduce_data = reduce_data[i]
  //
  // Example for 3 reduction variables a, b, c (of potentially different
  // types):
  //
  // buffer layout (struct of arrays):
  // a, a, ..., a, b, b, ... b, c, c, ... c
  // |__________|
  //     num_of_records
  //
  // local_data_reduce layout (struct):
  // a, b, c
  //
  // Each thread will have a local struct containing the values to be
  // reduced:
  //      1. do reduction within each warp.
  //      2. do reduction across warps.
  //      3. write the final result to the main reduction variable
  //         by returning 1 in the thread holding the reduction result.

  // Check if this is the very last team.
  unsigned NumRecs = kmpcMin(NumTeams, uint32_t(num_of_records));
  if (ChunkTeamCount == NumTeams - Bound - 1) {
    //
    // Last team processing.
    //
    if (ThreadId >= NumRecs)
      return 0;
    NumThreads = roundToWarpsize(kmpcMin(NumThreads, NumRecs));
    if (ThreadId >= NumThreads)
      return 0;

    // Load from buffer and reduce.
    glcpyFct(GlobalBuffer, ThreadId, reduce_data);
    for (uint32_t i = NumThreads + ThreadId; i < NumRecs; i += NumThreads)
      glredFct(GlobalBuffer, i, reduce_data);

    // Reduce across warps to the warp master.
    if (NumThreads > 1) {
      gpu_regular_warp_reduce(reduce_data, shflFct);

      // When we have more than [mapping::getWarpSize()] number of threads
      // a block reduction is performed here.
      uint32_t ActiveThreads = kmpcMin(NumRecs, NumThreads);
      if (ActiveThreads > mapping::getWarpSize()) {
        uint32_t WarpsNeeded = (ActiveThreads + mapping::getWarpSize() - 1) /
                               mapping::getWarpSize();
        // Gather all the reduced values from each warp
        // to the first warp.
        cpyFct(reduce_data, WarpsNeeded);

        uint32_t WarpId = ThreadId / mapping::getWarpSize();
        if (WarpId == 0)
          gpu_irregular_warp_reduce(reduce_data, shflFct, WarpsNeeded,
                                    ThreadId);
      }
    }

    if (IsMaster) {
      Cnt = 0;
      IterCnt = 0;
      return 1;
    }
    return 0;
  }
  if (IsMaster && ChunkTeamCount == num_of_records - 1) {
    // Allow SIZE number of teams to proceed writing their
    // intermediate results to the global buffer.
    atomic::add(&IterCnt, uint32_t(num_of_records), __ATOMIC_SEQ_CST);
  }

  return 0;
}

void __kmpc_nvptx_end_reduce(int32_t TId) { FunctionTracingRAII(); }

void __kmpc_nvptx_end_reduce_nowait(int32_t TId) { FunctionTracingRAII(); }
}

enum class RedOp : int8_t {
  ADD,
  MUL,
  // ...
};

enum class RedDataType : int8_t {
  INT8,
  INT16,
  INT32,
  INT64,
  FLOAT,
  DOUBLE,
  CUSTOM
};

enum class RedWidth : int8_t {
  WARP,
  TEAM,
  LEAGUE,
};

enum RedChoice : int8_t {
  RED_ITEMS_FULLY = 1,
  RED_ITEMS_PARTIALLY = 2,
  RED_ATOMIC_WITH_OFFSET = 4,
  RED_ATOMIC_AFTER_TEAM = 8,
  RED_ATOMIC_AFTER_WARP = 16,
  RED_LEAGUE_BUFFERED_DIRECT = 32,
  RED_LEAGUE_BUFFERED_ATOMIC = 64,
  RED_LEAGUE_BUFFERED_SYNCHRONIZED = 124,
};

struct ReductionInfo {
  RedOp Op;
  RedDataType DT;
  RedWidth Width;
  RedChoice RC;
  int8_t BatchSize;
  int32_t NumParticipants;
  int32_t NumElements;
  uint32_t* LeagueCounterPtr;
  char *LeagueBuffer = nullptr;
  void *CopyConstWrapper = nullptr;
};

template <typename Ty>
static void __llvm_omp_tgt_reduce_update_with_value(Ty *TypedOutput, Ty Value,
                                                    enum RedOp ROp,
                                                    bool Atomic) {
  switch (ROp) {
  case RedOp::ADD:
    if (Atomic)
      atomic::add((uint32_t *)TypedOutput, Value, __ATOMIC_SEQ_CST);
    else
      *TypedOutput += Value;
    break;
  case RedOp::MUL:
    break;
  default:
    __builtin_unreachable();
  }
}

template <typename Ty, int32_t InitDelta>
static void __llvm_omp_tgt_reduce_warp_typed_impl_specialized(
    Ty *Values, enum RedOp ROp, int32_t BatchSize, Ty *Out = nullptr) {
  int32_t Delta = InitDelta;
  do {
    Delta /= 2;
    for (int32_t i = 0; i < BatchSize; ++i) {
      Ty Acc = Values[i];
      switch (ROp) {
      case RedOp::ADD:
        Acc += utils::shuffleDown(-1, Acc, Delta, InitDelta);
        break;
      case RedOp::MUL:
        Acc *= utils::shuffleDown(-1, Acc, Delta, InitDelta);
        break;
      default:
        __builtin_unreachable();
      };
      if (Out)
        atomic::add((uint32_t *)&Out[i], Acc, __ATOMIC_SEQ_CST);
      else
        Values[i] = Acc;
    }
  } while (Delta > 1);
}

template <typename Ty, bool Partial = false>
static void
__llvm_omp_tgt_reduce_warp_typed_impl(Ty *Values, enum RedOp ROp, int32_t Width,
                                      int32_t BatchSize, Ty *Out = nullptr) {
  // We use the Width to prevent us from shuffling dead values into the result.
  // To simplify the code we will always do 5-6 shuffles though even if the
  // width could be checked.
  // printf("WR: W %i : BS %i\n", Width, BatchSize);
  int32_t Delta =
      mapping::getWarpSize() > Width ? Width : mapping::getWarpSize();
  //if (!Partial)
    //Delta = mapping::getWarpSize();
  // printf("WR: D %i : W %i : BS %i\n", Delta, Width, BatchSize);
  switch (Delta) {
  case 64:
    return __llvm_omp_tgt_reduce_warp_typed_impl_specialized<Ty, 64>(
        Values, ROp, BatchSize, Out);
  case 32:
    return __llvm_omp_tgt_reduce_warp_typed_impl_specialized<Ty, 32>(
        Values, ROp, BatchSize, Out);
  case 16:
    return __llvm_omp_tgt_reduce_warp_typed_impl_specialized<Ty, 16>(
        Values, ROp, BatchSize, Out);
  case 8:
    return __llvm_omp_tgt_reduce_warp_typed_impl_specialized<Ty, 8>(
        Values, ROp, BatchSize, Out);
  case 4:
    return __llvm_omp_tgt_reduce_warp_typed_impl_specialized<Ty, 4>(
        Values, ROp, BatchSize, Out);
  case 2:
    return __llvm_omp_tgt_reduce_warp_typed_impl_specialized<Ty, 2>(
        Values, ROp, BatchSize, Out);
  case 1:
    return;
  default:
    __builtin_unreachable();
  };
}

template <typename Ty>
static void __llvm_omp_tgt_reduce_warp_typed(IdentTy *Loc, ReductionInfo *RI,
                                             char *Input) {
  int32_t NumParticipants =
      RI->NumParticipants ? RI->NumParticipants : mapping::getWarpSize();
  Ty *TypedInput = reinterpret_cast<Ty *>(Input);

  __llvm_omp_tgt_reduce_warp_typed_impl(TypedInput, RI->Op, NumParticipants,
                                        RI->BatchSize);
}

static void __llvm_omp_tgt_reduce_warp(IdentTy *Loc, ReductionInfo *RI,
                                       char *Input) {
  switch (RI->DT) {
  case RedDataType::INT8:
    return __llvm_omp_tgt_reduce_warp_typed<int8_t>(Loc, RI, Input);
  case RedDataType::INT16:
    return __llvm_omp_tgt_reduce_warp_typed<int16_t>(Loc, RI, Input);
  case RedDataType::INT32:
    return __llvm_omp_tgt_reduce_warp_typed<int32_t>(Loc, RI, Input);
  case RedDataType::INT64:
    return __llvm_omp_tgt_reduce_warp_typed<int64_t>(Loc, RI, Input);
  case RedDataType::FLOAT:
    return __llvm_omp_tgt_reduce_warp_typed<float>(Loc, RI, Input);
  case RedDataType::DOUBLE:
    return __llvm_omp_tgt_reduce_warp_typed<double>(Loc, RI, Input);
  default:
    // TODO
    __builtin_trap();
  };
};

[[clang::loader_uninitialized]] static char TeamReductionScratchpad[64 * 16 * 8]
    __attribute__((aligned(32)));
#pragma omp allocate(TeamReductionScratchpad) allocator(omp_pteam_mem_alloc)

template <typename Ty, bool UseOutput, bool UseAtomic>
static void
__llvm_omp_tgt_reduce_team_typed_impl(IdentTy *Loc, ReductionInfo *RI,
                                      Ty *TypedInput, Ty *TypedOutput) {
   //printf("%s\n", __PRETTY_FUNCTION__);
  //  TODO: Verify the "Width" of the shuffles using tests with < WarpSize
  //  threads and others that have less than 32 Warps in use.
  int32_t NumParticipants =
      RI->NumParticipants ? RI->NumParticipants : mapping::getBlockSize();
  // printf("PART %i, FULL %i, NP %i, In %i, Out %i\n",(RI->RC &
  // RedChoice::RED_ITEMS_PARTIALLY),(RI->RC & RedChoice::RED_ITEMS_FULLY),
  // NumParticipants, *TypedInput, *TypedOutput);

  Ty *Out = nullptr;
  if (RI->RC & RedChoice::RED_ATOMIC_AFTER_WARP)
    Out = TypedOutput;

  // First reduce the values per warp.
  __llvm_omp_tgt_reduce_warp_typed_impl<Ty>(TypedInput, RI->Op, NumParticipants,
                                            RI->BatchSize, Out);

  if (RI->RC & RedChoice::RED_ITEMS_PARTIALLY) {

    for (int32_t i = RI->BatchSize; i < RI->NumElements; i += RI->BatchSize) {
      __llvm_omp_tgt_reduce_warp_typed_impl<Ty>(&TypedInput[i], RI->Op,
                                                NumParticipants, RI->BatchSize,
                                                Out ? &Out[i] : nullptr);
    }
  }

  if (RI->RC & RedChoice::RED_ATOMIC_AFTER_WARP)
    return;

  // if (OMP_UNLIKELY(NumParticipants <= mapping::getWarpSize()))
  // return;

  Ty *SharedMem = reinterpret_cast<Ty *>(&TeamReductionScratchpad[0]);
  int32_t WarpId = mapping::getWarpId();
  int32_t BlockId = mapping::getBlockId();

  int32_t NumWarps = mapping::getNumberOfWarpsInBlock();
  int32_t WarpTId = mapping::getThreadIdInWarp();
  int32_t IsWarpLead = WarpTId == 0;

  int32_t Idx = 0;
  do {
    if (/* UseLeaderInput */ false) {
      Ty* TypedLeaderInput = &TypedInput[-1 * (WarpTId * RI->NumElements)];
      for (int32_t i = WarpTId; i < RI->BatchSize;  i += mapping::getWarpSize()) {
        //if (BlockId == 0)
          //printf("SM: %i = %i\n", WarpId * RI->BatchSize + i , TypedInput[Idx + i]);
        SharedMem[WarpId * RI->BatchSize + i] = TypedLeaderInput[Idx + i];
      }
    } else if (IsWarpLead) {
      for (int32_t i = 0; i < RI->BatchSize;  ++i) {
        //if (BlockId == 0)
          //printf("SM: %i = %i\n", WarpId * RI->BatchSize + i , TypedInput[Idx + i]);
        SharedMem[WarpId * RI->BatchSize + i] = TypedInput[Idx + i];
      }
    }

    //for (int32_t i = NumWarps + WarpId; i < mapping::getWarpSize(); i += NumWarps) {
    //}

    // Wait for all shared memory updates.
    synchronize::threads();

    // The first warp performs the final reduction and stores away the result.
    if (WarpId == 0) {
      // Accumulate the shared memory results through shuffles.
      //printf("Sync   warp %i NP %i\n", WarpTId, NumParticipants / mapping::getWarpSize());
      __llvm_omp_tgt_reduce_warp_typed_impl<Ty, /* Partial */ true>(
          &SharedMem[0], RI->Op, mapping::getNumberOfWarpsInBlock(),
          RI->BatchSize);
      //printf("Synced warp %i NP %i\n", WarpTId, NumParticipants / mapping::getWarpSize());

      //  Only the final result is needed.

      if (/* UseLeaderInput */ false) {
        Ty* TypedLeaderInput = &TypedInput[-1 * (WarpTId * RI->NumElements)];
        for (int32_t i = WarpTId; i < RI->BatchSize; i+=mapping::getWarpSize()) {
          if (UseOutput)
            __llvm_omp_tgt_reduce_update_with_value<Ty>(
                &TypedOutput[Idx + i], SharedMem[i], RI->Op, UseAtomic);
          else
              TypedLeaderInput[Idx + i] = SharedMem[i];
        }
      } else if (IsWarpLead) {
        for (int32_t i = 0; i < RI->BatchSize; ++i) {
          //printf("TI/O: %i = %i\n", Idx +i , SharedMem[i]);
          if (UseOutput)
            __llvm_omp_tgt_reduce_update_with_value<Ty>(
                &TypedOutput[Idx + i], SharedMem[i], RI->Op, UseAtomic);
          else
              TypedInput[Idx + i] = SharedMem[i];
        }
      }
    }

    if (!(RI->RC & RedChoice::RED_ITEMS_PARTIALLY))
      break;

    synchronize::threads();

    Idx += RI->BatchSize;
    // printf("New Idx %i,  %i\n", Idx, RI->NumElements);
  } while (Idx < RI->NumElements);
}

template <typename Ty, bool UseOutput, bool UseAtomic = false>
static void __llvm_omp_tgt_reduce_team_typed(IdentTy *Loc, ReductionInfo *RI,
                                             char *Input, char *Output) {
   //printf("%s\n", __PRETTY_FUNCTION__);
  Ty *TypedInput = reinterpret_cast<Ty *>(Input);
  Ty *TypedOutput = reinterpret_cast<Ty *>(Output);

  __llvm_omp_tgt_reduce_team_typed_impl<Ty, UseOutput, UseAtomic>(
      Loc, RI, TypedInput, TypedOutput);

  if (RI->RC & RedChoice::RED_ITEMS_PARTIALLY)
    return;

  for (int32_t i = RI->BatchSize; i < RI->NumElements; i += RI->BatchSize)
    __llvm_omp_tgt_reduce_team_typed_impl<Ty, UseOutput, UseAtomic>(
        Loc, RI, &TypedInput[i], &TypedOutput[i]);
}

static void __llvm_omp_tgt_reduce_team(IdentTy *Loc, ReductionInfo *RI,
                                       char *Input, char *Output) {
   //printf("%s\n", __PRETTY_FUNCTION__);
  switch (RI->DT) {
  case RedDataType::INT8:
    return __llvm_omp_tgt_reduce_team_typed<int8_t, true>(Loc, RI, Input,
                                                          Output);
  case RedDataType::INT16:
    return __llvm_omp_tgt_reduce_team_typed<int16_t, true>(Loc, RI, Input,
                                                           Output);
  case RedDataType::INT32:
    return __llvm_omp_tgt_reduce_team_typed<int32_t, true>(Loc, RI, Input,
                                                           Output);
  case RedDataType::INT64:
    return __llvm_omp_tgt_reduce_team_typed<int64_t, true>(Loc, RI, Input,
                                                           Output);
  case RedDataType::FLOAT:
    return __llvm_omp_tgt_reduce_team_typed<float, true>(Loc, RI, Input,
                                                         Output);
  case RedDataType::DOUBLE:
    return __llvm_omp_tgt_reduce_team_typed<double, true>(Loc, RI, Input,
                                                          Output);
  default:
    // TODO
    __builtin_trap();
  };
}

template <typename Ty>
static void
__llvm_omp_tgt_reduce_league_typed_accumulate(IdentTy *Loc, ReductionInfo *RI,
                                              Ty *TypedInput, Ty *TypedOutput) {
  int32_t NumBlocks = mapping::getNumberOfBlocks();
  int32_t BlockId = mapping::getBlockId();
  int32_t TId = mapping::getThreadIdInBlock();
  int32_t NumThreads = mapping::getBlockSize();

  if (RI->RC & RedChoice::RED_LEAGUE_BUFFERED_DIRECT ||
      RI->RC & RedChoice::RED_LEAGUE_BUFFERED_ATOMIC) {

    synchronize::threads();

    Ty *TypedBuffer = reinterpret_cast<Ty *>(RI->LeagueBuffer);

    bool UseAtomic = RI->RC & RedChoice::RED_LEAGUE_BUFFERED_ATOMIC;

    if (/* UseLeaderInput */ false) {
      Ty* TypedLeaderInput = &TypedInput[-1 * (TId * RI->NumElements)];
      for (int32_t Idx = TId; Idx < RI->NumElements; Idx += NumThreads) {
        if (UseAtomic)
          __llvm_omp_tgt_reduce_update_with_value<Ty>(
              &TypedBuffer[BlockId * RI->NumElements + Idx], TypedLeaderInput[Idx],
              RI->Op, UseAtomic);
        else
          TypedBuffer[BlockId * RI->NumElements + Idx] = TypedLeaderInput[Idx];

        //if (BlockId == 0)
          //printf("%i : %i : %i :: %i (%i) --- %i (%p/%p)\n", TId, Idx, BlockId * RI->NumElements + Idx, TypedBuffer[BlockId * RI->NumElements + Idx], UseAtomic, TypedLeaderInput[Idx], &TypedLeaderInput[0], &TypedLeaderInput[Idx]);
      }
    } else if (TId == 0) {
      for (int32_t i = 0; i < RI->NumElements; ++i) {
        //if (BlockId == 0)
          //printf("%i : %i : %i :: %i (%i) --- %i (%p/%p)\n", TId, i, BlockId * RI->NumElements + i, TypedBuffer[BlockId * RI->NumElements + i], UseAtomic, TypedInput[i], &TypedInput[0], &TypedInput[i]);
        if (UseAtomic)
          __llvm_omp_tgt_reduce_update_with_value<Ty>(
              &TypedBuffer[BlockId * RI->NumElements + i], TypedInput[i],
              RI->Op, UseAtomic);
        else
          TypedBuffer[BlockId * RI->NumElements + i] = TypedInput[i];
        //if (BlockId == 0)
          //printf("%i : %i : %i :: %i (%i) --- %i (%p/%p)\n", TId, i, BlockId * RI->NumElements + i, TypedBuffer[BlockId * RI->NumElements + i], UseAtomic, TypedInput[i], &TypedInput[0], &TypedInput[i]);
      }
    }

    fence::system(__ATOMIC_SEQ_CST);

    synchronize::threads();

    static uint32_t SHARED(TeamLeagueCounter);

    if (TId == 0) {
      printf("TLC: %i\n", BlockId);
      TeamLeagueCounter =
          atomic::inc(RI->LeagueCounterPtr, NumBlocks - 1, __ATOMIC_SEQ_CST);
      printf("TLC: %i, %i\n", TeamLeagueCounter , BlockId);
    }

    synchronize::threads();

    if (TeamLeagueCounter != NumBlocks - 1)
      return;

    //Ty *SharedMem = reinterpret_cast<Ty *>(&TeamReductionScratchpad[0]);
    //for (int32_t i = 0; i < RI->NumElements; ++i)
      //SharedMem[TId + i] = TypedBuffer[TId * RI->NumElements + i];

    int32_t DstIdx = TId;
    int32_t SrcIdx = NumThreads * RI->NumElements + TId;
    while (SrcIdx < NumBlocks * RI->NumElements) {
      printf("1Acc %i -> %i :: %i\n" , SrcIdx, DstIdx, TypedBuffer[SrcIdx]);
      __llvm_omp_tgt_reduce_update_with_value<Ty>(
         &TypedBuffer[DstIdx], TypedBuffer[SrcIdx], RI->Op, false);
      printf("2Acc %i -> %i :: %i\n" , SrcIdx, DstIdx, TypedBuffer[DstIdx]);
      DstIdx = (DstIdx + NumThreads) % (NumThreads * RI->NumElements);
      SrcIdx += NumThreads;
    }

    synchronize::threads();

    ReductionInfo RI2 = *RI;
    RI2.NumParticipants = NumBlocks < NumThreads ? NumBlocks : NumThreads;
    if (TId < 10 || TId > 501)
      printf("BID %i TID %i : TB %i : NP %i\n", BlockId, TId, TypedBuffer[TId * RI->NumElements], RI->NumParticipants);

    if (RI->RC & RedChoice::RED_ATOMIC_AFTER_TEAM ||
        RI->RC & RedChoice::RED_ATOMIC_AFTER_WARP) {
      __llvm_omp_tgt_reduce_team_typed<Ty, true, true>(Loc, &RI2, (char*)&TypedBuffer[TId * RI->NumElements], (char*)TypedOutput);
    } else {
      __llvm_omp_tgt_reduce_team_typed<Ty, true, false>(Loc, &RI2, (char*)&TypedBuffer[TId * RI->NumElements], (char*)TypedOutput);
    }

  } else if (RI->RC & RedChoice::RED_LEAGUE_BUFFERED_SYNCHRONIZED) {
    __builtin_trap();
  } else {

    if (TId)
      return;

    int32_t StartIdx = 0;
    if (RI->RC & RedChoice::RED_ATOMIC_WITH_OFFSET)
      StartIdx = BlockId % RI->NumElements;

    // printf("TID %i BID %i S %i E %i, I %i, O %i\n", TId, BlockId, StartIdx,
    // RI->NumElements, *TypedInput, *TypedOutput);
    for (int32_t i = StartIdx; i < RI->NumElements; ++i) {
      atomic::add((uint32_t *)&TypedOutput[i], TypedInput[i], __ATOMIC_SEQ_CST);
    }
    for (int32_t i = 0; i < StartIdx; ++i) {
      atomic::add((uint32_t *)&TypedOutput[i], TypedInput[i], __ATOMIC_SEQ_CST);
    }
  }
}

template <typename Ty>
static void __llvm_omp_tgt_reduce_league_typed(IdentTy *Loc, ReductionInfo *RI,
                                               char *Input, char *Output) {
  Ty *TypedInput = reinterpret_cast<Ty *>(Input);
  Ty *TypedOutput = reinterpret_cast<Ty *>(Output);

  if (RI->RC & RedChoice::RED_ATOMIC_AFTER_TEAM ||
      RI->RC & RedChoice::RED_ATOMIC_AFTER_WARP) {
    __llvm_omp_tgt_reduce_team_typed<Ty, true, true>(Loc, RI, Input, Output);
    return;
  }

  __llvm_omp_tgt_reduce_team_typed<Ty, false>(Loc, RI, Input, nullptr);

  __llvm_omp_tgt_reduce_league_typed_accumulate<Ty>(Loc, RI, TypedInput,
                                                    TypedOutput);
}
static void __llvm_omp_tgt_reduce_league(IdentTy *Loc, ReductionInfo *RI,
                                         char *Input, char *Output) {
  switch (RI->DT) {
  case RedDataType::INT8:
    return __llvm_omp_tgt_reduce_league_typed<int8_t>(Loc, RI, Input, Output);
  case RedDataType::INT16:
    return __llvm_omp_tgt_reduce_league_typed<int16_t>(Loc, RI, Input, Output);
  case RedDataType::INT32:
    return __llvm_omp_tgt_reduce_league_typed<int32_t>(Loc, RI, Input, Output);
  case RedDataType::INT64:
    return __llvm_omp_tgt_reduce_league_typed<int64_t>(Loc, RI, Input, Output);
  case RedDataType::FLOAT:
    return __llvm_omp_tgt_reduce_league_typed<float>(Loc, RI, Input, Output);
  case RedDataType::DOUBLE:
    return __llvm_omp_tgt_reduce_league_typed<double>(Loc, RI, Input, Output);
  default:
    // TODO
    return;
  };
}

__attribute__((flatten, always_inline)) void
__llvm_omp_tgt_reduce(IdentTy *Loc, ReductionInfo *RI, char *Input,
                      char *Output) {
   //printf("%s\n", __PRETTY_FUNCTION__);
  switch (RI->Width) {
  case RedWidth::WARP:
    return __llvm_omp_tgt_reduce_warp(Loc, RI, Input);
  case RedWidth::TEAM:
    return __llvm_omp_tgt_reduce_team(Loc, RI, Input, Output);
  case RedWidth::LEAGUE:
    return __llvm_omp_tgt_reduce_league(Loc, RI, Input, Output);
  }
}

#if 0

template <typename Ty>
static Ty __llvm_omp_tgt_reduce_initial_value(enum RedOp ROp) {
  // TODO: This should be encoded in the ReductionInfo
  switch (ROp) {
  case RedOp::ADD:
    return (Ty(0));
  case RedOp::MUL:
    return (Ty(1));
  default:
    __builtin_unreachable();
  }
}

template <typename Ty, enum RedOp ROp>
static void __llvm_omp_tgt_reduce_league_standalone_impl(char *Input,
                                                         int32_t NumItems) {
  Ty *GlobalData = reinterpret_cast<Ty *>(Input);
  uint32_t ThreadId = mapping::getThreadIdInBlock();

  Ty InitialValue = __llvm_omp_tgt_reduce_initial_value<Ty>(ROp);
  Ty Accumulator = InitialValue;
  int32_t BlockSize = mapping::getBlockSize(/* IsSPMD */ true);
  int32_t TotalThreads = BlockSize * mapping::getNumberOfBlocks();

  // Reduce till we have no more input items than threads.
  {
    int32_t GlobalTId = mapping::getBlockId() * BlockSize + ThreadId;
    while (GlobalTId < NumItems) {
      switch (ROp) {
      case RedOp::ADD:
        Accumulator += GlobalData[GlobalTId];
        break;
      case RedOp::MUL:
        Accumulator *= GlobalData[GlobalTId];
        break;
      default:
        __builtin_trap();
      }
      GlobalTId += TotalThreads;
    }
  }

  [[clang::loader_uninitialized]] static Ty SharedMem[32]
      __attribute__((aligned(32)));
#pragma omp allocate(SharedMem) allocator(omp_pteam_mem_alloc)

  int32_t WarpId = mapping::getWarpId();
  __llvm_omp_tgt_reduce_warp_typed_impl<Ty>(&Accumulator, ROp,
                                            mapping::getWarpSize(), 1);
  if (ThreadId == 0)
    SharedMem[WarpId] = Accumulator;

  synchronize::threadsAligned();

  Accumulator = (ThreadId < mapping::getNumberOfWarpsInBlock())
                    ? SharedMem[ThreadId]
                    : InitialValue;

  if (WarpId == 0)
    __llvm_omp_tgt_reduce_warp_typed_impl<Ty>(&Accumulator, ROp,
                                              mapping::getWarpSize(), 1);

  if (ThreadId == 0)
    GlobalData[mapping::getBlockId()] = Accumulator;
}

#pragma omp end declare target

// Host and device code generation is required for the standalone kernels.
#pragma omp begin declare target

#define STANDALONE_LEAGUE_REDUCTION(Ty, ROp)                                   \
  omp_kernel void __llvm_omp_tgt_reduce_standalone_##Ty##_##ROp(               \
      char *Input, int32_t NumItems) {                                         \
    /* Use constant RI object once we start using copy function etc. */        \
    return __llvm_omp_tgt_reduce_league_standalone_impl<Ty, RedOp::ROp>(       \
        Input, NumItems);                                                      \
  }

STANDALONE_LEAGUE_REDUCTION(int32_t, ADD)
STANDALONE_LEAGUE_REDUCTION(int32_t, MUL)
STANDALONE_LEAGUE_REDUCTION(float, ADD)
STANDALONE_LEAGUE_REDUCTION(float, MUL)
STANDALONE_LEAGUE_REDUCTION(double, ADD)
STANDALONE_LEAGUE_REDUCTION(double, MUL)
#endif

#pragma omp end declare target
