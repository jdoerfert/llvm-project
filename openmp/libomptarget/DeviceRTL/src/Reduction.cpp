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

#pragma omp begin declare target device_type(nohost)

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

struct ReductionInfo {
  RedOp Op;
  RedDataType DT;
  RedWidth Width;
  int16_t NumThreads;
  int16_t ElementSize;
  void *CopyConstWrapper = nullptr;
};

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

// char *llvm_omp_target_dynamic_shared_alloc();

template <typename Ty>
static void __llvm_omp_tgt_reduce_op_add(Ty *TgtPtr, Ty *SrcPtr,
                                         int32_t NumPacketsPerThread) {
  for (int i = 0; i < NumPacketsPerThread; ++i) {
    // printf("T %i(%p) S %i(%p)\n", *((int32_t*)TgtPtr), TgtPtr,
    // *((int32_t*)SrcPtr), SrcPtr);
    TgtPtr[i] += SrcPtr[i];
    // printf("T %i(%p) S %i(%p)\n", *((int32_t*)TgtPtr), TgtPtr,
    // *((int32_t*)SrcPtr), SrcPtr);
  }
}
template <typename Ty>
static void __llvm_omp_tgt_reduce_op_mul(Ty *TgtPtr, Ty *SrcPtr,
                                         int32_t NumPacketsPerThread) {
  for (int i = 0; i < NumPacketsPerThread; ++i)
    TgtPtr[i] *= SrcPtr[i];
}

template <typename Ty>
static void __llvm_omp_tgt_reduce_in_memory(char *TgtPtrGen, char *SrcPtrGen,
                                            int32_t NumPacketsPerThread,
                                            ReductionInfo *RI) {
  Ty *TgtPtr = reinterpret_cast<Ty *>(TgtPtrGen);
  Ty *SrcPtr = reinterpret_cast<Ty *>(SrcPtrGen);
  // printf("rim %p %p, %i %i\n", TgtPtr, SrcPtr, RI->Op, RI->Op == RedOp::ADD);
  switch (RI->Op) {
  case RedOp::ADD:
    return __llvm_omp_tgt_reduce_op_add(TgtPtr, SrcPtr, NumPacketsPerThread);
  case RedOp::MUL:
    return __llvm_omp_tgt_reduce_op_mul(TgtPtr, SrcPtr, NumPacketsPerThread);
  default:
    return; // TODO
  }
}

static void __llvm_omp_tgt_reduce_in_memory(char *TgtPtr, char *SrcPtr,
                                            int32_t NumPacketsPerThread,
                                            ReductionInfo *RI) {
  switch (RI->DT) {
  case RedDataType::INT32:
    return __llvm_omp_tgt_reduce_in_memory<int32_t>(TgtPtr, SrcPtr,
                                                    NumPacketsPerThread, RI);
  case RedDataType::FLOAT:
    return __llvm_omp_tgt_reduce_in_memory<float>(TgtPtr, SrcPtr,
                                                  NumPacketsPerThread, RI);
  case RedDataType::DOUBLE:
    return __llvm_omp_tgt_reduce_in_memory<double>(TgtPtr, SrcPtr,
                                                   NumPacketsPerThread, RI);
  default:
    // TODO
    return;
  };
}

template <typename Ty>
static Ty __llvm_omp_tgt_reduce_warp_full(Ty Value, enum RedOp ROp) {
  int32_t Delta = mapping::getWarpSize();
  do {
    Delta /= 2;
    switch (ROp) {
    case RedOp::ADD:
      Value += utils::shuffleDown(-1, Value, Delta, mapping::getWarpSize());
      break;
    case RedOp::MUL:
      Value *= utils::shuffleDown(-1, Value, Delta, mapping::getWarpSize());
      break;
    default:
      __builtin_unreachable();
    };
  } while (Delta > 1);
  return Value;
}

static void __llvm_omp_tgt_reduce_warp(IdentTy *Loc, ReductionInfo *RI,
                                       char *Location) {
  LaneMaskTy Mask = mapping::activemask();
  int32_t Delta = mapping::getWarpSize();
  do {
    Delta /= 2;
    int32_t V = *((int32_t *)Location);
    int32_t R = utils::shuffleDown(Mask, V, Delta, mapping::getWarpSize());
    *((int32_t *)Location) = V + R;
  } while (Delta > 1);
}

// TODO: Use dynamic shared mem
[[clang::loader_uninitialized]] static char TeamSpaceTODO[32 * 8]
    __attribute__((aligned(32)));
#pragma omp allocate(TeamSpaceTODO) allocator(omp_pteam_mem_alloc)

static void __llvm_omp_tgt_reduce_team(IdentTy *Loc, ReductionInfo *RI,
                                       char *Location) {
  __llvm_omp_tgt_reduce_warp(Loc, RI, Location);

  int32_t NumThreads =
      RI->NumThreads ? RI->NumThreads : mapping::getBlockSize();
  if (NumThreads <= mapping::getWarpSize())
    return;

  int32_t PackedSize = (RI->ElementSize);
  int Idx = mapping::getThreadIdInWarp();

  // printf("warp %i/%i : %i\n", Idx, mapping::getWarpId(),
  // (*(int32_t*)(Location))); printf("NT %i : %i\n", NumThreads,
  // omp_get_num_threads());

  char *SharedMem = &TeamSpaceTODO[0]; // llvm_omp_target_dynamic_shared_alloc()
                                       // + MagicConstant;
  int32_t SharedMemIdx = mapping::getWarpId() * PackedSize;
  if (Idx == 0)
    __builtin_memcpy(&SharedMem[SharedMemIdx], Location, PackedSize);
  synchronize::threads();

  if (SharedMemIdx)
    return;

  __builtin_memcpy(Location, &SharedMem[Idx * PackedSize], PackedSize);

  __llvm_omp_tgt_reduce_warp(Loc, RI, Location);
}

template <typename Ty, enum RedOp ROp>
static void __llvm_omp_tgt_reduce_league_impl(char *Location,
                                              int32_t NumItems) {
  Ty *GlobalData = reinterpret_cast<Ty *>(Location);
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
        __llvm_omp_tgt_reduce_op_add(&Accumulator, &GlobalData[GlobalTId], 1);
      case RedOp::MUL:
        __llvm_omp_tgt_reduce_op_mul(&Accumulator, &GlobalData[GlobalTId], 1);
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
  Accumulator = __llvm_omp_tgt_reduce_warp_full<Ty>(Accumulator, ROp);
  if (ThreadId == 0)
    SharedMem[WarpId] = Accumulator;

  synchronize::threadsAligned();

  Accumulator = (ThreadId < mapping::getNumberOfWarpsInBlock())
                    ? SharedMem[ThreadId]
                    : InitialValue;

  if (WarpId == 0)
    Accumulator = __llvm_omp_tgt_reduce_warp_full<Ty>(Accumulator, ROp);

  if (ThreadId == 0)
    GlobalData[mapping::getBlockId()] = Accumulator;
}

template <typename Ty>
static void __llvm_omp_tgt_reduce_league_typed(IdentTy *Loc, ReductionInfo *RI,
                                               char *Location,
                                               int32_t NumItems) {
  switch (RI->Op) {
  case RedOp::ADD:
    return __llvm_omp_tgt_reduce_league_impl<Ty, RedOp::ADD>(Location,
                                                             NumItems);
  case RedOp::MUL:
    return __llvm_omp_tgt_reduce_league_impl<Ty, RedOp::MUL>(Location,
                                                             NumItems);
  default:
    return; // TODO
  }
}

static void __llvm_omp_tgt_reduce_league(IdentTy *Loc, ReductionInfo *RI,
                                         char *Location) {
  int32_t NumItems = RI->NumThreads ? RI->NumThreads : mapping::getBlockSize();
  switch (RI->DT) {
  case RedDataType::INT32:
    return __llvm_omp_tgt_reduce_league_typed<int32_t>(Loc, RI, Location,
                                                       NumItems);
  case RedDataType::FLOAT:
    return __llvm_omp_tgt_reduce_league_typed<float>(Loc, RI, Location,
                                                     NumItems);
  case RedDataType::DOUBLE:
    return __llvm_omp_tgt_reduce_league_typed<double>(Loc, RI, Location,
                                                      NumItems);
  default:
    // TODO
    return;
  };
}

__attribute__((flatten)) void
__llvm_omp_tgt_reduce(IdentTy *Loc, ReductionInfo *RI, char *Location) {
  switch (RI->Width) {
  case RedWidth::WARP:
    return __llvm_omp_tgt_reduce_warp(Loc, RI, Location);
  case RedWidth::TEAM:
    return __llvm_omp_tgt_reduce_team(Loc, RI, Location);
  case RedWidth::LEAGUE:
    return __llvm_omp_tgt_reduce_league(Loc, RI, Location);
  }
}
#pragma omp end declare target

// Host and device code generation is required for the standalone kernels.
#pragma omp begin declare target

#define STANDALONE_LEAGUE_REDUCTION(Ty, ROp)                                   \
  omp_kernel void __llvm_omp_tgt_reduce_standalone_##Ty##_##ROp(               \
      char *Location, int32_t NumItems) {                                      \
    /* Use constant RI object once we start using copy function etc. */        \
    return __llvm_omp_tgt_reduce_league_impl<Ty, RedOp::ROp>(Location,         \
                                                             NumItems);        \
  }

STANDALONE_LEAGUE_REDUCTION(int32_t, ADD)
STANDALONE_LEAGUE_REDUCTION(int32_t, MUL)
STANDALONE_LEAGUE_REDUCTION(float, ADD)
STANDALONE_LEAGUE_REDUCTION(float, MUL)
STANDALONE_LEAGUE_REDUCTION(double, ADD)
STANDALONE_LEAGUE_REDUCTION(double, MUL)

#pragma omp end declare target
