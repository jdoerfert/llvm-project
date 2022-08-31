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

#include "Reduction.h"
#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

using namespace ompx;

#pragma omp begin declare target device_type(nohost)

namespace {

constexpr int32_t BufferAlignment = 64;
constexpr int32_t MaxWarpSize = 64;
constexpr int32_t MaxBlockSize = 32;
constexpr int32_t MaxWarpAndBlockSize = MaxWarpSize;
static_assert(MaxWarpSize >= MaxBlockSize,
              "Expected at least as meany warp threads as block threads.");

constexpr int32_t MaxBatchSize = 16;
constexpr int32_t MaxDataTypeSize = 16;
static_assert(MaxDataTypeSize >= sizeof(double) &&
                  MaxDataTypeSize >= sizeof(int64_t),
              "Max data type size is too small!");

using RedVarTy = __llvm_omp_default_reduction_var_configuration_ty;
using RedConfigTy = __llvm_omp_default_reduction_configuration_ty;
using RedOpTy = __llvm_omp_reduction_operation;
using ElementTypeTy = __llvm_omp_reduction_element_type;
using ReducerFnTy = __llvm_omp_reduction_reducer_fn_ty;
using RedPrivInfoTy = __llvm_omp_default_reduction_private_info_ty;
using RedLeagueConfigTy = __llvm_omp_default_reduction_league_configuration_ty;

template <typename Ty>
static void foreachVar(void *SrcPtr, RedConfigTy &Config, Ty Fn) {
  int32_t NumVars = Config.__num_reduction_vars;
  RedVarTy &Var = Config.__reduction_vars[0][0];
  Fn(SrcPtr, Var);
  SrcPtr = utils::advance(SrcPtr, Var.__item_size * Var.__num_items);
  if (NumVars > 1) {
    RedVarTy &Var = Config.__reduction_vars[0][1];
    Fn(SrcPtr, Var);
    SrcPtr = utils::advance(SrcPtr, Var.__item_size * Var.__num_items);
  }
  for (int32_t i = 2; i < NumVars; ++i) {
    RedVarTy &Var = Config.__reduction_vars[0][i];
    Fn(SrcPtr, Var);
    SrcPtr = utils::advance(SrcPtr, Var.__item_size * Var.__num_items);
  }
}

template <typename Ty>
static void forVarBatched(void *SrcPtr, RedConfigTy &Config, RedVarTy &Var,
                          Ty Fn) {
  int32_t NumItems = Var.__num_items;
  int32_t ItemSize = Var.__item_size;
  void *DstPtr = Var.__dst_ptr;
  auto BatchFn = [&](int32_t BatchSize) {
    while (NumItems >= BatchSize) {
      Fn(SrcPtr, Config, Var, DstPtr);
      SrcPtr = utils::advance(SrcPtr, BatchSize * ItemSize);
      DstPtr = utils::advance(DstPtr, BatchSize * ItemSize);
      NumItems -= BatchSize;
    }
  };
  int32_t BatchSize = Var.__batch_size;
  BatchFn(BatchSize);
  if (BatchSize >= 1 && NumItems)
    BatchFn(1);
}

template <typename Ty>
static void foreachVarBatched(void *SrcPtr, RedConfigTy &Config, Ty Fn) {
  auto VarFn = [&](void *SrcPtr, RedVarTy &Var) {
    forVarBatched(SrcPtr, Config, Var, Fn);
  };
  foreachVar(SrcPtr, Config, VarFn);
}

/// Helper methods
///{
///
#define TYPE_DEDUCER(FN_NAME)                                                  \
  void FN_NAME(void *DstPtr, void *SrcPtr, RedConfigTy &Config, RedVarTy &Var, \
               intptr_t Payload = 0) {                                         \
    switch (Var.__element_type) {                                              \
    case _INT8:                                                                \
      return FN_NAME<int8_t, int8_t>(reinterpret_cast<int8_t *>(DstPtr),       \
                                     reinterpret_cast<int8_t *>(SrcPtr),       \
                                     Config, Var, Payload);                    \
    case _INT16:                                                               \
      return FN_NAME<int16_t, int16_t>(reinterpret_cast<int16_t *>(DstPtr),    \
                                       reinterpret_cast<int16_t *>(SrcPtr),    \
                                       Config, Var, Payload);                  \
    case _INT32:                                                               \
      return FN_NAME<int32_t, int32_t>(reinterpret_cast<int32_t *>(DstPtr),    \
                                       reinterpret_cast<int32_t *>(SrcPtr),    \
                                       Config, Var, Payload);                  \
    case _INT64:                                                               \
      return FN_NAME<int64_t, int64_t>(reinterpret_cast<int64_t *>(DstPtr),    \
                                       reinterpret_cast<int64_t *>(SrcPtr),    \
                                       Config, Var, Payload);                  \
    case _FLOAT:                                                               \
      return FN_NAME<float, int32_t>(reinterpret_cast<float *>(DstPtr),        \
                                     reinterpret_cast<float *>(SrcPtr),        \
                                     Config, Var, Payload);                    \
    case _DOUBLE:                                                              \
      return FN_NAME<double, int64_t>(reinterpret_cast<double *>(DstPtr),      \
                                      reinterpret_cast<double *>(SrcPtr),      \
                                      Config, Var, Payload);                   \
    case _CUSTOM_TYPE:                                                         \
      return FN_NAME<void, void>(reinterpret_cast<void *>(DstPtr),             \
                                 reinterpret_cast<void *>(SrcPtr), Config,     \
                                 Var, Payload);                                \
    }                                                                          \
    __builtin_unreachable();                                                   \
  }

/// Non-atomically perform `*LHSPtr = *LHSPtr <op> RHS`.
template <typename Ty, typename IntTy>
void reduceValues(Ty *LHS, Ty RHS, RedOpTy Op, ReducerFnTy *ReducerFn) {
  switch (Op) {
  case _ADD:
    *LHS = *LHS + RHS;
    return;
  case _SUB:
    *LHS = *LHS - RHS;
    return;
  case _BIT_OR:
    *(IntTy *)(LHS) = *(IntTy *)(LHS) | utils::convertViaPun<IntTy>(RHS);
    return;
  case _BIT_XOR:
    *(IntTy *)(LHS) = *(IntTy *)(LHS) ^ utils::convertViaPun<IntTy>(RHS);
    return;
  case _LOGIC_OR:
    *LHS =
        utils::convertViaPun<IntTy>(*LHS) || utils::convertViaPun<IntTy>(RHS);
    return;
  case _BIT_AND:
    *(IntTy *)(LHS) = *(IntTy *)(LHS)&utils::convertViaPun<IntTy>(RHS);
    return;
  case _MUL:
    *LHS = *LHS * RHS;
    return;
  case _LOGIC_AND:
    *LHS =
        utils::convertViaPun<IntTy>(*LHS) && utils::convertViaPun<IntTy>(RHS);
    return;
  case _MAX:
    *LHS = *LHS > RHS ? *LHS : RHS;
    return;
  case _MIN:
    *LHS = *LHS > RHS ? RHS : *LHS;
    return;
  case _CUSTOM_OP:
    ReducerFn(LHS, &RHS);
    return;
  }
  __builtin_unreachable();
}

/// Atomically perform `*LHSPtr = *LHSPtr <op> RHS`.
template <typename Ty, typename IntTy>
void reduceValuesAtomically(Ty *LHSPtr, Ty RHS, RedOpTy Op,
                            ReducerFnTy *ReducerFn) {
  switch (Op) {
  case _ADD:
    atomic::add(LHSPtr, RHS, atomic::seq_cst);
    return;
  case _SUB:
    atomic::add(LHSPtr, -RHS, atomic::seq_cst);
    return;
  case _BIT_OR:
    atomic::bit_or((IntTy *)LHSPtr, *((IntTy *)&RHS), atomic::seq_cst);
    return;
  case _BIT_XOR:
    atomic::bit_xor((IntTy *)LHSPtr, *((IntTy *)&RHS), atomic::seq_cst);
    return;
  case _LOGIC_OR:
    break;
  case _BIT_AND:
    atomic::bit_and((IntTy *)LHSPtr, *((IntTy *)&RHS), atomic::seq_cst);
    return;
  case _MUL:
    atomic::mul(LHSPtr, RHS, atomic::seq_cst);
    return;
  case _LOGIC_AND:
    break;
  case _MAX:
    atomic::max(LHSPtr, RHS, atomic::seq_cst);
    return;
  case _MIN:
    atomic::min(LHSPtr, RHS, atomic::seq_cst);
    return;
  case _CUSTOM_OP:
    // The user enabled atomic reduction via a configuration flag. It's the
    // users responsibility to ensure the reducer function will work in this
    // way.
    ReducerFn(LHSPtr, &RHS);
    return;
  }
  __builtin_unreachable();
}

///}

/// WARP methods
///{

/// The threads in \p Mask will reduce the \p BatchSize values in the array
/// pointed to by \p TypedSrcPtr into \p TypedDstPtr.
template <typename Ty, typename IntTy>
void reduceWarp(Ty *__restrict__ TypedDstPtr, Ty *__restrict__ TypedSrcPtr,
                RedConfigTy &Config, RedVarTy &Var, intptr_t Payload) {
  static_assert(sizeof(Ty) == sizeof(IntTy),
                "Type and integer type need to match in size!");

  // Use a local buffer (accumulator) for ease of handling.
  IntTy IntTypedAcc[MaxWarpSize];

  RedOpTy Op = Var.__op;
  ReducerFnTy *ReducerFn = Var.__reducer_fn;
  int32_t BatchSize = Var.__batch_size;

  ASSERT(BatchSize <= MaxWarpSize);
  __builtin_memcpy(&IntTypedAcc[0], TypedSrcPtr, BatchSize * sizeof(Ty));

  // View of the local buffer as typed pointer.
  Ty *TypedAcc = reinterpret_cast<Ty *>(&IntTypedAcc[0]);

  int32_t WarpSize = mapping::getWarpSize();
  int32_t WarpTId = mapping::getThreadIdInWarp();
  int32_t Delta = WarpSize;
  int64_t Mask = Payload ? Payload : lanes::All;

  // Iterate log(WarpSize) times, always dividing the shuffle distance (Delta)
  // by 2. For each distance we work on the entire batch.
  do {
    Delta /= 2;
    for (int32_t i = 0; i < BatchSize; ++i) {
      // First we treat the values as IntTy to do the shuffle.
      IntTy IntTypedShuffleVal =
          utils::shuffleDown(Mask, IntTypedAcc[i], Delta, WarpSize);

      // Now we convert into Ty to do the reduce.
      Ty TypedShuffleVal = utils::convertViaPun<Ty>(IntTypedShuffleVal);
      reduceValues<Ty, IntTy>(&TypedAcc[i], TypedShuffleVal, Op, ReducerFn);
    }
  } while (Delta > 1);

  // Perform the final copy (and potentially reduce) from the local buffer into
  // the destination.
  bool ReduceInto = TypedDstPtr != TypedSrcPtr;
  bool Atomically = Config.__policy & _REDUCE_ATOMICALLY_AFTER_WARP;
  for (int32_t i = WarpTId; i < BatchSize; i += WarpSize) {
    if (Atomically) {
      if (ReduceInto) {
        reduceValuesAtomically<Ty, IntTy>(&TypedDstPtr[i], TypedAcc[i], Op,
                                          ReducerFn);
      } else {
        // Not a supported combination.
        __builtin_trap();
      }
    } else {
      if (ReduceInto) {
        reduceValues<Ty, IntTy>(&TypedDstPtr[i], TypedAcc[i], Op, ReducerFn);
      } else {
        TypedDstPtr[i] = TypedAcc[i];
      }
    }
  }
}

template <>
void reduceWarp<void, void>(void *__restrict__ TypedDstPtr,
                            void *__restrict__ TypedSrcPtr, RedConfigTy &Config,
                            RedVarTy &Var, intptr_t) {
  // TODO: User defined reductions.
}

/// Simple wrapper around the templated reduceWarp to determine the actual
/// and corresponding integer type of the reduction.
TYPE_DEDUCER(reduceWarp);

void reduceWarpWrapper(void *SrcPtr, RedConfigTy &Config) {
  foreachVar(SrcPtr, Config, [&](void *SrcPtr, RedVarTy &Var) {
    reduceWarp(Var.__dst_ptr, SrcPtr, Config, Var);
  });
}

void reduceVarWarp(void *DstPtr, void *SrcPtr, RedConfigTy &Config,
                   RedVarTy &Var) {
  forVarBatched(SrcPtr, Config, Var,
                [](void *SrcPtr, RedConfigTy &Config, RedVarTy &Var,
                   void *DstPtr) { reduceWarp(DstPtr, SrcPtr, Config, Var); });
}
///}

/// TEAM methods
///{

/// Two shared memory buffers, one for batched and one for non-batched
/// execution. TODO: We should either have more in different sizes or teach the
/// optimizer to shrink them based on usage.
[[clang::loader_uninitialized]] static char
    SharedMemScratchpadBatched[MaxWarpAndBlockSize * MaxBatchSize *
                               MaxDataTypeSize]
    __attribute__((aligned(BufferAlignment)));
#pragma omp allocate(SharedMemScratchpadBatched) allocator(omp_pteam_mem_alloc)

[[clang::loader_uninitialized]] static char
    SharedMemScratchpadScalar[MaxWarpAndBlockSize * MaxDataTypeSize]
    __attribute__((aligned(BufferAlignment)));
#pragma omp allocate(SharedMemScratchpadScalar) allocator(omp_pteam_mem_alloc)

///
void reduceTeamImplHelper(void *SrcPtr, RedConfigTy &Config,
                          RedVarTy *Var = nullptr, void *DstPtr = nullptr) {
  int32_t TId = mapping::getThreadIdInBlock();
  int32_t NumParticipants = Config.__num_participants
                                ? Config.__num_participants
                                : mapping::getBlockSize();
  uint64_t Mask = utils::ballotSync(lanes::All, TId < NumParticipants);
  bool ReduceWarpsFirst = Config.__policy & _REDUCE_WARP_FIRST;

  if (ReduceWarpsFirst) {
    ASSERT(Var == 0);
    foreachVar(SrcPtr, Config, [&](void *SrcPtr, RedVarTy &Var) {
      reduceVarWarp(SrcPtr, SrcPtr, Config, Var);
    });
  } else {
    reduceWarp(SrcPtr, SrcPtr, Config, *Var, Mask);
  }

  int32_t WarpSize = mapping::getWarpSize();
  int32_t WarpId = mapping::getWarpId();

#if 0
  if (OMP_UNLIKELY(NumParticipants <= WarpSize)) {
    if (TId == 0) {
      for (int32_t i = 0; i < NumItems; i++)
        reduceValues<Ty, IntTy>(&TypedDstPtr[i], TypedSrcPtr[i], Op, ReducerFn);
    }
    return;
  }
#endif

  // assert(MaxWarpSize >= mapping::getWarpSize());
  // assert(MaxBatchSize >= BatchSize);

  DstPtr = DstPtr ? DstPtr : Var->__dst_ptr;
  int32_t NumItems = Var->__num_items;
  int32_t ItemSize = Var->__item_size;

  int32_t BlockId = mapping::getBlockId();

  int32_t NumWarps = mapping::getNumberOfWarpsInBlock();
  int32_t WarpTId = mapping::getThreadIdInWarp();
  int32_t IsWarpLead = WarpTId == 0;

  Mask = utils::ballotSync(lanes::All, WarpTId < NumWarps);

  auto SyncThreads = [&]() {
    if (mapping::isSPMDMode() || NumParticipants == mapping::getBlockSize())
      synchronize::threadsAligned();
    else
      synchronize::threadsPartial(NumParticipants);
  };

  auto WarpResultsToDst = [&](int32_t BatchSize) {
    void *SharedMem;
    if (BatchSize > 1)
      SharedMem = &SharedMemScratchpadBatched[0];
    else
      SharedMem = &SharedMemScratchpadScalar[0];

    while (NumItems >= BatchSize) {
      if (IsWarpLead) {
        void *SharedMemWarpPtr =
            utils::advance(SharedMem, WarpId * BatchSize * ItemSize);
        __builtin_memcpy(SharedMemWarpPtr, SrcPtr, BatchSize * ItemSize);
      }

      // Wait for all shared memory updates.
      SyncThreads();

      // The first warp performs the final reduction and stores away the result.
      if (WarpId == 0) {

        // Accumulate the shared memory results through shuffles.
        if (WarpTId < NumWarps) {
          void *SharedMemWarpPtr =
              utils::advance(SharedMem, WarpTId * BatchSize * ItemSize);
          reduceWarp(DstPtr, SharedMemWarpPtr, Config, *Var, Mask);
        }
      }

      if (!ReduceWarpsFirst)
        break;

      // Wait for all shared memory reads.
      SyncThreads();

      SrcPtr = utils::advance(SrcPtr, BatchSize * ItemSize);
      DstPtr = utils::advance(DstPtr, BatchSize * ItemSize);
      NumItems -= BatchSize;
    }
  };

  int32_t BatchSize = Var->__batch_size;
  WarpResultsToDst(BatchSize);
  if (ReduceWarpsFirst && BatchSize > 1 && NumItems)
    WarpResultsToDst(1);
}

void reduceTeamImpl(void *SrcPtr, RedConfigTy &Config) {
  // Warps first will reduce all warps in a single call while we otherwise do
  // one warp at a time.
  if (Config.__policy & _REDUCE_WARP_FIRST) {
    reduceTeamImplHelper(SrcPtr, Config);
    return;
  }

  foreachVarBatched(
      SrcPtr, Config,
      [](void *SrcPtr, RedConfigTy &Config, RedVarTy &Var, void *DstPtr) {
        reduceTeamImplHelper(SrcPtr, Config, &Var, DstPtr);
      });
}

void reduceTeam(void *__restrict__ SrcPtr, RedConfigTy &Config) {

  if (!(Config.__policy & _REDUCE_ATOMICALLY_AFTER_WARP)) {
    reduceTeamImpl(SrcPtr, Config);
    return;
  }

  foreachVar(SrcPtr, Config, [&](void *SrcPtr, RedVarTy &Var) {
    reduceVarWarp(Var.__dst_ptr, SrcPtr, Config, Var);
  });
}

///}

/// LEAGUE methods
///{
///

template <typename Ty, typename IntTy>
void reduceLeagueViaAtomicsTyped(Ty *__restrict__ TypedDstPtr,
                                 Ty *__restrict__ TypedSrcPtr,
                                 RedConfigTy &Config, RedVarTy &Var,
                                 intptr_t StartIdx) {
  RedOpTy Op = Var.__op;
  ReducerFnTy *ReducerFn = Var.__reducer_fn;
  int32_t NumItems = Var.__num_items;
  for (int32_t i = StartIdx; i < NumItems; ++i) {
    reduceValuesAtomically<Ty, IntTy>(&TypedDstPtr[i], TypedSrcPtr[i], Op,
                                      ReducerFn);
  }
  for (int32_t i = 0; i < StartIdx; ++i) {
    reduceValuesAtomically<Ty, IntTy>(&TypedDstPtr[i], TypedSrcPtr[i], Op,
                                      ReducerFn);
  }
}

template <>
void reduceLeagueViaAtomicsTyped<void, void>(void *__restrict__ TypedDstPtr,
                                             void *__restrict__ TypedSrcPtr,
                                             RedConfigTy &Config, RedVarTy &Var,
                                             intptr_t StartIdx) {
  // TODO: user defined reductions
}

TYPE_DEDUCER(reduceLeagueViaAtomicsTyped);

void reduceLeagueViaAtomics(void *SrcPtr, RedConfigTy &Config) {

  int32_t TId = mapping::getThreadIdInBlock();
  if (TId)
    return;

  int32_t BlockId = mapping::getBlockId();
  foreachVar(SrcPtr, Config, [&](void *SrcPtr, RedVarTy &Var) {
    int32_t NumItems = Var.__num_items;
    int32_t StartIdx = 0;
    if (Config.__policy & _REDUCE_LEAGUE_VIA_ATOMICS_WITH_OFFSET)
      StartIdx = BlockId % NumItems;
    reduceLeagueViaAtomicsTyped(Var.__dst_ptr, SrcPtr, Config, Var, StartIdx);
  });
}

void reduceLeague(void *__restrict__ SrcPtr, RedConfigTy &Config,
                  RedLeagueConfigTy *__restrict__ LeagueConfig) {

  if (Config.__policy & _REDUCE_ATOMICALLY_AFTER_WARP) {
    return reduceTeam(SrcPtr, Config);
  }

  if (Config.__policy & _REDUCE_TEAM_AS_PART_OF_LEAGUE) {
    reduceTeamImpl(SrcPtr, Config);
  }

  if (Config.__policy & _REDUCE_LEAGUE_VIA_SECOND_KERNEL) {

    int32_t TId = mapping::getThreadIdInBlock();
    if (TId)
      return;

    foreachVar(SrcPtr, Config, [](void *SrcPtr, RedVarTy &Var) {
      int32_t BlockId = mapping::getBlockId();
      int32_t NumItems = Var.__num_items;
      int32_t ItemSize = Var.__item_size;
      void *DstPtr =
          utils::advance(Var.__dst_ptr, ItemSize * NumItems * BlockId);
      uint64_t *TypedDstPtr = reinterpret_cast<uint64_t *>(DstPtr);
      uint64_t *TypedSrcPtr = reinterpret_cast<uint64_t *>(SrcPtr);
      for (int32_t i = 0; i < NumItems; ++i) {
        __builtin_nontemporal_store(TypedSrcPtr[i], &TypedDstPtr[i]);
      }
    });

    return;
  }

  // Default scheme, atomics.
  reduceLeagueViaAtomics(SrcPtr, Config);
}

template <typename Ty, typename IntTy>
void reduceLeagueStandalone(Ty *TypedDstPtr, Ty *TypedSrcPtr,
                            RedConfigTy &Config, RedVarTy &Var,
                            intptr_t Payload) {
  RedLeagueConfigTy *__restrict__ LeagueConfig =
      reinterpret_cast<RedLeagueConfigTy *>(Payload);
  int32_t BlockSize = mapping::getBlockSize(/* IsSPMD */ true);
  int32_t TotalThreads = BlockSize * mapping::getNumberOfBlocks();
  int32_t NumItems = LeagueConfig->__num_items;
  int32_t BatchSize = Var.__batch_size;
  auto Op = Var.__op;
  auto ReducerFn = Var.__reducer_fn;

  int32_t TId = mapping::getThreadIdInBlock();
  if (TId >= NumItems)
    return;

  // Reduce till we have no more input items than threads.
  {
    int32_t It = TId + TotalThreads;
    while (It < NumItems) {
      reduceValues<Ty, IntTy>(&TypedSrcPtr[TId], TypedSrcPtr[It], Op,
                              ReducerFn);
      It += TotalThreads;
    }
  }

  Ty *TypedSharedMem;
  if (BatchSize > 1)
    TypedSharedMem = reinterpret_cast<Ty *>(&SharedMemScratchpadBatched[0]);
  else
    TypedSharedMem = reinterpret_cast<Ty *>(&SharedMemScratchpadScalar[0]);

  int32_t WarpId = mapping::getWarpId();
  reduceWarp<Ty, IntTy>(&TypedSharedMem[WarpId], &TypedSrcPtr[TId], Config, Var,
                        0);

  if (WarpId)
    return;

  synchronize::threadsAligned();

  int32_t NumWarps = mapping::getNumberOfWarpsInBlock();
  uint64_t Mask = utils::ballotSync(lanes::All, TId < NumWarps);
  if (TId < NumWarps)
    reduceWarp<Ty, IntTy>(TypedDstPtr, &TypedSharedMem[TId], Config, Var, Mask);
}

template <>
void reduceLeagueStandalone<void, void>(void *TypedDstPtr, void *TypedSrcPtr,
                                        RedConfigTy &Config, RedVarTy &Var,
                                        intptr_t) {}

/// Simple wrapper around the templated reduceLeagueStandalone to determine the
/// actual and corresponding integer type of the reduction.
TYPE_DEDUCER(reduceLeagueStandalone);

void reduceLeagueStandaloneWrapper(
    void *SrcPtr, RedConfigTy &Config,
    RedLeagueConfigTy *__restrict__ LeagueConfig) {
  for (int32_t i = 0; i < Config.__num_reduction_vars; ++i) {
    RedVarTy &Var = Config.__reduction_vars[0][i];
    reduceLeagueStandalone(Var.__dst_ptr, SrcPtr, Config, Var,
                           reinterpret_cast<intptr_t>(LeagueConfig));
    SrcPtr = utils::advance(SrcPtr, Var.__item_size * Var.__num_items);
  }
}

///}

} // namespace

/// TODO
extern "C" void __llvm_omp_default_reduction_init(
    RedPrivInfoTy *__restrict__ PrivateConfig,
    RedLeagueConfigTy *const __restrict__ LeagueConfig) {

  RedConfigTy *__restrict__ Config = PrivateConfig->__config;

  int32_t __private_copy_size = 0;
  for (int32_t i = 0; i < Config->__num_reduction_vars; ++i)
    __private_copy_size += Config->__reduction_vars[0][i].__item_size *
                           Config->__reduction_vars[0][i].__num_items;

  // Set the pointer to the private default data, potentially after allocating
  // the required memory.
  if (Config->__alloc_config &
      __llvm_omp_reduction_allocation_configuration::_PREALLOCATED) {
    // assert(PrivateConfig->__private_default_data == &PrivateCopy[1]);
  } else if (Config->__allocator_fn) {
    PrivateConfig->__private_default_data =
        Config->__allocator_fn(__private_copy_size);
  } else {
    PrivateConfig->__private_default_data =
        memory::allocGlobal(__private_copy_size, "Privatized reduction memory");
  }

  if (Config->__alloc_config &
      __llvm_omp_reduction_allocation_configuration::_PRE_INITIALIZED)
    return;

  // Initialize the memory with the neutral element.
  char *__restrict__ __private_default_data =
      reinterpret_cast<char *>(PrivateConfig->__private_default_data);

  __llvm_omp_reduction_initializer_fn_ty *__init_fn = nullptr;

  for (int32_t i = 0; i < Config->__num_reduction_vars; ++i) {
    RedVarTy &Var = Config->__reduction_vars[0][i];
    switch (Var.__op) {
    case _ADD:
    case _SUB:
    case _BIT_OR:
    case _BIT_XOR:
    case _LOGIC_OR:
      __init_fn = __llvm_omp_reduction_get_initializer_fn(_VALUE_ZERO,
                                                          Var.__element_type);
      break;
    case _BIT_AND:
      __init_fn = __llvm_omp_reduction_get_initializer_fn(_VALUE_MONE,
                                                          Var.__element_type);
      break;
    case _MUL:
    case _LOGIC_AND:
      __init_fn = __llvm_omp_reduction_get_initializer_fn(_VALUE_ONE,
                                                          Var.__element_type);
      break;
    case _MAX:
      __init_fn = __llvm_omp_reduction_get_initializer_fn(_VALUE_MIN,
                                                          Var.__element_type);
      break;
    case _MIN:
      __init_fn = __llvm_omp_reduction_get_initializer_fn(_VALUE_MAX,
                                                          Var.__element_type);
      break;
    case _CUSTOM_OP:
      __init_fn = Var.__initializer_fn;
      break;
    default:
      __builtin_unreachable();
    };

    ASSERT(__init_fn && "Init function expected.");

    // #pragma clang loop vectorize(assume_safety)
    for (int32_t i = 0; i < Var.__num_items; ++i) {
      __init_fn(&__private_default_data[i * Var.__item_size]);
    }
  }
}

__attribute__((always_inline)) extern "C" void
__llvm_omp_default_reduction_warp(
    __llvm_omp_default_reduction_private_info_ty *__restrict__ PrivateConfig) {
  void *SrcPtr = PrivateConfig->__private_default_data;
  RedConfigTy *__restrict__ Config = PrivateConfig->__config;
  return reduceWarpWrapper(SrcPtr, *Config);
}

__attribute__((always_inline)) extern "C" void
__llvm_omp_default_reduction_team(
    __llvm_omp_default_reduction_private_info_ty *__restrict__ PrivateConfig) {
  void *SrcPtr = PrivateConfig->__private_default_data;
  RedConfigTy *__restrict__ Config = PrivateConfig->__config;
  return reduceTeam(SrcPtr, *Config);
}

__attribute__((always_inline)) extern "C" void
__llvm_omp_default_reduction_league(
    __llvm_omp_default_reduction_private_info_ty *__restrict__ PrivateConfig,
    __llvm_omp_default_reduction_league_configuration_ty
        *const __restrict__ LeagueConfig) {
  void *SrcPtr = PrivateConfig->__private_default_data;
  RedConfigTy *__restrict__ Config = PrivateConfig->__config;
  return reduceLeague(SrcPtr, *Config, LeagueConfig);
}

#if 0
__attribute__((always_inline)) extern "C" void
__llvm_omp_default_reduction_combine(
    RedPrivInfoTy *__restrict__ PrivateConfig,
    RedLeagueConfigTy *const __restrict__ LeagueConfig) {

  void *SrcPtr = PrivateConfig->__private_default_data;
  RedConfigTy *__restrict__ Config = PrivateConfig->__config;

  switch (Config->__level) {
  case _LEAGUE:
    return reduceLeague(SrcPtr, *Config, LeagueConfig);
  case _TEAM:
    return reduceTeam(SrcPtr, *Config);
  case _WARP:
    return reduceWarpWrapper(SrcPtr, *Config);
  }
  __builtin_unreachable();
}
#endif

///

__attribute__((always_inline, flatten)) extern "C" void
__llvm_omp_default_reduction_combine_league_lvl2(
    RedPrivInfoTy *__restrict__ PrivateConfig,
    RedLeagueConfigTy *const __restrict__ LeagueConfig) {
  void *SrcPtr = PrivateConfig->__private_default_data;

  RedConfigTy *__restrict__ Config = PrivateConfig->__config;

  reduceLeagueStandaloneWrapper(SrcPtr, *Config, LeagueConfig);
}

#pragma omp end declare target
