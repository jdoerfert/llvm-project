//===----- Workshare.cpp -  OpenMP workshare implementation ------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Configuration.h"
#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Types.h"
#include "Utils.h"

using namespace _OMP;

#pragma omp declare target

namespace {

// Helper class to hide the generic loop nest and provide the template argument
// throughout.
template <typename Ty> class StaticLoopChunker {

  // Generic loop nest that handles block and/or thread distribution in the
  // absence of user specified chunk sizes. This implicily picks a block chunk
  // size equal to the number of threads in the block and a thread chunk size
  // equal to one. In contrast to the chunked version we can get away with a
  // single loop in this case
  static void NormalizedLoopNestNoChunk(void (*LoopBody)(Ty, void *), void *Arg,
                                        Ty NumBlocks, Ty BId, Ty NumThreads,
                                        Ty TId, Ty NumIters, bool Break) {
    Ty KernelIteration = NumBlocks * NumThreads;

    // Start index in the normalized space.
    Ty IV = BId * NumThreads + TId;
    ASSERT(IV > 0);

    // Cover the entire iteration space, assumptions in the caller might allow
    // to simplify this loop to a conditional.
    if (IV < NumIters) {
      do {

        // Execute the loop body.
        LoopBody(IV, Arg);

        // Every thread executed one block and thread chunk now.
        IV += KernelIteration;

        if (Break)
          break;

      } while (IV < NumIters);
    }
  }

  // Generic loop nest that handles block and/or thread distribution in the
  // presence of user specified chunk sizes (for at least one of them).
  static void NormalizedLoopNestChunked(void (*LoopBody)(Ty, void *), void *Arg,
                                        Ty BlockChunk, Ty NumBlocks, Ty BId,
                                        Ty ThreadChunk, Ty NumThreads, Ty TId,
                                        Ty NumIters, bool ) {
    Ty KernelIteration = NumBlocks * BlockChunk;

    // Start index in the chunked space.
    Ty IV = BId * BlockChunk + TId;
    ASSERT(IV > 0);

    // Cover the entire iteration space, assumptions in the caller might allow
    // to simplify this loop to a conditional.
    do {

      Ty BlockChunkLeft =
          BlockChunk >= TId * ThreadChunk ? BlockChunk - TId * ThreadChunk : 0;
      Ty ThreadChunkLeft =
          ThreadChunk <= BlockChunkLeft ? ThreadChunk : BlockChunkLeft;

      while (ThreadChunkLeft--) {

        // Given the blocking it's hard to keep track of what to execute.
        if (IV >= NumIters)
          return;

        // Execute the loop body.
        LoopBody(IV, Arg);

        ++IV;
      }

      IV += KernelIteration;

    } while (IV < NumIters);
  }

public:
  // Worksharing `for` loop.
  static void For(IdentTy *Loc, void (*LoopBody)(Ty, void *), void *Arg,
                  Ty NumIters, Ty ThreadChunk) {
    ASSERT(NumIters > 0);
    ASSERT(ThreadChunk >= 0);

    // All threads need to participate but we don't know if we are in a
    // parallel at all or if the user might have used a `num_threads` clause
    // on the parallel and reduced the number compared to the block size.
    // Since nested parallels are possible too we need to get the thread id
    // from the `omp` getter and not the mapping directly.
    Ty NumThreads = omp_get_num_threads();
    Ty TId = omp_get_thread_num();

    // There are no blocks involved here.
    Ty BlockChunk = 0;
    Ty NumBlocks = 1;
    Ty BId = 0;

    // If the tread chunk is not specified we pick a default now.
    if (ThreadChunk == 0)
      ThreadChunk = 1;

    // If we know we have more threads than iterations we can indicate that to
    // avoid an outer loop.
    bool Break = false;
    if (__omp_rtl_assume_threads_oversubscription) {
      ASSERT(NumThreads >= NumIters);
      Break = true;
    }

    if (ThreadChunk != 1)
      NormalizedLoopNestChunked(LoopBody, Arg, BlockChunk, NumBlocks, BId,
                                ThreadChunk, NumThreads, TId, NumIters, Break);
    else
      NormalizedLoopNestNoChunk(LoopBody, Arg, NumBlocks, BId, NumThreads, TId,
                                NumIters, Break);
  }

  static void Distribute(IdentTy *Loc, void (*LoopBody)(Ty, void *), void *Arg,
                         Ty NumIters, Ty BlockChunk) {
    ASSERT(icv::Level == 0);
    ASSERT(icv::ActiveLevel == 0);
    ASSERT(state::ParallelRegionFn == nullptr);
    ASSERT(state::ParallelTeamSize == 1);

    ASSERT(NumIters > 0);
    ASSERT(BlockChunk >= 0);

    // There are no threads involved here.
    Ty ThreadChunk = 0;
    Ty NumThreads = 1;
    Ty TId = 0;
    ASSERT(TId == mapping::getThreadIdInBlock());

    // All teams need to participate.
    Ty NumBlocks = mapping::getNumberOfBlocks();
    Ty BId = mapping::getBlockId();

    // If the block chunk is not specified we pick a default now.
    if (BlockChunk == 0)
      BlockChunk = NumThreads;

    // If we know we have more blocks than iterations we can indicate that to
    // avoid an outer loop.
    bool Break = false;
    if (__omp_rtl_assume_teams_oversubscription) {
      ASSERT(NumBlocks >= NumIters);
      Break = true;
    }

    if (BlockChunk != NumThreads)
      NormalizedLoopNestChunked(LoopBody, Arg, BlockChunk, NumBlocks, BId,
                                ThreadChunk, NumThreads, TId, NumIters, Break);
    else
      NormalizedLoopNestNoChunk(LoopBody, Arg, NumBlocks, BId, NumThreads, TId,
                                NumIters, Break);

    ASSERT(icv::Level == 0);
    ASSERT(icv::ActiveLevel == 0);
    ASSERT(state::ParallelRegionFn == nullptr);
    ASSERT(state::ParallelTeamSize == 1);
  }

  static void DistributeFor(IdentTy *Loc, void (*LoopBody)(Ty, void *),
                            void *Arg, Ty NumIters, Ty BlockChunk,
                            Ty ThreadChunk) {
    ASSERT(icv::Level == 1);
    ASSERT(icv::ActiveLevel == 1);
    ASSERT(state::ParallelRegionFn == nullptr);

    ASSERT(NumIters > 0);
    ASSERT(BlockChunk >= 0);
    ASSERT(ThreadChunk >= 0);

    // All threads need to participate but the user might have used a
    // `num_threads` clause on the parallel and reduced the number compared to
    // the block size.
    Ty NumThreads = omp_get_num_threads();
    Ty TId = mapping::getThreadIdInBlock();

    // All teams need to participate.
    Ty NumBlocks = mapping::getNumberOfBlocks();
    Ty BId = mapping::getBlockId();

    // If the block chunk is not specified we pick a default now.
    if (BlockChunk == 0)
      BlockChunk = NumThreads;

    // If the tread chunk is not specified we pick a default now.
    if (ThreadChunk == 0)
      ThreadChunk = 1;

    // If we know we have more threads (across all blocks) than iterations we
    // can indicate that to avoid an outer loop.
    bool Break = false;
    if (__omp_rtl_assume_teams_oversubscription &
        __omp_rtl_assume_threads_oversubscription) {
      Break = true;
      ASSERT(NumBlocks * NumThreads >= NumIters);
    }

    if (BlockChunk != NumThreads || ThreadChunk != 1)
      NormalizedLoopNestChunked(LoopBody, Arg, BlockChunk, NumBlocks, BId,
                                ThreadChunk, NumThreads, TId, NumIters, Break);
    else
      NormalizedLoopNestNoChunk(LoopBody, Arg, NumBlocks, BId, NumThreads, TId,
                                NumIters, Break);

    ASSERT(icv::Level == 1);
    ASSERT(icv::ActiveLevel == 1);
    ASSERT(state::ParallelRegionFn == nullptr);
  }
};

} // namespace

#define _OMP_LOOP_ENTRY(BW, TY)                                                \
  __attribute__((flatten)) void __kmpc_distribute_for_static_loop##BW(         \
      IdentTy *loc, void (*fn)(TY, void *), void *arg, TY num_iters,           \
      TY block_chunk, TY thread_chunk) {                                       \
    StaticLoopChunker<TY>::DistributeFor(loc, fn, arg, num_iters + 1,          \
                                         block_chunk, thread_chunk);           \
  }                                                                            \
  __attribute__((flatten)) void __kmpc_distribute_static_loop##BW(             \
      IdentTy *loc, void (*fn)(TY, void *), void *arg, TY num_iters,           \
      TY block_chunk) {                                                        \
    StaticLoopChunker<TY>::Distribute(loc, fn, arg, num_iters + 1,             \
                                      block_chunk);                            \
  }                                                                            \
  __attribute__((flatten)) void __kmpc_for_static_loop##BW(                    \
      IdentTy *loc, void (*fn)(TY, void *), void *arg, TY num_iters,           \
      TY thread_chunk) {                                                       \
    StaticLoopChunker<TY>::For(loc, fn, arg, num_iters + 1, thread_chunk);     \
  }

extern "C" {
_OMP_LOOP_ENTRY(_4, int32_t)
_OMP_LOOP_ENTRY(_4u, uint32_t)
_OMP_LOOP_ENTRY(_8, int64_t)
_OMP_LOOP_ENTRY(_8u, uint64_t)
}

#pragma omp end declare target
