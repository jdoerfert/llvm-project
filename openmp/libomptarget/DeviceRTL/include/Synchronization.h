//===- Synchronization.h - OpenMP synchronization utilities ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_SYNCHRONIZATION_H
#define OMPTARGET_DEVICERTL_SYNCHRONIZATION_H

#include "Types.h"

namespace _OMP {

namespace synchronize {

/// Initialize the synchronization machinery. Must be called by all threads.
void init(bool IsSPMD);

/// Synchronize all threads in a warp identified by \p Mask.
void warp(LaneMaskTy Mask);

/// Synchronize all threads in a block.
void threads();

/// Synchronizing threads is allowed even if they all hit different instances of
/// `synchronize::threads()`. However, `synchronize::threadsAligned()` is more
/// restrictive in that it requires all threads to hit the same instance. The
/// noinline is removed by the openmp-opt pass and helps to preserve the
/// information till then.
///{
#pragma omp begin assumes ext_aligned_barrier

/// Synchronize all threads in a block, they are are reaching the same
/// instruction (hence all threads in the block are "aligned").
__attribute__((noinline)) void threadsAligned();

#pragma omp end assumes
///}

} // namespace synchronize

namespace fence {

/// Memory fence with \p Ordering semantics for the team.
void team(int Ordering);

/// Memory fence with \p Ordering semantics for the contention group.
void kernel(int Ordering);

/// Memory fence with \p Ordering semantics for the system.
void system(int Ordering);

} // namespace fence

namespace atomic {

/// Atomically load \p Addr with \p Ordering semantics.
uint32_t load(uint32_t *Addr, int Ordering);

/// Atomically store \p V to \p Addr with \p Ordering semantics.
void store(uint32_t *Addr, uint32_t V, int Ordering);

/// Atomically increment \p *Addr and wrap at \p V with \p Ordering semantics.
uint32_t inc(uint32_t *Addr, uint32_t V, int Ordering);

/// Atomically perform <op> on \p V and \p *Addr with \p Ordering semantics. The
/// result is stored in \p *Addr;
/// {

#define ATOMIC_FP_OP(TY) TY add(TY *Addr, TY V, int Ordering);

#define ATOMIC_OP(TY)                                                          \
  ATOMIC_FP_OP(TY)                                                             \
  TY bit_or(TY *Addr, TY V, int Ordering);                                     \
  TY bit_and(TY *Addr, TY V, int Ordering);                                    \
  TY bit_xor(TY *Addr, TY V, int Ordering);                                    \
  TY min(TY *Addr, TY V, int Ordering);                                        \
  TY max(TY *Addr, TY V, int Ordering);

ATOMIC_OP(int8_t)
ATOMIC_OP(int16_t)
ATOMIC_OP(int32_t)
ATOMIC_OP(int64_t)
ATOMIC_OP(uint8_t)
ATOMIC_OP(uint16_t)
ATOMIC_OP(uint32_t)
ATOMIC_OP(uint64_t)
ATOMIC_FP_OP(float)
ATOMIC_FP_OP(double)

#undef ATOMIC_FP_OP
#undef ATOMIC_OP

///}

} // namespace atomic

} // namespace _OMP

#endif
