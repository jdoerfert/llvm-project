//===--------- Misc.cpp - OpenMP device misc interfaces ----------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "Misc.h"

namespace {

/// Fallback implementations are missing to trigger a link time error.
/// Implementations for new devices, including the host, should go into a
/// dedicated begin/end declare variant.
///
///{

double getWTickImpl();

double getWTimeImpl();

///}

/// AMDGCN implementations of the shuffle sync idiom.
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

double getWTickImpl() { return ((double)1E-9); }

double getWTimeImpl() {
  // The intrinsics for measuring time have undocumented frequency
  // This will probably need to be found by measurement on a number of
  // architectures. Until then, return 0, which is very inaccurate as a
  // timer but resolves the undefined symbol at link time.
  return 0;
}

#pragma omp end declare variant
///}

/// NVPTX implementations of the shuffle and shuffle sync idiom.
///
///{
#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

double getWTickImpl() {
  // Timer precision is 1ns
  return ((double)1E-9);
}

double getWTimeImpl() {
  unsigned long long nsecs;
  asm("mov.u64  %0, %%globaltimer;" : "=l"(nsecs));
  return (double)nsecs * getWTickImpl();
}

#pragma omp end declare variant

///}

} // namespace

#pragma omp declare target

int32_t __kmpc_cancellationpoint(IdentTy *Loc, int32_t TId, int32_t CancelVal) {
  return 0;
}

int32_t __kmpc_cancel(IdentTy *Loc, int32_t TId, int32_t CancelVal) {
  return 0;
}

double omp_get_wtick(void) { return getWTickImpl(); }

double omp_get_wtime(void) { return getWTimeImpl(); }

#pragma omp end declare target
