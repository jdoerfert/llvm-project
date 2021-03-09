//===--------- Utils.h - OpenMP device runtime utility functions -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
#ifndef OMPTARGET_UTILS_H
#define OMPTARGET_UTILS_H

#include <stdint.h>

#include "ICVs.h"
#include "Mapping.h"
#include "target_interface.h"

namespace omp {

#pragma omp declare target

template<typename Ty>
struct LazyInitArrayInSharedMem {
  Ty** T;
  uint32_t Arbiter;

  void init() { T = nullptr; Arbiter = 0; initialize(); }

  void initialize() {
    if (!isLeaderInSIMD())
      return;
    if (!__kmpc_atomic_cas(&Arbiter, 0, 1))
      return;
    int NumTy = getNumberOfThreadsAccessingSharedMem();
    T = static_cast<Ty**>(malloc(sizeof(Ty*) * NumTy));
    for (int i = 0; i < NumTy; ++i)
      T[i] = nullptr;
  }

  operator bool() { return Arbiter; }

  Ty**& operator->() { return T; }
  Ty*& operator[](unsigned Idx) { return T[Idx]; }
};

#pragma omp end declare target

} // namespace omp

#endif
