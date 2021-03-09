//===--------- TeamState.h - OpenMP team state description -------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
#ifndef OMPTARGET_TEAM_STATE_H
#define OMPTARGET_TEAM_STATE_H

#include <stdint.h>

#include "allocator.h"
#include "ICVs.h"

namespace omp {

#pragma omp declare target

struct TeamStateTy {

  void init() { }

  /// ICVs
  ///
  /// Preallocated storage for ICV values that are used if the threads have not
  /// set a custom default. The latter is supported but unlikely and slow(er).
  ///
  ///{
  ICVStateTy ICVState;
  ///}

  uint16_t ParallelTeamSize;
};

extern TeamStateTy EXTERN_SHARED(TeamState);

#pragma omp end declare target

} // namespace omp

#endif
