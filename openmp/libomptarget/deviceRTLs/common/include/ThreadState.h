//===--------- ThreadState.h - OpenMP thread state description ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
#ifndef OMPTARGET_THREAD_STATE_H
#define OMPTARGET_THREAD_STATE_H

#include "ICVs.h"
#include "TeamState.h"
#include "Utils.h"
#include "allocator.h"

namespace omp {

#pragma omp declare target

struct ThreadStateTy {

  /// ICVs have preallocated storage in the TeamStateTy which is used if a
  /// thread has not set a custom value. The latter is supported but unlikely.
  /// When it happens we will allocate dynamic memory to hold the values of all
  /// ICVs. Thus, the first time an ICV is set by a thread we will allocate an
  /// ICV struct to hold them all. This is slower than alternatives but allows
  /// users to pay only for what they use.
  ///
  ICVStateTy ICVState;

  ThreadStateTy *PreviousThreadState;

  void init() {
       ICVState = TeamState.ICVState;
       PreviousThreadState = nullptr;
  }

  void init(ThreadStateTy &PreviousTS) {
       ICVState = PreviousTS.ICVState;
       PreviousThreadState = &PreviousTS;
  }

  static void enterDataEnvironment();
  static void exitDataEnvironment();
};

extern LazyInitArrayInSharedMem<ThreadStateTy> EXTERN_SHARED(ThreadStates);

#pragma omp end declare target

} // namespace omp

#endif
