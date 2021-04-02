//===-------- State.h - OpenMP State & ICV interface ------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_STATE_H
#define OMPTARGET_STATE_H

#include "Types.h"

namespace _OMP {

namespace state {

#pragma omp declare target

// TODO: Expose this via CMAKE.
inline constexpr uint32_t SharedScratchpadSize = 2048;

/// Initialize the state machinery. Must be called by all threads.
void init(bool IsSPMD);

/// TODO
enum ValueKind {
  VK_NThreads,
  VK_Level,
  VK_ActiveLevel,
  VK_MaxActiveLevels,
  VK_RunSched,
  // ---
  VK_RunSchedChunk,
  VK_ParallelRegionFn,
  VK_ParallelTeamSize,
};

/// TODO
void enterDataEnvironment();

/// TODO
void exitDataEnvironment();

/// TODO
struct DateEnvironmentRAII {
  DateEnvironmentRAII() { enterDataEnvironment(); }
  ~DateEnvironmentRAII() { exitDataEnvironment(); }
};

/// TODO
void resetStateForThread();

uint32_t &lookup32(ValueKind VK, bool IsReadonly);
void* &lookupPtr(ValueKind VK, bool IsReadonly);

/// A mookup class without actual state used to provide
/// a nice interface to lookup and update ICV values
/// we can declare in global scope.
template <typename Ty, ValueKind Kind> struct Value {
  operator Ty() { return lookup(/* IsReadonly */ true); }

  Value &operator=(const Ty &Other) {
    set(Other);
    return *this;
  }

  Value &operator++() {
    inc(1);
    return *this;
  }

  Value &operator--() {
    inc(-1);
    return *this;
  }

private:

  Ty &lookup(bool IsReadonly) {
    return lookup32(Kind, IsReadonly);
  }

  Ty &inc(int UpdateVal) {
    return (lookup(/* IsReadonly */ false) += UpdateVal);
  }

  Ty &set(Ty UpdateVal) {
    return (lookup(/* IsReadonly */ false) = UpdateVal);
  }
};

/// A mookup class without actual state used to provide
/// a nice interface to lookup and update ICV values
/// we can declare in global scope.
template <typename Ty, ValueKind Kind> struct PtrValue {
  operator Ty() { return lookup(/* IsReadonly */ true); }

  PtrValue &operator=(const Ty Other) {
    set(Other);
    return *this;
  }

private:

  Ty &lookup(bool IsReadonly) {
    return lookupPtr(Kind, IsReadonly);
  }

  Ty &set(Ty UpdateVal) {
    return (lookup(/* IsReadonly */ false) = UpdateVal);
  }
};

/// TODO
inline state::Value<uint32_t, state::VK_RunSchedChunk> RunSchedChunk;

/// TODO
inline state::Value<uint32_t, state::VK_ParallelTeamSize> ParallelTeamSize;

/// TODO
inline state::PtrValue<ParallelRegionFnTy, state::VK_ParallelRegionFn>
    ParallelRegionFn;
#pragma omp end declare target

void runAndCheckState(void(Func(void)));

void assumeInitialState(bool IsSPMD);

} // namespace state

namespace icv {

#pragma omp declare target

/// TODO
inline state::Value<uint32_t, state::VK_NThreads> NThreads;

/// TODO
inline state::Value<uint32_t, state::VK_Level> Level;

/// The `active-level` describes which of the parallel level counted with the
/// `level-var` is active. There can only be one.
///
/// active-level-var is 1, if ActiveLevelVar is not 0, otherweise it is 0.
inline state::Value<uint32_t, state::VK_ActiveLevel> ActiveLevel;

/// TODO
inline state::Value<uint32_t, state::VK_MaxActiveLevels> MaxActiveLevels;

/// TODO
inline state::Value<uint32_t, state::VK_RunSched> RunSched;

#pragma omp end declare target

} // namespace icv

namespace memory {

/// Alloca \p Size bytes in shared memory, if possible, for \p Reason.
///
/// Note: See the restrictions on __kmpc_alloc_shared for proper usage.
void *allocShared(uint64_t Size, const char *Reason);

/// Free \p Ptr, alloated via allocShared, for \p Reason.
///
/// Note: See the restrictions on __kmpc_free_shared for proper usage.
void freeShared(void *Ptr, const char *Reason);

/// Alloca \p Size bytes in global memory, if possible, for \p Reason.
void *allocGlobal(uint64_t Size, const char *Reason);

/// Free \p Ptr, alloated via allocGlobal, for \p Reason.
void freeGlobal(void *Ptr, const char *Reason);

} // namespace memory

} // namespace _OMP

#endif
