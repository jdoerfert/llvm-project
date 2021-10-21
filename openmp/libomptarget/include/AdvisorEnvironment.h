//===---- AdvisorEnvironment.h - OpenMP GPU advisor environment --- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_ADVISOR_ENVIRONMENT_H_
#define _OMPTARGET_ADVISOR_ENVIRONMENT_H_

// deviceRTL uses <stdint> and DeviceRTL uses explicit definitions

#ifdef _OMPTARGET_DEVICE_RUNTIME_
#include "Synchronization.h"
using namespace _OMP;
#else
using IdentTy = ident_t;
#endif

class AdvisorValueTy {
#ifndef _OMPTARGET_DEVICE_RUNTIME_
  char const *const Name = nullptr;
#endif
  uint32_t Value;
  IdentTy *FirstLoc;

public:
#ifndef _OMPTARGET_DEVICE_RUNTIME_
  /// Host implementation
  ///{

  AdvisorValueTy(const char *Name = nullptr)
      : Name(Name), Value(0), FirstLoc(nullptr) {}

  void set(IdentTy *Loc = nullptr, uint32_t V = 1) {
    printf("Cannot set an advisor element on the host!\n");
  }
  void inc(IdentTy *Loc = nullptr) {
    printf("Cannot increment an advisor element on the host!\n");
  }

  uint32_t getValue() { return Value; }
  const char *getName() { return Name; }
  IdentTy *getFirstLoc() { return FirstLoc; }

  ///}

#else

  /// Device implementation
  ///{

  void set(IdentTy *Loc = nullptr, uint32_t NewV = 1) {
    uintptr_t OldLoc = 0;
    uintptr_t NewLoc = uintptr_t(Loc);
    if (atomic::compareAndSwap((uintptr_t *)(FirstLoc), OldLoc, NewLoc,
                               __ATOMIC_ACQ_REL))
      Value = NewV;
  }
  void inc(IdentTy *Loc = nullptr) {
    uintptr_t OldLoc = 0;
    uintptr_t NewLoc = uintptr_t(Loc);
    atomic::compareAndSwap((uintptr_t *)(FirstLoc), OldLoc, NewLoc,
                           __ATOMIC_ACQ_REL);
    atomic::inc(&Value, 1, __ATOMIC_ACQ_REL);
  }

  ///}

#endif
};

struct AdvisorEnvironmentTy {
#ifndef _OMPTARGET_DEVICE_RUNTIME_
#define Element(Name) AdvisorValueTy Name = AdvisorValueTy(#Name)
#else
#define Element(Name) AdvisorValueTy Name;
#endif

  Element(NonSPMDModeKernels);
  Element(NonGenericModeKernels);
  Element(ParallelRegionsInGenericMode);
  Element(ThreadStateUsage);
  Element(SharedMemoryStackUsage);
  Element(UserICVUpdates);
  Element(PrintCalls);
  Element(AssertionCalls);
#undef Element
};

#endif
