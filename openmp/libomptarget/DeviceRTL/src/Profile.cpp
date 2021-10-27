//===--- Profile.cpp - OpenMP device profile interface ------------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Profile.h"
#include "Configuration.h"
#include "DeviceEnvironment.h"
#include "Mapping.h"
#include "State.h"
#include "Types.h"
#include "ProfileEnvironment.h"
#include "Synchronization.h"

using namespace _OMP;

#pragma omp declare target

/// Pointer to the profiling environment used by this kernel invocation.
static profile::ProfileEnvironmentTy *SHARED(ProfileEnvironment);

void profile::init(bool IsSPMD, kernel::KernelEnvironmentTy &KernelEnv) {
  // Early exit if we are not profiling.
  if (!profile::isInProfileOrAdvisorMode())
    return;

  if (!mapping::getThreadIdInBlock()) {
    // The actual profile environment storage is provided by the kernel info
    // object. Every block will have a pointer to it in shared memory.
    ProfileEnvironment = &KernelEnv.ProfileEnvironment;
  }

  synchronize::threadsAligned();
}

bool profile::isInProfileMode() {
  return config::isConfigurationEnabled(config::EnableProfile);
}
bool profile::isInAdvisorMode() {
  return config::isConfigurationEnabled(config::EnableAdvisor);
}
bool profile::isInProfileOrAdvisorMode() {
  return isInProfileMode() || isInAdvisorMode();
}

/// Profile environment implementations
///
///{

void profile::ProfileValueTy::set(IdentTy *Loc, uint32_t NewV) {
  uintptr_t OldLoc = 0;
  uintptr_t NewLoc = uintptr_t(Loc);
  if (atomic::compareAndSwap((uintptr_t *)(FirstLoc), OldLoc, NewLoc,
                             __ATOMIC_ACQ_REL))
    Value = NewV;
}

void profile::ProfileValueTy::inc(IdentTy *Loc) {
  uintptr_t OldLoc = 0;
  uintptr_t NewLoc = uintptr_t(Loc);
  atomic::compareAndSwap((uintptr_t *)(FirstLoc), OldLoc, NewLoc,
                         __ATOMIC_ACQ_REL);
  atomic::inc(&Value, 1, __ATOMIC_ACQ_REL);
}

///}

/// Event handler implementations
///
///{

void profile::EventHandler<profile::KernelInit>::enter() {}
void profile::EventHandler<profile::KernelInit>::exit() {
  if (profile::isInAdvisorMode()) {
    IdentTy *Loc = &state::getKernelEnvironment().Ident;
    if (mapping::isSPMDMode())
      ProfileEnvironment->NonGenericModeKernels.inc(Loc);
    else
      ProfileEnvironment->NonSPMDModeKernels.inc(Loc);
  }
}
void profile::EventHandler<profile::ParallelRegion>::enter(IdentTy *Loc) {
  bool IsSPMD = mapping::isSPMDMode();
  if (profile::isInAdvisorMode() && !IsSPMD)
    ProfileEnvironment->ParallelRegionsInGenericMode.inc(Loc);
}
void profile::EventHandler<profile::ParallelRegion>::exit() {}

void profile::EventHandler<profile::SharedStackUsage>::singleton(IdentTy *Loc) {
  if (profile::isInAdvisorMode())
    ProfileEnvironment->SharedMemoryStackUsage.inc(Loc);
}
void profile::EventHandler<profile::ThreadStateUsage>::singleton(IdentTy *Loc) {
  if (profile::isInAdvisorMode())
    ProfileEnvironment->ThreadStateUsage.inc(Loc);
}
void profile::EventHandler<profile::PrintCall>::singleton() {
  if (profile::isInAdvisorMode())
    ProfileEnvironment->PrintCalls.inc();
}
void profile::EventHandler<profile::AssertionCall>::singleton(
    const char *File, unsigned Line, const char *Function) {
  if (profile::isInAdvisorMode()) {
    // TODO: Build a IdentTy object from the arguments or pass them along.
    ProfileEnvironment->AssertionCalls.inc();
  }
}
void profile::EventHandler<profile::UserICVUpdate>::singleton(IdentTy *Loc) {
  if (profile::isInAdvisorMode())
    ProfileEnvironment->UserICVUpdates.inc(Loc);
}
void profile::EventHandler<profile::SequentializedParallel>::singleton(
    IdentTy *Loc) {
  if (profile::isInAdvisorMode())
    ProfileEnvironment->SequentializedParallel.inc(Loc);
}
void profile::EventHandler<profile::IdleThreads>::singleton(IdentTy *Loc) {
  if (profile::isInAdvisorMode())
    ProfileEnvironment->IdleThreads.inc(Loc);
}

///}

#pragma omp end declare target
