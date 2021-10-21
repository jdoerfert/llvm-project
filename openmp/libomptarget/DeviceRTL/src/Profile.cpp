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
#define _OMPTARGET_DEVICE_RUNTIME_
#include "AdvisorEnvironment.h"
#include "Configuration.h"
#include "Mapping.h"
#include "Synchronization.h"

using namespace _OMP;

#pragma omp declare target

AdvisorEnvironmentTy GLOBAL(__llvm_omp_advisor_environment);

bool profile::isInProfileMode() {
  return config::isConfigurationEnabled(config::Profile);
}
bool profile::isInAdvisorMode() {
  return config::isConfigurationEnabled(config::Advisor);
}
bool profile::isInProfileOrAdvisorMode() {
  return isInProfileMode() || isInAdvisorMode();
}

void profile::EventHandler<profile::KernelInit>::enter(
    IdentTy *Loc, int8_t Mode, bool UseGenericStateMachine) {
  const bool IsSPMD = Mode & OMP_TGT_EXEC_MODE_SPMD;
  if (profile::isInAdvisorMode()) {
    if (IsSPMD)
      __llvm_omp_advisor_environment.NonGenericModeKernels.inc(Loc);
    else
      __llvm_omp_advisor_environment.NonSPMDModeKernels.inc(Loc);
  }
}
void profile::EventHandler<profile::KernelInit>::exit() {}

void profile::EventHandler<profile::ParallelRegion>::enter(IdentTy *Loc) {
  bool IsSPMD = mapping::isSPMDMode();
  if (profile::isInAdvisorMode() && !IsSPMD)
    __llvm_omp_advisor_environment.ParallelRegionsInGenericMode.inc(Loc);
}
void profile::EventHandler<profile::ParallelRegion>::exit() {}

void profile::EventHandler<profile::SharedStackUsage>::singleton(IdentTy *Loc) {
  if (profile::isInAdvisorMode())
    __llvm_omp_advisor_environment.SharedMemoryStackUsage.inc(Loc);
}
void profile::EventHandler<profile::ThreadStateUsage>::singleton(IdentTy *Loc) {
  if (profile::isInAdvisorMode())
    __llvm_omp_advisor_environment.ThreadStateUsage.inc(Loc);
}
void profile::EventHandler<profile::PrintCall>::singleton(IdentTy *Loc) {
  if (profile::isInAdvisorMode())
    __llvm_omp_advisor_environment.PrintCalls.inc(Loc);
}
void profile::EventHandler<profile::AssertionCall>::singleton(IdentTy *Loc) {
  if (profile::isInAdvisorMode())
    __llvm_omp_advisor_environment.AssertionCalls.inc(Loc);
}
void profile::EventHandler<profile::UserICVUpdate>::singleton(IdentTy *Loc) {
  if (profile::isInAdvisorMode())
    __llvm_omp_advisor_environment.UserICVUpdates.inc(Loc);
}

#pragma omp end declare target
