//===--- Kernel.cpp - OpenMP device kernel interface -------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the kernel entry points for the device.
//
//===----------------------------------------------------------------------===//

#include "Debug.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"

using namespace _OMP;

namespace {
void inititializeRuntime(bool IsSPMD) {
  synchronize::init(IsSPMD);
  mapping::init(IsSPMD);
  state::init(IsSPMD);
}
} // namespace

#pragma omp declare target

extern "C" {
void __kmpc_kernel_init(int, int16_t) {
  inititializeRuntime(/* IsSPMD */ false);
  // No need to wait since only the main threads will execute user
  // code and workers will run into a barrier right away.

  ASSERT(!mapping::isSPMDMode());
}

void __kmpc_kernel_deinit(int16_t) {
  // Signal the workers to exit the state machine and exit the kernel.
  state::ParallelRegionFn = nullptr;
}

void __kmpc_spmd_kernel_init(int, int16_t) {
  inititializeRuntime(/* IsSPMD */ true);

  state::runAndCheckState(synchronize::threads);

  ASSERT(mapping::isSPMDMode());
}

void __kmpc_spmd_kernel_deinit_v2(int16_t) {
  state::assumeInitialState(/* IsSPMD */ true);
}

int8_t __kmpc_is_spmd_exec_mode() { return mapping::isSPMDMode(); }
}

#pragma omp end declare target
