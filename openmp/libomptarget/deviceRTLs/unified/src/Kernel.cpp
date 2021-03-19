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

#include "Kernel.h"

#include "State.h"
#include "Mapping.h"
#include "Types.h"
#include "Synchronization.h"

using namespace _OMP;

static void inititializeRuntime(bool IsSPMD) {
  synchronize::init(IsSPMD);
  mapping::init(IsSPMD);
  state::init(IsSPMD);
}

#pragma omp declare target

void __kmpc_kernel_init(int, int16_t) {
  inititializeRuntime(/* IsSPMD */ false);
  // No need to wait since only the main threads will execute user
  // code and workers will run into a barrier right away.
}

void __kmpc_kernel_deinit(int16_t) {
  // Signal the workers to exit the state machine and exit the kernel.
  state::ParallelRegionFn = nullptr;
}

void __kmpc_spmd_kernel_init(int, int16_t) {
  inititializeRuntime(/* IsSPMD */ true);
  // Wait to make sure initialization is complete.
  synchronize::threads();
}

void __kmpc_spmd_kernel_deinit_v2(int16_t) {}

int8_t __kmpc_is_spmd_exec_mode() { return mapping::isSPMDMode(); }

#pragma omp end declare target
