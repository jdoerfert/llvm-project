//===- ConfigurationEnvironment.h - OpenMP GPU config environment - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_CONFIGURATION_ENVIRONMENT_H
#define OMPTARGET_CONFIGURATION_ENVIRONMENT_H

// deviceRTL uses <stdint> and DeviceRTL uses explicit definitions

#include "Environment.h"
#include "OMPConstants.h"

#ifdef OMPTARGET_DEVICE_RUNTIME
namespace _OMP {
#endif

struct ConfigurationEnvironmentTy {
  bool UseGenericStateMachine;
  bool NoSequentializedParallelRegions;
  bool NoIdleThreads;
  bool NoSharedStackUsage;
  bool NoThreadStateUsage;
  uint8_t Unused : 3;
  llvm::omp::OMPTgtExecModeFlags ExecMode;

  /// The number of print slots available (=pre-allocated) in the print
  /// environment.
  uint32_t NumPrintSlots;
};

#ifdef OMPTARGET_DEVICE_RUNTIME
} // namespace _OMP
#endif

#endif
