//===---- KernelEnvironment.h - OpenMP GPU kernel environment ----- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_KERNEL_ENVIRONMENT_H
#define OMPTARGET_KERNEL_ENVIRONMENT_H

// deviceRTL uses <stdint> and DeviceRTL uses explicit definitions

#include "ConfigurationEnvironment.h"
#include "Environment.h"
#include "PrintEnvironment.h"

#ifdef OMPTARGET_DEVICE_RUNTIME
namespace _OMP {
#endif

struct KernelEnvironmentTy {
  IdentTy Ident;

  ConfigurationEnvironmentTy Configuration;

  /// Current indentation level for the function trace. Only accessed by thread
  /// 0.
  uint16_t DebugIndentionLevel;

  /// Pointer to a print environment, basically an array of print slots. The
  /// size of the array is set as part of the constant kernel configuration
  /// environment (see Configuration member).
  PrintEnvironmentTy *PrintEnvironment;
};

#ifdef OMPTARGET_DEVICE_RUNTIME
} // namespace _OMP
#endif

#endif
