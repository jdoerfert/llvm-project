//===--- Configuration.h - OpenMP device configuration interface -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// API to query the global (constant) device environment.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_CONFIGURATION_H
#define OMPTARGET_CONFIGURATION_H

#include "Types.h"

namespace _OMP {
namespace config {

enum DebugLevel : int32_t { Assertion };

/// Return the number of devices in the system, same number as returned on the
/// host by omp_get_num_devices.
int32_t getNumDevices();

/// Return the user choosen debug level.
int32_t getDebugLevel();

/// Return the amonut of shared memory scratchpad allocated to the main thread
/// in generic mode. Unused in SPMD-mode.
uint32_t getGenericModeMainThreadSharedMemoryStorage();

bool isDebugMode(DebugLevel Level);

} // namespace config
} // namespace _OMP

#endif
