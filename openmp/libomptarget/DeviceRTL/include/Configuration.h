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

#include "llvm/Frontend/OpenMP/OMPGridValues.h"

namespace _OMP {
namespace config {

/// Return the number of devices in the system, same number as returned on the
/// host by omp_get_num_devices.
uint32_t getNumDevices();

/// Return the number of devices in the system, same number as returned on the
/// host by omp_get_num_devices.
uint32_t getDeviceNum();

/// Return the user choosen debug level.
uint32_t getDebugKind();

/// Return the user choosen configuration options (as bitfield).
uint64_t getConfiguration();

/// Return the amount of dynamic shared memory that was allocated at launch.
uint64_t getDynamicMemorySize();

/// Return true if the configuration option \p Kind is enabled for this run.
bool isConfigurationEnabled(uint64_t Kind);

#pragma omp declare target

/// AMDGCN Implementation
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

constexpr const llvm::omp::GV &getGridValue() {
  return llvm::omp::getAMDGPUGridValues<__AMDGCN_WAVEFRONT_SIZE>();
}

#pragma omp end declare variant
///}

/// NVPTX Implementation
///
///{

#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})
constexpr const llvm::omp::GV &getGridValue() {
  return llvm::omp::NVPTXGridValues;
}

#pragma omp end declare variant
///}

} // namespace config
} // namespace _OMP

#endif
