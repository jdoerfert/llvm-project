//===---- device_environment.h - OpenMP GPU device environment ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device environment definition used by the host and device runtime.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_DEVICE_ENVIRONMENT_H_
#define _OMPTARGET_DEVICE_ENVIRONMENT_H_

// deviceRTL uses <stdint> and DeviceRTL uses explicit definitions

struct DeviceEnvironmentTy {
  uint64_t DynamicConfiguration;
  uint32_t NumDevices;
  uint32_t DeviceNum;
  uint32_t DynamicMemSize;
};

#pragma omp declare target
namespace _OMP {
namespace config {};

using namespace config;

/// Helper to get access to default values from the host and device.
namespace defaults {
/// The number of threads that we use by default.
static constexpr uint32_t NumThreads = getGridValue().GV_Default_WG_Size;
static constexpr uint32_t MaxConcurrentKernels = getGridValue().GV_Max_Kernels;
} // namespace defaults

/// A set of configuration bits layed out through variables.
/// If the corresponding bit is set in the configuration bit-field, which is
/// combined from the static and dynamic configuration the user choose, the
/// method `config::isConfigurationEnabled` will return true.
namespace config {
static constexpr uint32_t EnableAssertion = 1U << 0;
static constexpr uint32_t EnableFunctionTracing = 1U << 1;
static constexpr uint32_t EnableProfile = 1U << 2;
static constexpr uint32_t EnableAdvisor = 1U << 3;
} // namespace config
} // namespace _OMP
#pragma omp end declare target

#endif
