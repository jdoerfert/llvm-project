//===- Configuration.cpp - OpenMP device configuration interface -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the data object of the constant device environment and the
// query API.
//
//===----------------------------------------------------------------------===//

#include "Configuration.h"
#include "State.h"
#include "Types.h"

using namespace _OMP;

#pragma omp declare target

extern uint64_t __omp_rtl_static_configuration;

// TOOD: We want to change the name as soon as the old runtime is gone.
DeviceEnvironmentTy CONSTANT(omptarget_device_environment)
    __attribute__((used));

uint64_t config::getConfiguration() {
  return __omp_rtl_static_configuration & omptarget_device_environment.DynamicConfiguration;
}

uint32_t config::getNumDevices() {
  return omptarget_device_environment.NumDevices;
}

uint32_t config::getDeviceNum() {
  return omptarget_device_environment.DeviceNum;
}

uint64_t config::getDynamicMemorySize() {
  return omptarget_device_environment.DynamicMemSize;
}

bool config::isConfigurationEnabled(uint64_t Kind) {
  return config::getConfiguration() & Kind;
}

#pragma omp end declare target
