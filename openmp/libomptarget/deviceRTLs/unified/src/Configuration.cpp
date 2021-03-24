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

using namespace _OMP;

struct DeviceEnvironmentTy {
  int32_t DebugLevel;
//  uint32_t GenericModeMainThreadSharedMemoryStorage;
};

#pragma omp declare target

// TOOD: We want to change the name as soon as the old runtime is gone.
DeviceEnvironmentTy CONSTANT(omptarget_device_environment) __attribute__((used));

int32_t config::getDebugLevel() { return omptarget_device_environment.DebugLevel; }

int32_t config::getNumDevices() { return 0; }

uint32_t config::getGenericModeMainThreadSharedMemoryStorage() {
  //return DeviceEnvironment.GenericModeMainThreadSharedMemoryStorage;
  return state::SharedScratchpadSize / 3;
}

#pragma omp end declare target
