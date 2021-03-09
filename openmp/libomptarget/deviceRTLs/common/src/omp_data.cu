//===------------ omp_data.cu - OpenMP GPU objects --------------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the data objects used on the GPU device.
//
//===----------------------------------------------------------------------===//
#pragma omp declare target

#include "common/allocator.h"
#include "common/device_environment.h"
#include "common/omptarget.h"

////////////////////////////////////////////////////////////////////////////////
// global device environment
////////////////////////////////////////////////////////////////////////////////

omptarget_device_environmentTy omptarget_device_environment;

////////////////////////////////////////////////////////////////////////////////
// The team master sets the outlined parallel function in this variable to
// communicate with the workers.  Since it is in shared memory, there is one
// copy of these variables for each kernel, instance, and team.
////////////////////////////////////////////////////////////////////////////////
volatile omptarget_nvptx_WorkFn SHARED(omptarget_nvptx_workFn);

////////////////////////////////////////////////////////////////////////////////
// OpenMP kernel execution parameters
////////////////////////////////////////////////////////////////////////////////
uint32_t SHARED(execution_param);

////////////////////////////////////////////////////////////////////////////////
// Scratchpad for teams reduction.
////////////////////////////////////////////////////////////////////////////////
void *SHARED(ReductionScratchpadPtr);

#pragma omp end declare target
