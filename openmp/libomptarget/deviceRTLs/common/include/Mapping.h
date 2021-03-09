//===--------- Mapping.h - OpenMP device runtime mapping helpers -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
#ifndef OMPTARGET_MAPPING_H
#define OMPTARGET_MAPPING_H

namespace omp {

#pragma omp declare target

/// TODO
bool isMainThreadInGenericMode();

bool isLeaderInSIMD();

unsigned getNumberOfThreadsAccessingSharedMem();
unsigned getThreadIdForSharedMemArrayAccess();

#pragma omp end declare target

} // namespace omp

#endif
