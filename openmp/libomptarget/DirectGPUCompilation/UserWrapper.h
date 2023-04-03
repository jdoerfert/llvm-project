//===------- UserWrapper.h - User code wrapper --------------- C++ --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_DIRECTGPUCOMPILATION_USERWRAPPER_H
#define OPENMP_LIBOMPTARGET_DIRECTGPUCOMPILATION_USERWRAPPER_H

#pragma omp begin declare target device_type(nohost)

int main(int, char *[]) asm("__user_main");

#endif
