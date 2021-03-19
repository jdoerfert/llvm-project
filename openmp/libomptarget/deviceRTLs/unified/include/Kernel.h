//===--------- Kernel.h - OpenMP device kernel interfaces --------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_KERNEL_H
#define OMPTARGET_KERNEL_H

#include "Types.h"

/// External API
///
///{

extern "C" {

void __kmpc_kernel_init(int, int16_t);

void __kmpc_kernel_deinit(int16_t);

void __kmpc_spmd_kernel_init(int, int16_t);

void __kmpc_spmd_kernel_deinit_v2(int16_t);

int8_t __kmpc_is_spmd_exec_mode();

}

///}

#endif
