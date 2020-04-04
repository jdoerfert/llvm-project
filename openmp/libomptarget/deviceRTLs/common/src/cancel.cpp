//===------ cancel.cpp - NVPTX OpenMP cancel interface ------------ c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface to be used in the implementation of OpenMP cancel.
//
//===----------------------------------------------------------------------===//

//#include "common/debug.h"
#include "common/target.h"

struct kmp_Ident;

__DEVICE_SCOPE_BEGIN()

int32_t __kmpc_cancellationpoint(kmp_Ident *loc, int32_t global_tid,
                                 int32_t cancelVal) {
  //PRINT(LD_IO, "call kmpc_cancellationpoint(cancel val %d)\n", (int)cancelVal);
  // disabled
  return 1;
}

int32_t __kmpc_cancel(kmp_Ident *loc, int32_t global_tid, int32_t cancelVal) {
  //PRINT(LD_IO, "call kmpc_cancel(cancel val %d)\n", (int)cancelVal);
  // disabled
  return 0;
}

__DEVICE_SCOPE_END()
