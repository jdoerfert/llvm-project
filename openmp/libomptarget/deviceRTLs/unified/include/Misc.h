//===--------- Misc.h - OpenMP device misc interfaces ------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_MISC_H
#define OMPTARGET_MISC_H

#include "Types.h"

/// External API
///
///{

extern "C" {

/// TODO
double omp_get_wtick(void);

/// TODO
double omp_get_wtime(void);

/// TODO
int32_t __kmpc_cancellationpoint(IdentTy *Loc, int32_t TId, int32_t CancelVal);

/// TODO
int32_t __kmpc_cancel(IdentTy *Loc, int32_t TId, int32_t CancelVal);

}

///}

#endif
