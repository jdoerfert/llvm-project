//===--------- Reduction.h - OpenMP device reduction functions ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_REDUCTION_H
#define OMPTARGET_REDUCTION_H

#include "Types.h"

/// External API
///
///{

extern "C" {

/// TODO
void __kmpc_nvptx_end_reduce(int32_t TId);

/// TODO
void __kmpc_nvptx_end_reduce_nowait(int32_t TId);

/// TODO
int32_t __kmpc_nvptx_parallel_reduce_nowait_v2(
    IdentTy *Loc, int32_t TId, int32_t num_vars, uint64_t reduce_size,
    void *reduce_data, ShuffleReductFnTy shflFct, InterWarpCopyFnTy cpyFct);

/// TODO
int32_t __kmpc_nvptx_teams_reduce_nowait_v2(
    IdentTy *Loc, int32_t TId, void *GlobalBuffer, int64_t num_of_records,
    void *reduce_data, ShuffleReductFnTy shflFct, InterWarpCopyFnTy cpyFct,
    ListGlobalFnTy lgcpyFct, ListGlobalFnTy lgredFct, ListGlobalFnTy glcpyFct,
    ListGlobalFnTy glredFct);

}

///}

#endif
