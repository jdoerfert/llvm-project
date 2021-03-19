//===-- Parallelism.h - OpenMP device parallelism utilities ------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_PARALLELISM_H
#define OMPTARGET_PARALLELISM_H

#include "Types.h"

/// External API
///
///{

extern "C" {

/// TODO
void __kmpc_kernel_prepare_parallel(ParallelRegionFnTy WorkFn);

/// TODO
bool __kmpc_kernel_parallel(ParallelRegionFnTy *WorkFn);

/// TODO
void __kmpc_kernel_end_parallel();

/// TODO
void __kmpc_serialized_parallel(IdentTy *Loc, uint32_t);

/// TODO
void __kmpc_end_serialized_parallel(IdentTy *Loc, uint32_t);

/// TODO
void __kmpc_push_proc_bind(IdentTy *Loc, uint32_t TId, int ProcBind);

/// TODO
void __kmpc_push_num_teams(IdentTy *Loc, int32_t TId, int32_t NumTeams,
                           int32_t ThreadLimit);

/// TODO
uint16_t __kmpc_parallel_level(IdentTy *Loc, uint32_t);

/// TODO
void __kmpc_push_num_threads(IdentTy *Loc, int32_t, int32_t NumThreads);
}

///}

#endif
