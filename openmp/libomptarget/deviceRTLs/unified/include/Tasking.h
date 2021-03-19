//===--------- Tasking.h - OpenMP device tasking functions -------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_TASKING_H
#define OMPTARGET_TASKING_H

#include "Types.h"

/// External shuffle API
///
///{

extern "C" {

 int omp_in_final(void);

 int omp_get_max_task_priority(void);

TaskDescriptorTy *__kmpc_omp_task_alloc(IdentTy *, uint32_t, int32_t,
                                        uint32_t TaskSizeInclPrivateValues,
                                        uint32_t SharedValuesSize,
                                        TaskFnTy TaskFn);

int32_t __kmpc_omp_task(IdentTy *Loc, uint32_t TId,
                        TaskDescriptorTy *TaskDescriptor);

int32_t __kmpc_omp_task_with_deps(IdentTy *Loc, uint32_t TId,
                                  TaskDescriptorTy *TaskDescriptor, int32_t,
                                  void *, int32_t, void *);

void __kmpc_omp_task_begin_if0(IdentTy *Loc, uint32_t TId,
                               TaskDescriptorTy *TaskDescriptor);

void __kmpc_omp_task_complete_if0(IdentTy *Loc, uint32_t TId,
                                  TaskDescriptorTy *TaskDescriptor);

void __kmpc_omp_wait_deps(IdentTy *Loc, uint32_t TId, int32_t, void *, int32_t,
                          void *);

void __kmpc_taskgroup(IdentTy *Loc, uint32_t TId);

void __kmpc_end_taskgroup(IdentTy *Loc, uint32_t TId);

int32_t __kmpc_omp_taskyield(IdentTy *Loc, uint32_t TId, int);

int32_t __kmpc_omp_taskwait(IdentTy *Loc, uint32_t TId);

void __kmpc_taskloop(IdentTy *Loc, uint32_t TId,
                     TaskDescriptorTy *TaskDescriptor, int,
                     uint64_t *LowerBound, uint64_t *UpperBound, int64_t, int,
                     int32_t, uint64_t, void *);

}

///}

#endif
