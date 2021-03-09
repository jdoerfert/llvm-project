//===---------- Types.h - OpenMP types ---------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_TYPES_H
#define OMPTARGET_TYPES_H

#include <stdint.h>

typedef enum omp_sched_t {
  omp_sched_static = 1,  /* chunkSize >0 */
  omp_sched_dynamic = 2, /* chunkSize >0 */
  omp_sched_guided = 3,  /* chunkSize >0 */
  omp_sched_auto = 4,    /* no chunkSize */
} omp_sched_t;

#endif
