/*===------------- math.h - Alternative math.h header ----------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#pragma omp begin declare variant match(device = {kind(host)})
#include_next <math.h>
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(gpu)})
#include <__clang_openmp_math.h>
#pragma omp end declare variant
