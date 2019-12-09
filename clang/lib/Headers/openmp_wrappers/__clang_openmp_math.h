/*===---- __clang_openmp_math.h - OpenMP target math support ---------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#if defined(__NVPTX__) && defined(_OPENMP)

#define __CUDA__

#if defined(__cplusplus)
  #include <__clang_cuda_cmath.h>
#endif

#undef __CUDA__

#endif

