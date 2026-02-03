//===- f2cpp_rt_types.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
//===----------------------------------------------------------------------===//

#ifndef F2CPP_RT_TYPES_H
#define F2CPP_RT_TYPES_H

#include <stdint.h>

#define INTEGER int32_t
#define REAL float
#define LOGICAL bool

#define INTENT_IN const
#define INTENT_INOUT &
#define INTENT_OUT &

#define PARAMETER const

#define POINTER *
#define TARGET

#endif
