//===- Environment.h - OpenMP GPU environment helper declarations - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_ENVIRONMENT_H
#define OMPTARGET_ENVIRONMENT_H

// deviceRTL uses <stdint> and DeviceRTL uses explicit definitions

#ifdef OMPTARGET_DEVICE_RUNTIME
#include "Types.h"
#else
#ifdef OMPTARGET_OLD_DEVICE_RUNTIME
#include "interface.h"
using IdentTy = ident_t;
#else
#include "SourceInfo.h"
using IdentTy = ident_t;
#endif
#endif

#endif
