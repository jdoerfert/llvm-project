//===---- KernelEnvironment.h - OpenMP GPU kernel environment ----- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_KERNEL_ENVIRONMENT_H_
#define _OMPTARGET_KERNEL_ENVIRONMENT_H_

// deviceRTL uses <stdint> and DeviceRTL uses explicit definitions

#ifdef OMPTARGET_DEVICE_RUNTIME
#include "Types.h"
#else
#include <SourceInfo.h>
using IdentTy = ident_t;
#endif

#include "ProfileEnvironment.h"

#ifdef OMPTARGET_DEVICE_RUNTIME
namespace _OMP {
#endif

namespace kernel {

struct KernelEnvironmentTy {
  IdentTy Ident;
  profile::ProfileEnvironmentTy ProfileEnvironment;
};
} // namespace kernel

#ifdef OMPTARGET_DEVICE_RUNTIME
}
#endif

#endif
