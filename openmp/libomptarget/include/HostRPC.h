//===------- HostRPC.h - Host RPC ---------------------------- C++ --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_INCLUDE_HOSTRPC_H
#define OPENMP_LIBOMPTARGET_INCLUDE_HOSTRPC_H

#ifdef OMPTARGET_DEVICE_RUNTIME
#include "Types.h"
#else
#include <cstdint>
#endif

namespace hostrpc {

/// RPC call identifier. Note: negative value only. Non-negative values are for
/// compiler generated functions.
enum CallId {
  CALLID___kmpc_launch_parallel_51_kernel = -1,
  CALLID_invalid = -2147483648,
};

/// Execution status.
enum ExecutionStatus {
  EXEC_STAT_CREATED = 0,
  EXEC_STAT_DONE = 1,
};

enum Type {
  ARG_LITERAL = 0,
  ARG_POINTER = 1,
};

struct Argument {
  intptr_t Value;
  int64_t ArgType;
};

struct Descriptor {
  // The following member will be used by both host and device.
  int32_t Id;
  struct Argument *Args;
  int64_t NumArgs;
  volatile int64_t Status;
  volatile int64_t ReturnValue;

  // The following members will only be used by device.
  void **ArgInfo;
  void *ArgMap;
};

/// A wrapper of HostRPCDescriptor that will only be used between plugins and
/// libomptarget. It contains the three stdio global variables.
struct DescriptorWrapper {
  Descriptor D;
  void *StdIn = nullptr;
  void *StdOut = nullptr;
  void *StdErr = nullptr;
};

} // namespace hostrpc

#endif
