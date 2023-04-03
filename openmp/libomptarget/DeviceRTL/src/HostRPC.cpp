//===------- HostRPC.cpp - Implementation of host RPC ------------- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma omp begin declare target device_type(nohost)

#include "HostRPC.h"

#include "Debug.h"
#include "LibC.h"
#include "Memory.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

#include "llvm/Frontend/OpenMP/OMPDeviceConstants.h"

#ifdef HOSTRPC_DEBUG
#define DEBUG_PREFIX "host-rpc-device"
#define DP(FMT, ...)                                                           \
  { printf("%s --> " FMT, DEBUG_PREFIX, __VA_ARGS__); }
#else
#define DP(FMT, ...)
#endif

using namespace ompx;
using namespace hostrpc;

using ArgType = llvm::omp::OMPTgtHostRPCArgType;

Descriptor *omptarget_hostrpc_descriptor
    __attribute__((used, retain, weak, visibility("protected")));
int32_t *omptarget_hostrpc_futex
    __attribute__((used, retain, weak, visibility("protected")));
char *omptarget_hostrpc_memory_buffer
    __attribute__((used, retain, weak, visibility("protected")));
size_t omptarget_hostrpc_memory_buffer_size
    __attribute__((used, retain, weak, visibility("protected")));

namespace {
size_t HostRPCMemoryBufferCurrentPosition = 0;
constexpr const size_t Alignment = 16;

// FIXME: For now we only allow one thread requesting host RPC.
mutex::TicketLock HostRPCLock;

void *HostRPCMemAlloc(size_t Size) {
  Size = utils::align_up(Size, Alignment);

  if (Size + HostRPCMemoryBufferCurrentPosition <
      omptarget_hostrpc_memory_buffer_size) {
    void *R =
        omptarget_hostrpc_memory_buffer + HostRPCMemoryBufferCurrentPosition;
    atomic::add(&HostRPCMemoryBufferCurrentPosition, Size, atomic::acq_rel);
    return R;
  }

  printf("%s:%d\n", __FILE__, __LINE__);
  __builtin_trap();

  return nullptr;
}

// For now we just reset the buffer.
void HostRPCMemReset() { HostRPCMemoryBufferCurrentPosition = 0; }

static_assert(sizeof(intptr_t) == sizeof(int64_t), "pointer size not match");

struct HostRPCArgInfo {
  void *BasePtr;
  int64_t Type;
  int64_t Size;
  HostRPCArgInfo *Next;
};

struct HostRPCPointerMapEntry {
  void *BasePtr;
  void *MappedBasePtr;
  int64_t Size;
  int64_t Kind;
};

void *getMappedPointer(Descriptor *D, void *BasePtr, int64_t Size,
                       int64_t Offset, int64_t Kind) {
  assert(D->ArgMap && "ArgMap should not be nullptr");

  HostRPCPointerMapEntry *MapTable =
      reinterpret_cast<HostRPCPointerMapEntry *>(D->ArgMap);
  int I = 0;
  for (; I < D->NumArgs && MapTable[I].BasePtr; ++I)
    if (MapTable[I].BasePtr == BasePtr)
      return utils::advance(MapTable[I].MappedBasePtr, Offset);

  MapTable[I].BasePtr = BasePtr;
  MapTable[I].MappedBasePtr = HostRPCMemAlloc(Size);
  MapTable[I].Size = Size;
  MapTable[I].Kind = Kind;

  if (Kind & ArgType::OMP_HOST_RPC_ARG_COPY_TO) {
    __builtin_memcpy(MapTable[I].MappedBasePtr, BasePtr, Size);
    DP("getMappedPointer: copy %ld bytes memory from %p to %p.\n", Size,
       BasePtr, MapTable[I].MappedBasePtr);
  }

  return utils::advance(MapTable[I].MappedBasePtr, Offset);
}

void copybackIfNeeded(Descriptor *D) {
  if (!D->ArgMap)
    return;

  auto *MapTable = reinterpret_cast<HostRPCPointerMapEntry *>(D->ArgMap);
  for (int I = 0; I < D->NumArgs && MapTable[I].BasePtr; ++I)
    if (MapTable[I].Kind & ArgType::OMP_HOST_RPC_ARG_COPY_FROM) {
      __builtin_memcpy(MapTable[I].BasePtr, MapTable[I].MappedBasePtr,
                       MapTable[I].Size);
      DP("copybackIfNeeded: copy %ld bytes memory from %p to %p.\n",
         MapTable[I].Size, MapTable[I].MappedBasePtr, MapTable[I].BasePtr);
    }
}

} // namespace

extern "C" {
__attribute__((noinline, used)) void *
__kmpc_host_rpc_get_desc(int32_t CallId, int32_t NumArgs, void *ArgInfo) {
  assert(omptarget_hostrpc_descriptor && omptarget_hostrpc_futex &&
         "no host rpc pointer");

  DP("device: stdin=%p, stdout=%p, stderr=%p\n", stdin, stdout, stderr);
  DP("get desc for request (id=%d), NumArgs=%d, ArgInfo=%p.\n", CallId, NumArgs,
     ArgInfo);
#ifdef HOSTRPC_DEBUG
  {
    void **AIs = reinterpret_cast<void **>(ArgInfo);
    for (int I = 0; I < NumArgs; ++I)
      DP("ArgInfo[%d]=%p.\n", I, AIs[I]);
  }
#endif

  HostRPCLock.lock();

  // TODO: change it after we support a queue-like data structure.
  Descriptor *D = omptarget_hostrpc_descriptor;

  D->Id = CallId;
  D->ArgInfo = reinterpret_cast<void **>(ArgInfo);
  D->NumArgs = NumArgs;
  D->Status = EXEC_STAT_CREATED;
  D->ReturnValue = 0;
  D->Args =
      reinterpret_cast<Argument *>(HostRPCMemAlloc(sizeof(Argument) * NumArgs));
  D->ArgMap = HostRPCMemAlloc(sizeof(HostRPCPointerMapEntry) * NumArgs);

  assert(!NumArgs || (D->Args && D->ArgMap) && "out of host rpc memory!");

  // Reset the map table.
  auto *ArgMap = reinterpret_cast<HostRPCPointerMapEntry *>(D->ArgMap);
  for (int I = 0; I < NumArgs; ++I)
    ArgMap[I].BasePtr = nullptr;

  return D;
}

__attribute__((noinline, used)) void
__kmpc_host_rpc_add_arg(void *Desc, int64_t ArgVal, int32_t ArgNum) {
  auto *D = reinterpret_cast<Descriptor *>(Desc);
  assert(ArgNum < D->NumArgs && "out-of-range arguments");
  Argument &ArgInDesc = D->Args[ArgNum];

  DP("add arg (no=%d), arg=%lx to request (id=%d).\n", ArgNum, ArgVal, D->Id);

  // This early branch can rule out nullptr and zero scalar value because it
  // doesn't matter whether it is a pointer or scalar value.
  if (ArgVal == 0) {
    ArgInDesc.Value = 0;
    ArgInDesc.ArgType = Type::ARG_LITERAL;

    DP("arg (no=%d) is null, done.\n", ArgNum);

    return;
  }

  void *ArgPtr = reinterpret_cast<void *>(ArgVal);

  if (ArgPtr == stdin || ArgPtr == stdout || ArgPtr == stderr) {
    ArgInDesc.Value = ArgVal;
    ArgInDesc.ArgType = Type::ARG_POINTER;

    DP("arg (no=%d) is stdin/stdout/stderr, done.\n", ArgNum);

    return;
  }

  const auto *AI = reinterpret_cast<HostRPCArgInfo *>(D->ArgInfo[ArgNum]);

  DP("try to find arg (no=%d) from args AI=%p\n", ArgNum, AI);

  if (AI) {
    // Let's first check if Arg is a scalar.
    if (AI->Type == ArgType::OMP_HOST_RPC_ARG_SCALAR) {
      assert(AI->BasePtr == ArgPtr && "invalid scalar argument info");
      assert(AI->Next == nullptr && "invalid scalar argument info");

      ArgInDesc.Value = ArgVal;
      ArgInDesc.ArgType = Type::ARG_LITERAL;

      DP("arg (no=%d) is scalar, done.\n", ArgNum);

      return;
    }

    // Then let's see if it is a literal pointer that we don't need copy.
    auto *P = AI;
    while (P) {
      if (P->Type == ArgType::OMP_HOST_RPC_ARG_PTR && P->BasePtr == ArgPtr) {
        ArgInDesc.Value = ArgVal;
        ArgInDesc.ArgType = Type::ARG_POINTER;

        DP("arg (no=%d) is literal pointer, done.\n", ArgNum);

        return;
      }
      P = P->Next;
    }

    // Next we check if it is within the range of any buffer described in
    // argument info.
    P = AI;
    while (P) {
      if ((P->Type & ArgType::OMP_HOST_RPC_ARG_PTR) && P->Size) {
        if (utils::isInRange(ArgPtr, P->BasePtr, P->Size)) {
          auto Size = P->Size;
          auto Offset = utils::getPtrDiff(ArgPtr, P->BasePtr);

          ArgInDesc.Value = utils::ptrtoint(
              getMappedPointer(D, P->BasePtr, Size, Offset, P->Type));
          ArgInDesc.ArgType = Type::ARG_POINTER;

          DP("found a match for arg (no=%d). done.\n", ArgNum);

          return;
        }
      }
      P = P->Next;
    }
  }

  // Now we can't find a match from argument info, then we assume it is from
  // dynamic allocation.
  memory::MemoryAllocationInfo MAI = memory::getMemoryAllocationInfo(ArgPtr);
  if (MAI.isValid()) {
    auto Size = MAI.Size;
    auto Offset = utils::getPtrDiff(ArgPtr, MAI.BasePtr);

    ArgInDesc.Value = utils::ptrtoint(
        getMappedPointer(D, MAI.BasePtr, Size, Offset,
                         /* Kind */ ArgType::OMP_HOST_RPC_ARG_COPY_TOFROM));
    ArgInDesc.ArgType = Type::ARG_POINTER;

    DP("arg (no=%d) is from malloc. done.\n", ArgNum);

    return;
  }

  printf("request (id=%d) arg (no=%d, val=%p) is unknown. send it to host "
         "directly.\n",
         D->Id, ArgNum, ArgPtr);

  ArgInDesc.Value = ArgVal;
  ArgInDesc.ArgType = Type::ARG_POINTER;
}

__attribute__((noinline, used)) int64_t
__kmpc_host_rpc_send_and_wait(void *Desc) {
  auto *D = reinterpret_cast<Descriptor *>(Desc);
  int32_t Id = D->Id;

  atomic::add(omptarget_hostrpc_futex, 1U, atomic::acq_rel);

  // A system fence is required to make sure futex on the host is also
  // updated if USM is supported.
  fence::system(atomic::seq_cst);

  DP("sent request (id=%d) to host. waiting for finish.\n", Id);

  unsigned NS = 8;

  while (atomic::load(omptarget_hostrpc_futex, atomic::aquire)) {
    asm volatile("nanosleep.u32 %0;" : : "r"(NS));
    // if (NS < 64)
    //   NS *= 2;
    fence::system(atomic::seq_cst);
  }

  DP("finish waiting for request (id=%d).\n", Id);

  int64_t Ret = D->ReturnValue;

  assert(!D->NumArgs || D->ArgMap && "arg map should not be nullptr");
  if (D->ArgMap) {
    DP("copy memory back for request (id=%d).\n", Id);
    copybackIfNeeded(D);
    DP("finish copy memory back for request (id=%d).\n", Id);
  }

  HostRPCMemReset();

  // We can unlock now as we already get all temporary part.
  // TODO: If we have a queue, we don't need this step.
  HostRPCLock.unlock();

  DP("request (id=%d) is done with return code=%lx.\n", Id, Ret);

  return Ret;
}

__attribute__((noinline, used)) void
__kmpc_launch_parallel_51_kernel(const char *name, int32_t gtid,
                                 int32_t if_expr, int32_t num_threads,
                                 void **args, int64_t nargs) {
  constexpr const int64_t NumArgs = 6;
  HostRPCArgInfo ArgInfoArray[NumArgs];
  void *ArgInfo[NumArgs];
  for (unsigned I = 0; I < NumArgs; ++I) {
    ArgInfoArray[I].BasePtr = 0;
    ArgInfoArray[I].Type = ArgType::OMP_HOST_RPC_ARG_SCALAR;
    ArgInfoArray[I].Size = 0;
    ArgInfoArray[I].Next = nullptr;
    ArgInfo[I] = &ArgInfoArray[I];
  }
  auto *D = (Descriptor *)__kmpc_host_rpc_get_desc(0, NumArgs, (void *)ArgInfo);

  // Set up arg info struct.
  ArgInfoArray[0].BasePtr = const_cast<char *>(name);
  ArgInfoArray[0].Type = ArgType::OMP_HOST_RPC_ARG_COPY_TO;
  ArgInfoArray[0].Size = strlen(name) + 1;
  ArgInfoArray[1].BasePtr =
      reinterpret_cast<void *>(static_cast<int64_t>(gtid));
  ArgInfoArray[2].BasePtr =
      reinterpret_cast<void *>(static_cast<int64_t>(if_expr));
  ArgInfoArray[3].BasePtr =
      reinterpret_cast<void *>(static_cast<int64_t>(num_threads));
  ArgInfoArray[5].BasePtr = reinterpret_cast<void *>(nargs);

  // We need to treat args in a little bit different way because nargs might be
  // on the stack.
  ArgInfoArray[4].Size = sizeof(void *) * nargs;
  void *Args = nullptr;
  if (nargs) {
    Args = HostRPCMemAlloc(ArgInfoArray[4].Size);
    __builtin_memcpy(Args, args, ArgInfoArray[4].Size);
  }
  ArgInfoArray[4].BasePtr = Args;

  D->Id = CALLID___kmpc_launch_parallel_51_kernel;

  __kmpc_host_rpc_add_arg(D, reinterpret_cast<int64_t>(name), 0);
  __kmpc_host_rpc_add_arg(D, gtid, 1);
  __kmpc_host_rpc_add_arg(D, if_expr, 2);
  __kmpc_host_rpc_add_arg(D, num_threads, 3);
  __kmpc_host_rpc_add_arg(D, reinterpret_cast<int64_t>(Args), 4);
  __kmpc_host_rpc_add_arg(D, nargs, 5);

  (void)__kmpc_host_rpc_send_and_wait(D);
}
}

#pragma omp end declare target
