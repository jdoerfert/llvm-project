// LLVM Instrumentor stub runtime

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <stdio.h>

#include "vm_obj.h"

using namespace __ig;

#ifndef NDEBUG
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif

#define IG_API_ATTRS __attribute__((always_inline))

struct ParameterValuePackTy {
  int32_t Size;
  int32_t TypeId;
  char Value[0];
};

struct __attribute__((packed)) BranchConditionValuePackTy {
  int32_t Kind;
  int32_t TypeId;
  int32_t Size;
  char Value[0];
};
struct AllocationInfoTy {
  char *Name;
  int32_t SizeLHSArgNo, SizeRHSArgNo, AlignArgNo;
  uint8_t InitialValueKind;
  uint32_t InitialValue;
};

extern thread_local ObjectManager ThreadOM;

extern "C" {

IG_API_ATTRS
void __ig_pre_function(char *address, char *name, int32_t num_arguments,
                       char *arguments) {
  PRINTF("function pre -- address: %p, name: %s, num_arguments: %i, arguments: "
         "%p\n",
         address, name, num_arguments, arguments);

  if (num_arguments == 2 && !std::strcmp("main", name)) {
    ParameterValuePackTy *VP = (ParameterValuePackTy *)arguments;
    if (VP->TypeId != 12)
      return;

    int ArgC = *((int *)VP->Value);
    arguments += sizeof(ParameterValuePackTy) + VP->Size;
    VP = (ParameterValuePackTy *)arguments;

    if (VP->TypeId != 14)
      return;

    char **NewArgv = (char **)malloc(ArgC * sizeof(char **));
    char **ArgV = *((char ***)&VP->Value);
    for (int i = 0; i < ArgC; ++i) {
      auto *P = ThreadOM.encode(ArgV[i], std::strlen(ArgV[i]));
      NewArgv[i] = P;
    }
    *((char **)&VP->Value) =
        ThreadOM.encode((char *)NewArgv, ArgC * sizeof(char **));
  }
}

IG_API_ATTRS
char *__ig_pre_globals(char *address, char *name, int64_t initial_value,
                       int32_t initial_value_size, int8_t is_constant) {
  PRINTF("globals pre -- address: %p, name: %s, initial_value: %lli, "
         "initial_value_size: %i, is_constant: %i\n",
         address, name, initial_value, initial_value_size, is_constant);
  __builtin_memcpy(address, &initial_value, initial_value_size);
  auto *P = ThreadOM.encode(address, initial_value_size);
  return P;
}

IG_API_ATTRS
char *__ig_pre_globals_ind(char *address, char *name,
                           int64_t *initial_value_ptr,
                           int32_t initial_value_size, int8_t is_constant) {
  PRINTF("globals pre -- address: %p, name: %s, initial_value: %p, "
         "initial_value_size: %i, is_constant: %i\n",
         address, name, (void *)initial_value_ptr, initial_value_size,
         is_constant);
  __builtin_memcpy(address, initial_value_ptr, initial_value_size);
  auto *P = ThreadOM.encode(address, initial_value_size);
  return P;
}

IG_API_ATTRS
void __ig_pre_unreachable() { PRINTF("unreachable pre -- \n"); }

IG_API_ATTRS
void __ig_pre_call(char *callee, char *callee_name, int64_t intrinsic_id,
                   char *allocation_info, int32_t num_parameters,
                   char *parameters) {
  PRINTF("call pre -- callee: %p, callee_name: %s, intrinsic_id: %lli, "
         "allocation_info: %p, num_parameters: %i, parameters: %p\n",
         callee, callee_name, intrinsic_id, allocation_info, num_parameters,
         parameters);
  if (!intrinsic_id)
    return;

  for (int32_t idx = 0; idx < num_parameters; ++idx) {
    ParameterValuePackTy *VP = (ParameterValuePackTy *)parameters;
    if (VP->TypeId == 14) {
      char **VPtr = (char **)&VP->Value;
      auto [P, Size, Offset] = ThreadOM.decode(*VPtr);
      *VPtr = P;
    }
    parameters += sizeof(ParameterValuePackTy) + VP->Size;
  }
}

IG_API_ATTRS
char *__ig_pre_load(char *pointer, char *base_pointer_info, int32_t value_size,
                    int64_t alignment, int32_t value_type_id) {
  PRINTF("load pre -- pointer: %p, base_pointer_info: %p, value_size: %i, "
         "alignment: %lli, "
         "value_type_id: %i\n",
         pointer, base_pointer_info, value_size, alignment, value_type_id);
  return ThreadOM.decodeForAccess(pointer, value_size, value_type_id, READ,
                                  base_pointer_info);
}

IG_API_ATTRS
char *__ig_pre_store(char *pointer, char *base_pointer_info, int64_t value,
                     int32_t value_size, int64_t alignment,
                     int32_t value_type_id) {
  PRINTF("store pre -- pointer: %p, base_pointer_info: %p, value: %lli, "
         "value_size: %i, alignment: %lli, value_type_id: %i\n",
         pointer, base_pointer_info, value, value_size, alignment,
         value_type_id);
  return ThreadOM.decodeForAccess(pointer, value_size, value_type_id, WRITE,
                                  base_pointer_info);
}

IG_API_ATTRS
char *__ig_pre_store_ind(char *pointer, char *base_pointer_info,
                         int64_t *value_ptr, int32_t value_size,
                         int64_t alignment, int32_t value_type_id) {
  PRINTF("store pre -- pointer: %p, base_pointer_info: %p, value_ptr: %p, "
         "value_size: %i, alignment: %lli, value_type_id: %i\n",
         pointer, base_pointer_info, (void *)value_ptr, value_size, alignment,
         value_type_id);
  return ThreadOM.decodeForAccess(pointer, value_size, value_type_id, WRITE,
                                  base_pointer_info);
}

IG_API_ATTRS
int64_t __ig_post_call(char *callee, char *callee_name, int64_t intrinsic_id,
                       char *allocation_info, int64_t return_value,
                       int32_t return_value_size, int32_t num_parameters,
                       char *parameters) {
  PRINTF("call post -- callee: %p, callee_name: %s, intrinsic_id: %lli, "
         "allocation_info: %p, return_value: %lli, return_value_size: %i, "
         "num_parameters: %i, parameters: %p\n",
         callee, callee_name, intrinsic_id, allocation_info, return_value,
         return_value_size, num_parameters, parameters);
  if (allocation_info) {
    AllocationInfoTy *AI = (AllocationInfoTy *)allocation_info;
    auto MaxSizeArg = 1 + std::max(AI->SizeLHSArgNo, AI->SizeRHSArgNo);
    if ((unsigned)MaxSizeArg > (unsigned)num_parameters)
      __builtin_trap();
    int Size = 1;
    for (int32_t idx = 0; idx < std::min(MaxSizeArg, num_parameters); ++idx) {
      ParameterValuePackTy *VP = (ParameterValuePackTy *)parameters;
      if (idx == AI->SizeLHSArgNo || idx == AI->SizeRHSArgNo) {
        int *VPtr = (int *)&VP->Value;
        Size *= *VPtr;
      }
      parameters += sizeof(ParameterValuePackTy) + VP->Size;
    }
    char *VPtr = ThreadOM.encode((char *)return_value, Size);
    return (uint64_t)VPtr;
  }
  return return_value;
}

IG_API_ATTRS
char *__ig_post_alloca(char *address, int64_t size, int64_t alignment) {
  PRINTF("alloca post -- address: %p, size: %lli, alignment: %lli\n", address,
         size, alignment);
  return ThreadOM.encode(address, size);
}

IG_API_ATTRS
char *__ig_post_base_pointer_info(char *base_pointer,
                                  int32_t base_pointer_kind) {
  PRINTF("base_pointer_info post -- base_pointer: %p, base_pointer_kind: %i\n",
         base_pointer, base_pointer_kind);
  return ThreadOM.getBasePtrInfo(base_pointer);
}

IG_API_ATTRS
int8_t __ig_post_icmp(int8_t value, int8_t is_ptr_cmp,
                      int32_t cmp_predicate_kind, int64_t lhs, int64_t rhs) {
  PRINTF("icmp post -- value %i, cmp_predicate_kind: %i, is_ptr_cmp: %i, lhs: "
         "%lli, rhs: %lli\n",
         value, is_ptr_cmp, cmp_predicate_kind, lhs, rhs);
  if (!is_ptr_cmp)
    return value;

  auto [LHSInfo, LHSOffset] = ThreadOM.getPtrInfo((char *)lhs, true);
  auto [RHSInfo, RHSOffset] = ThreadOM.getPtrInfo((char *)rhs, true);
  if (LHSInfo >= 0 || RHSInfo >= 0)
    return ThreadOM.comparePtrs(value, (char *)lhs, LHSInfo, LHSOffset,
                                (char *)rhs, RHSInfo, RHSOffset);

  return value;
}

int64_t __ig_post_ptrtoint(char *pointer, int64_t value) {
  PRINTF("ptrtoint post -- pointer: %p, value: %lli\n", pointer, value);
  return ThreadOM.ptrToInt((char *)pointer, value);
}

void __ig_pre_branch_condition_info(int32_t branch_condition_no,
                                    char *branch_condition_fn,
                                    int32_t num_branch_condition_arguments,
                                    char *arguments) {
  PRINTF("branch_condition_info pre -- branch_condition_no: %i, "
         "branch_condition_fn: %p, num_branch_condition_arguments: %i, "
         "arguments: %p\n",
         branch_condition_no, branch_condition_fn,
         num_branch_condition_arguments, arguments);

  uint32_t MaxSize = 256;
  uint32_t ArgMemSize = 0;
  char *ArgMemPtr = (char *)malloc(MaxSize);

  auto *BCI = new decltype(ThreadOM)::BranchConditionInfo;
  BCI->Fn = ((char (*)(void *))branch_condition_fn);
  BCI->FreeValueInfos.reserve(num_branch_condition_arguments);
  BCI->No = branch_condition_no;

  for (auto I = 0; I < num_branch_condition_arguments; ++I) {
    auto *BCVPtr = (BranchConditionValuePackTy *)arguments;
    if (ArgMemSize + BCVPtr->Size > MaxSize) {
      MaxSize *= 4;
      auto *NewArgMemPtr = (char *)malloc(MaxSize);
      __builtin_memcpy(NewArgMemPtr, ArgMemPtr, ArgMemSize);
      free(ArgMemPtr);
      ArgMemPtr = NewArgMemPtr;
    }
    switch (BCVPtr->Kind) {
    case /*Instruction*/ 0:
      __builtin_memcpy(ArgMemPtr + ArgMemSize, &BCVPtr->Value, BCVPtr->Size);
      ArgMemSize += BCVPtr->Size;
      break;
    case /*Argument*/ 1:
      __builtin_memcpy(ArgMemPtr + ArgMemSize, &BCVPtr->Value, BCVPtr->Size);
      ArgMemSize += BCVPtr->Size;
      break;
    case /*Load*/ 2:
      auto *VPtr = *(char **)BCVPtr->Value;
      BCI->FreeValueInfos.emplace_back(ArgMemSize, BCVPtr->TypeId, BCVPtr->Size,
                                       VPtr);
      ArgMemSize += BCVPtr->Size;
      break;
    }
    arguments += sizeof(BranchConditionValuePackTy) +
                 (BCVPtr->Kind == 2 ? sizeof(void *) : BCVPtr->Size);
  }

  if (BCI->FreeValueInfos.empty()) {
    free(ArgMemPtr);
    delete BCI;
    return;
  }

  BCI->ArgMemPtr = ArgMemPtr;
  for (auto &FVI : BCI->FreeValueInfos)
    ThreadOM.BranchConditions[FVI.VPtr].push_back(BCI);
}
}
