// LLVM Instrumentor stub runtime

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <stdio.h>

#include "vm_obj.h"
#include "vm_values.h"

using namespace __ig;

#ifndef NDEBUG
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif

#define IG_API_ATTRS __attribute__((always_inline))

struct __attribute__((packed)) ParameterValuePackTy {
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
struct __attribute__((packed)) AllocationInfoTy {
  char *Name;
  int32_t SizeLHSArgNo, SizeRHSArgNo, AlignArgNo;
  uint8_t InitialValueKind;
  uint32_t InitialValue;
};

extern thread_local ObjectManager ThreadOM;

extern "C" {

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
                   char *parameters, int8_t is_definition) {
  PRINTF("call pre -- callee: %p, callee_name: %s, intrinsic_id: %lli, "
         "allocation_info: %p, num_parameters: %i, parameters: %p, "
         "is_definition: %i\n",
         callee, callee_name, intrinsic_id, allocation_info, num_parameters,
         parameters, is_definition);

  if (is_definition || !parameters)
    return;

  for (int32_t idx = 0; idx < num_parameters; ++idx) {
    ParameterValuePackTy *VP = (ParameterValuePackTy *)parameters;
    if (VP->TypeId == 14) {
      char *VPtr = *(char **)&VP->Value;
      auto [P, Size, Offset] = ThreadOM.decode(VPtr);
      PRINTF("Call arg %p -> %p [%u:%u]\n", VPtr, P, Size, Offset);
      if (!strcmp(callee_name, "__sprintf_chk"))
        PRINTF(" --> '%s'\n", P);
      *(char **)&VP->Value = P;
    } else if (VP->TypeId == 12) {
      PRINTF("Call arg %llu @ %p\n", *(uint64_t *)&VP->Value, &VP->Value);
    }
    parameters += sizeof(ParameterValuePackTy) + VP->Size +
                  (VP->Size % 8 ? 8 - VP->Size % 8 : 0);
  }
}

IG_API_ATTRS
char *__ig_pre_load(char *pointer, char *base_pointer_info, int32_t value_size,
                    int64_t alignment, int32_t value_type_id) {
  PRINTF("load pre -- pointer: %p, base_pointer_info: %p, value_size: %i, "
         "alignment: %lli, "
         "value_type_id: %i\n",
         pointer, base_pointer_info, value_size, alignment, value_type_id);
  ThreadOM.checkBranchConditions(pointer, base_pointer_info);
  bool IsInitialized;
  auto *P = ThreadOM.decodeForAccess(pointer, value_size, value_type_id, READ,
                                     base_pointer_info, IsInitialized);
  return P;
}

IG_API_ATTRS
char *__ig_pre_store(char *pointer, char *base_pointer_info, int64_t value,
                     int32_t value_size, int64_t alignment,
                     int32_t value_type_id) {
  PRINTF("store pre -- pointer: %p, base_pointer_info: %p, value: %lli, "
         "value_size: %i, alignment: %lli, value_type_id: %i\n",
         pointer, base_pointer_info, value, value_size, alignment,
         value_type_id);
  bool IsInitialized;
  auto *MPtr =
      ThreadOM.decodeForAccess(pointer, value_size, value_type_id, WRITE,
                               base_pointer_info, IsInitialized);
  PRINTF("--> %p\n", MPtr);
  return MPtr;
}

IG_API_ATTRS
char *__ig_pre_store_ind(char *pointer, char *base_pointer_info,
                         int64_t *value_ptr, int32_t value_size,
                         int64_t alignment, int32_t value_type_id) {
  PRINTF("store pre -- pointer: %p, base_pointer_info: %p, value_ptr: %p, "
         "value_size: %i, alignment: %lli, value_type_id: %i\n",
         pointer, base_pointer_info, (void *)value_ptr, value_size, alignment,
         value_type_id);
  bool IsInitialized;
  return ThreadOM.decodeForAccess(pointer, value_size, value_type_id, WRITE,
                                  base_pointer_info, IsInitialized);
}

IG_API_ATTRS
int64_t __ig_post_call(char *callee, char *callee_name, int64_t intrinsic_id,
                       char *allocation_info, int64_t return_value,
                       int32_t return_value_size, int32_t num_parameters,
                       char *parameters, int8_t is_definition) {
  PRINTF("call post -- callee: %p, callee_name: %s, intrinsic_id: %lli, "
         "allocation_info: %p, return_value: %lli, return_value_size: %i, "
         "num_parameters: %i, parameters: %p, is_definition: %i\n",
         callee, callee_name, intrinsic_id, allocation_info, return_value,
         return_value_size, num_parameters, parameters, is_definition);
  if (allocation_info) {
    AllocationInfoTy *AI = (AllocationInfoTy *)allocation_info;
    auto MaxSizeArg = 1 + std::max(AI->SizeLHSArgNo, AI->SizeRHSArgNo);
    if ((unsigned)MaxSizeArg > (unsigned)num_parameters)
      __builtin_trap();
    int Size = 1;
    for (int32_t I = 0; I < std::min(MaxSizeArg, num_parameters); ++I) {
      ParameterValuePackTy *VP = (ParameterValuePackTy *)parameters;
      if (I == AI->SizeLHSArgNo || I == AI->SizeRHSArgNo) {
        if (VP->Size == 4)
          Size *= *(uint32_t *)&VP->Value;
        else if (VP->Size == 8) {
          Size *= *(uint64_t *)&VP->Value;
        } else
          __builtin_trap();
      }
      parameters += sizeof(ParameterValuePackTy) + VP->Size +
                    (VP->Size % 8 ? 8 - VP->Size % 8 : 0);
    }
    char *VPtr = ThreadOM.encode((char *)return_value, Size);
    PRINTF("allocation (%s) %p -> %p [%i]\n", callee_name, (void *)return_value,
           VPtr, Size);
    return (uint64_t)VPtr;
  }
  return return_value;
}

IG_API_ATTRS
char *__ig_post_alloca(char *address, int64_t size, int64_t alignment) {
  PRINTF("alloca post -- address: %p, size: %lli, alignment: %lli\n", address,
         size, alignment);
  char *VPtr = ThreadOM.encode(address, size);
  PRINTF("--> %p\n", VPtr);
  return VPtr;
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

char *__ig_decode(char *pointer) {
  return std::get<0>(ThreadOM.decode(pointer));
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
  char *ArgMemPtr;
  bool IsKnown = false;
  auto *BCI = ThreadOM.getBranchCondition(branch_condition_no);
  if (BCI) {
    IsKnown = true;
    ArgMemPtr = BCI->ArgMemPtr;
  } else {
    ArgMemPtr = (char *)new char[MaxSize];
    BCI = new BranchConditionInfo;
    BCI->Fn = ((char (*)(void *))branch_condition_fn);
    BCI->FreeValueInfos.reserve(num_branch_condition_arguments);
    BCI->No = branch_condition_no;
  }

  for (auto I = 0; I < num_branch_condition_arguments; ++I) {
    auto *BCVPtr = (BranchConditionValuePackTy *)arguments;
    if (!IsKnown && ArgMemSize + BCVPtr->Size > MaxSize) {
      MaxSize *= 4;
      auto *NewArgMemPtr = (char *)new char[MaxSize];
      __builtin_memcpy(NewArgMemPtr, ArgMemPtr, ArgMemSize);
      delete[] ArgMemPtr;
      ArgMemPtr = NewArgMemPtr;
    }
    switch (BCVPtr->Kind) {
    case /*instruction*/ 0: {
      arguments += BCVPtr->Size;
      if (BCVPtr->TypeId == 14) {
        char *Value = *((char **)&BCVPtr->Value);
        Value = std::get<0>(ThreadOM.decode(Value));
        PRINTF("Inst %u:: %p -> %p [%i @ %u]\n", ArgMemSize,
               *((char **)&BCVPtr->Value), Value, BCVPtr->Size, ArgMemSize);
        __builtin_memcpy(ArgMemPtr + ArgMemSize, &Value, BCVPtr->Size);
      } else {
        PRINTF("Inst %u:: %i [%i @ %u]\n", ArgMemSize, *((int *)&BCVPtr->Value),
               BCVPtr->Size, ArgMemSize);
        __builtin_memcpy(ArgMemPtr + ArgMemSize, &BCVPtr->Value, BCVPtr->Size);
      }
      ArgMemSize += BCVPtr->Size;
      break;
    }
    case /*argument*/ 1: {
      arguments += BCVPtr->Size;
      if (BCVPtr->TypeId == 14) {
        char *Value = *((char **)&BCVPtr->Value);
        Value = std::get<0>(ThreadOM.decode(Value));
        PRINTF("Arg %u:: %p -> %p [%i @ %u]\n", ArgMemSize,
               *((char **)&BCVPtr->Value), Value, BCVPtr->Size, ArgMemSize);
        __builtin_memcpy(ArgMemPtr + ArgMemSize, &Value, BCVPtr->Size);
      } else {
        PRINTF("Arg %u:: %i [%i @ %u]\n", ArgMemSize, *((int *)&BCVPtr->Value),
               BCVPtr->Size, ArgMemSize);
        __builtin_memcpy(ArgMemPtr + ArgMemSize, &BCVPtr->Value, BCVPtr->Size);
      }
      ArgMemSize += BCVPtr->Size;
      break;
    }
    case /*load*/ 2: {
      arguments += sizeof(void *);
      if (IsKnown)
        break;
      auto *VPtr = *(char **)BCVPtr->Value;
      PRINTF("Load %i: %p\n", BCVPtr->TypeId, VPtr);
      BCI->FreeValueInfos.push_back(
          FreeValueInfo(BCVPtr->TypeId, BCVPtr->Size, VPtr));
      break;
    }
    case /*memcmp*/ 3: {
      arguments += sizeof(void *) * 2 + sizeof(size_t);
      if (IsKnown)
        break;
      auto *CPtr = (char *)&BCVPtr->Value;
      auto *SizePtr = (size_t *)CPtr;
      auto *VPtr = (char **)(CPtr + sizeof(*SizePtr));
      PRINTF("MEMCMP %i: %p %p\n", BCVPtr->TypeId, VPtr[0], VPtr[1]);
      BCI->FreeValueInfos.push_back(FreeValueInfo(BCVPtr->TypeId, BCVPtr->Size,
                                                  VPtr[0], VPtr[1], *SizePtr));
      break;
    }
    case /*strcmp*/ 4: {
      arguments += sizeof(void *) * 2;
      if (IsKnown)
        break;
      auto *VPtr = (char **)&BCVPtr->Value;
      PRINTF("STRCMP %i: %p %p\n", BCVPtr->TypeId, VPtr[0], VPtr[1]);
      BCI->FreeValueInfos.push_back(
          FreeValueInfo(BCVPtr->TypeId, BCVPtr->Size, VPtr[0], VPtr[1], -1));
      break;
    }
    }
    arguments += sizeof(BranchConditionValuePackTy);
  }

  if (IsKnown)
    return;

  if (BCI->FreeValueInfos.empty()) {
    delete[] ArgMemPtr;
    delete BCI;
    return;
  }

  BCI->ArgMemPtr = ArgMemPtr;
  for (auto &FVI : BCI->FreeValueInfos) {
    assert(FVI.VPtr);
    ThreadOM.addBranchCondition(FVI.VPtr, BCI);
    if (FVI.VCmpPtr)
      ThreadOM.addBranchCondition(FVI.VCmpPtr, BCI);
  }
}

int __ig_memcmp(char *s1, char *s2, size_t n) {
  PRINTF("memcmp -- s1: %p, s2: %p, n: %zu\n", s1, s2, n);
  auto *BPI1 = __ig_post_base_pointer_info(s1, 0);
  auto *BPI2 = __ig_post_base_pointer_info(s2, 0);
  // TODO: Workaround until global supported.

  if (BPI1 || BPI2)
    ThreadOM.checkBranchConditions(s1, BPI1, s2, BPI2);

  bool IsInitialized1 = false, IsInitialized2 = false;
  auto *MPtr1 =
      BPI1 ? ThreadOM.decodeForAccess(s1, n, 12, READ, BPI1, IsInitialized1)
           : s1;
  auto *MPtr2 =
      BPI2 ? ThreadOM.decodeForAccess(s2, n, 12, READ, BPI2, IsInitialized2)
           : s2;
  PRINTF("memcmp -- s1: '%s', s2: '%s', n: %zu\n", MPtr1, MPtr2, n);
  return memcmp(MPtr1, MPtr2, n);
}
int __ig_memcmp2(char *s1, char *s2, size_t n) {
  PRINTF("memcmp2 -- s1: %p, s2: %p, n: %zu\n", s1, s2, n);
  auto *MPtr1 = __ig_decode(s1);
  auto *MPtr2 = __ig_decode(s2);
  PRINTF("memcmp2 -- s1: %p, s2: %p, n: %zu\n", MPtr1, MPtr2, n);
  PRINTF("memcmp2 -- s1: '%s', s2: '%s', n: %zu\n", MPtr1, MPtr2, n);
  return memcmp(MPtr1, MPtr2, n);
}
int __ig_strcmp(char *s1, char *s2) {
  PRINTF("strcmp -- s1: %p, s2: %p\n", s1, s2);
  auto *BPI1 = __ig_post_base_pointer_info(s1, 0);
  auto *BPI2 = __ig_post_base_pointer_info(s2, 0);
  // TODO: Workaround until global supported.

  if (BPI1 || BPI2)
    ThreadOM.checkBranchConditions(s1, BPI1, s2, BPI2);

  PRINTF("strcmp -- s1: %p, s2: %p\n", s1, s2);
  do {
    bool IsInitialized1 = false, IsInitialized2 = false;
    auto *MPtr1 =
        BPI1 ? ThreadOM.decodeForAccess(s1, 1, 12, READ, BPI1, IsInitialized1)
             : s1;
    auto *MPtr2 =
        BPI2 ? ThreadOM.decodeForAccess(s2, 1, 12, READ, BPI2, IsInitialized2)
             : s2;
    if (*MPtr1 != *MPtr2)
      return *MPtr1 < *MPtr2;
    if (!*MPtr1)
      return 0;
    ++s1;
    ++s2;
  } while (true);
}

int __ig_strcmp2(char *s1, char *s2) {
  PRINTF("strcmp2 -- s1: %p, s2: %p\n", s1, s2);
  auto *MPtr1 = __ig_decode(s1);
  auto *MPtr2 = __ig_decode(s2);
  PRINTF("strcmp2 -- s1: %p, s2: %p\n", MPtr1, MPtr2);
  PRINTF("strcmp2 -- s1: '%s', s2: '%s'\n", MPtr1, MPtr2);
  return strcmp(MPtr1, MPtr2);
}

int __ig___sprintf_chk(char *s, int flags, size_t slen, const char *format,
                       ...) {
  PRINTF("sprintf_chk -- s: %p, flags: %i, slen: %zu, format %p\n", s, flags,
         slen, format);
  auto *MPtr1 = __ig_decode(s);
  auto *MPtr2 = __ig_decode((char *)format);
  PRINTF("sprintf_chk -- s: %p, flags: %i, slen: %zu, format %p\n", MPtr1,
         flags, slen, MPtr2);
  PRINTF("sprintf_chk -- s1: %p, s2: %p\n", MPtr1, MPtr2);
  PRINTF("sprintf_chk -- s1: '%s', s2: '%s'\n", MPtr1, MPtr2);
  return 0;
}

void __ig_exit(int exit_code) {
  PRINTF("User exit %i\n", exit_code);
  error(exit_code);
}
}
