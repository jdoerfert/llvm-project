
#include <algorithm>
#include <cstdint>

#include "include/obj_encoding.h"

#define OBJSAN_HIDDEN_API_ATTRS [[gnu::flatten, clang::noinline]]
#define OBJSAN_SMALL_API_ATTRS [[gnu::flatten, clang::always_inline]]
#define OBJSAN_BIG_API_ATTRS [[clang::always_inline]]

using namespace __objsan;

using SmallObjectsTy = BucketSchemeTy</*EncodingNo=*/1,
                                      /*OffsetBits=*/12, /*BucketBits=*/3,
                                      /*RealPtrBits=*/32>;
using LargeObjectsTy = LedgerSchemeTy</*EncodingNo=*/2, /*ObjectBits=*/20>;

static SmallObjectsTy SmallObjects;
static LargeObjectsTy LargeObjects;

struct __attribute__((packed)) ParameterValuePackTy {
  int32_t Size;
  int32_t TypeId;
  char Value[0];
};
struct __attribute__((packed)) AllocationInfoTy {
  char *Name;
  int32_t SizeLHSArgNo, SizeRHSArgNo, AlignArgNo;
  uint8_t InitialValueKind;
  uint32_t InitialValue;
};

#define ENCODING_NO_SWITCH(Function, EncodingNo, Default, ...)                 \
  switch (EncodingNo) {                                                        \
  case 1:                                                                      \
    return SmallObjects.Function(__VA_ARGS__);                                 \
  case 2:                                                                      \
    return LargeObjects.Function(__VA_ARGS__);                                 \
  default:                                                                     \
    return Default;                                                            \
  }

__attribute__((always_inline)) static int64_t
getOffsetBits(char *VPtr, uint64_t OffsetBits) {
  return ((uint64_t)VPtr) & ((1 << OffsetBits) - 1);
}

__attribute__((always_inline)) static bool sizeIsKnown(uint64_t ObjSize) {
  return ObjSize != ~0ULL;
}

static int NumToSanitize = ~0;
static __attribute__((constructor(0))) void __init(void) {
  if (auto *NS = getenv("NUM_SAN"))
    NumToSanitize = atoi(NS);
}

extern "C" {

OBJSAN_BIG_API_ATTRS
char *__objsan_register_object(char *MPtr, uint64_t ObjSize,
                               bool RequiresTemporalCheck) {
  if (NumToSanitize != ~0) [[unlikely]] {
    static int X = 0;
    if (++X > NumToSanitize)
      return MPtr;
  }
  if (ObjSize < SmallObjectsTy::getMaxSize() && !RequiresTemporalCheck)
    return SmallObjects.encode(MPtr, ObjSize);
  return LargeObjects.encode(MPtr, ObjSize);
}

OBJSAN_BIG_API_ATTRS
char *__objsan_post_alloca(char *MPtr, int64_t ObjSize,
                           int8_t RequiresTemporalCheck) {
  return __objsan_register_object(MPtr, ObjSize, RequiresTemporalCheck);
}

OBJSAN_BIG_API_ATTRS
void __objsan_pre_call(int64_t IntrinsicId, char *allocation_info,
                       int32_t num_parameters, char *parameters,
                       int8_t is_definition) {}

OBJSAN_BIG_API_ATTRS
int64_t __objsan_post_call(char *callee, char *callee_name,
                           int64_t intrinsic_id, char *allocation_info,
                           int64_t return_value, int32_t return_value_size,
                           int32_t num_parameters, char *parameters,
                           int8_t is_definition) {
  // TODO: translate return pointers back

  if (!allocation_info)
    return return_value;

  AllocationInfoTy *AI = (AllocationInfoTy *)allocation_info;
  auto MaxSizeArg = 1 + std::max(AI->SizeLHSArgNo, AI->SizeRHSArgNo);
  if ((unsigned)MaxSizeArg > (unsigned)num_parameters)
    __builtin_trap();
  uint64_t ObjSize = 1;
  for (int32_t I = 0; I < std::min(MaxSizeArg, num_parameters); ++I) {
    ParameterValuePackTy *VP = (ParameterValuePackTy *)parameters;
    if (I == AI->SizeLHSArgNo || I == AI->SizeRHSArgNo) {
      if (VP->Size == 4)
        ObjSize *= *(uint32_t *)&VP->Value;
      else if (VP->Size == 8) {
        ObjSize *= *(uint64_t *)&VP->Value;
      } else
        __builtin_trap();
    }
    parameters += sizeof(ParameterValuePackTy) + VP->Size +
                  (VP->Size % 8 ? 8 - VP->Size % 8 : 0);
  }
  char *MPtr = reinterpret_cast<char *>(return_value);
  return reinterpret_cast<int64_t>(
      __objsan_register_object(MPtr, ObjSize,
                               /*RequiresTemporalCheck*/ true));
}

OBJSAN_SMALL_API_ATTRS
void __objsan_free_object(char *VPtr, uint8_t EncodingNo) {
  ENCODING_NO_SWITCH(free, EncodingNo, , VPtr);
}

OBJSAN_HIDDEN_API_ATTRS
uint64_t __objsan_get_object_size(char *VPtr, uint8_t EncodingNo) {
  ENCODING_NO_SWITCH(getSize, EncodingNo, 0, VPtr);
}

OBJSAN_SMALL_API_ATTRS
uint8_t __objsan_get_encoding(char *VPtr) {
  return EncodingCommonTy::getEncodingNo(VPtr);
}

OBJSAN_SMALL_API_ATTRS
char *__objsan_post_base_pointer_info(char *VPtr, uint64_t *SizePtr,
                                      uint8_t *EncodingNoPtr,
                                      uint64_t *NumOffsetBitsPtr) {
  uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(VPtr);
  *EncodingNoPtr = EncodingNo;
  ENCODING_NO_SWITCH(getBasePointerInfo, EncodingNo, 0, VPtr, SizePtr,
                     NumOffsetBitsPtr);
}

OBJSAN_SMALL_API_ATTRS
void *__objsan_post_loop_value_range(int64_t InitialLoopValue,
                                     int64_t FinalLoopValue, int64_t MaxOffset,
                                     char *BaseMPtr, uint64_t ObjSize,
                                     int64_t NumOffsetBits, int8_t EncodingNo,
                                     int8_t IsDefinitivelyExecuted) {
  char *VPtr = (char *)InitialLoopValue;
  int64_t LoopSize = FinalLoopValue - InitialLoopValue;
  if (!EncodingNo || !EncodingCommonTy::checkMagic(VPtr, NumOffsetBits))
      [[unlikely]]
    return nullptr;
  int64_t Offset = getOffsetBits(VPtr, NumOffsetBits);
  if (!sizeIsKnown(ObjSize)) [[unlikely]]
    ObjSize = __objsan_get_object_size(VPtr, EncodingNo);
  if (EncodingCommonTy::checkAndAdjust(
          VPtr, NumOffsetBits, BaseMPtr, LoopSize + MaxOffset, Offset, ObjSize,
          /*FailOnError=*/IsDefinitivelyExecuted)) [[likely]]
    return VPtr;
  return nullptr;
}

OBJSAN_SMALL_API_ATTRS
void *__objsan_pre_load(char *VPtr, char *BaseMPtr, char *LVRI,
                        uint64_t AccessSize, uint64_t ObjSize,
                        int64_t NumOffsetBits, int8_t EncodingNo,
                        int8_t WasChecked) {
  if (!EncodingNo) [[unlikely]]
    return VPtr;
  int64_t Offset = getOffsetBits(VPtr, NumOffsetBits);
  if (WasChecked || (BaseMPtr && LVRI)) [[likely]]
    return BaseMPtr + Offset;
  if (!sizeIsKnown(ObjSize)) [[unlikely]]
    ObjSize = __objsan_get_object_size(VPtr, EncodingNo);
  return EncodingCommonTy::checkAndAdjust(VPtr, NumOffsetBits, BaseMPtr,
                                          AccessSize, Offset, ObjSize,
                                          /*FailOnError=*/true);
}

OBJSAN_SMALL_API_ATTRS
void *__objsan_pre_store(char *VPtr, char *BaseMPtr, char *LVRI,
                         uint64_t AccessSize, uint64_t ObjSize,
                         int64_t NumOffsetBits, int8_t EncodingNo,
                         int8_t WasChecked) {
  if (!EncodingNo) [[unlikely]]
    return VPtr;
  int64_t Offset = getOffsetBits(VPtr, NumOffsetBits);
  if (WasChecked || (BaseMPtr && LVRI)) [[likely]]
    return BaseMPtr + Offset;
  if (!sizeIsKnown(ObjSize)) [[unlikely]]
    ObjSize = __objsan_get_object_size(VPtr, EncodingNo);
  return EncodingCommonTy::checkAndAdjust(VPtr, NumOffsetBits, BaseMPtr,
                                          AccessSize, Offset, ObjSize,
                                          /*FailOnError=*/true);
}
}
