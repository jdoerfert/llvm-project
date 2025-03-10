
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
using FixedObjectsTy =
    FixedLedgerSchemeTy</*EncodingNo=*/3, /*ObjectBits=*/20, 16>;

extern SmallObjectsTy SmallObjects;
extern LargeObjectsTy LargeObjects;
extern FixedObjectsTy FixedObjects;

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
  if (EncodingNo == 2) [[likely]]                                              \
    return LargeObjects.Function(__VA_ARGS__);                                 \
  if (EncodingNo == 1) [[likely]]                                              \
    return SmallObjects.Function(__VA_ARGS__);                                 \
  return Default;

//  case 3:
//    return FixedObjects.Function(__VA_ARGS__);

__attribute__((always_inline)) static std::pair<int64_t, int64_t>
getOffsetAndMagic(char *VPtr, uint64_t OffsetBits) {
  uint64_t V = ((uint64_t)VPtr) &
               ((1ULL << (OffsetBits + EncodingCommonTy::NumMagicBits)) - 1);
  uint64_t Offset = ((uint64_t)VPtr) & ((1ULL << OffsetBits) - 1);
  uint64_t Magic = V >> OffsetBits;
  return {Offset, Magic};
}

//__attribute__((always_inline)) static bool sizeIsKnown(uint64_t ObjSize) {
//  return ObjSize != ~0ULL;
//}

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
      [[likely]]
    if (auto *VPtr = SmallObjects.encode(MPtr, ObjSize)) [[likely]]
      return VPtr;
  //  if (ObjSize == FixedObjectsTy::ObjSize)
  //    return FixedObjects.encode(MPtr, ObjSize);
  return LargeObjects.encode(MPtr, ObjSize);
}

OBJSAN_BIG_API_ATTRS
char *__objsan_post_alloca(char *MPtr, int64_t ObjSize,
                           int8_t RequiresTemporalCheck) {
  return __objsan_register_object(MPtr, ObjSize, RequiresTemporalCheck);
}

OBJSAN_BIG_API_ATTRS
void __objsan_pre_call(int64_t IntrinsicId, int32_t num_parameters,
                       char *parameters, int8_t is_definition) {
  for (int32_t I = 0; I < num_parameters; ++I) {
    ParameterValuePackTy *VP = (ParameterValuePackTy *)parameters;
    if (VP->TypeId == 14) {
      char **PtrAddr = reinterpret_cast<char **>(&VP->Value);
      uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(*PtrAddr);
      if (EncodingNo)
        *PtrAddr = [&]() -> char * {
          ENCODING_NO_SWITCH(decode, EncodingNo, nullptr, *PtrAddr);
        }();
    }
    parameters += sizeof(ParameterValuePackTy) + VP->Size +
                  (VP->Size % 8 ? 8 - VP->Size % 8 : 0);
  }
}

OBJSAN_BIG_API_ATTRS
int64_t __objsan_post_call(char *allocation_info, int64_t return_value,
                           int32_t num_parameters, char *parameters) {
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
void __objsan_free_object(char *__restrict VPtr) {
  uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(VPtr);
  if (!EncodingNo)
    return;
  ENCODING_NO_SWITCH(free, EncodingNo, , VPtr);
}

OBJSAN_SMALL_API_ATTRS
void __objsan_free_alloca(char *__restrict VPtr) {
  ENCODING_NO_SWITCH(free, 2, , VPtr);
}

OBJSAN_HIDDEN_API_ATTRS
uint64_t __objsan_get_object_size(char *__restrict VPtr, uint8_t EncodingNo) {
  auto Size = [&]() -> uint64_t {
    ENCODING_NO_SWITCH(getSize, EncodingNo, 0, VPtr)
  }();
  return Size;
}

OBJSAN_SMALL_API_ATTRS
uint8_t __objsan_get_encoding(char *__restrict VPtr) {
  return EncodingCommonTy::getEncodingNo(VPtr);
}

OBJSAN_SMALL_API_ATTRS
char *__objsan_post_base_pointer_info(char *__restrict VPtr,
                                      uint64_t *__restrict SizePtr,
                                      uint8_t *__restrict EncodingNoPtr,
                                      uint64_t *__restrict NumOffsetBitsPtr) {
  uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(VPtr);
  *EncodingNoPtr = EncodingNo;
  *SizePtr = 0;
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
  if (!EncodingNo) [[unlikely]]
    return nullptr;
  auto [Offset, Magic] = getOffsetAndMagic(VPtr, NumOffsetBits);
  //  for (int I = 0; I < FinalLoopValue; I += 64)
  //    __builtin_prefetch(BaseMPtr + Offset + I, 0, 3);
  //  if (!sizeIsKnown(ObjSize)) [[unlikely]]
  //    ObjSize = __objsan_get_object_size(VPtr, EncodingNo);
  if (EncodingCommonTy::checkAndAdjust(
          VPtr, Magic, BaseMPtr, LoopSize + MaxOffset, Offset, ObjSize,
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
  auto [Offset, Magic] = getOffsetAndMagic(VPtr, NumOffsetBits);
  __builtin_prefetch(BaseMPtr + Offset, 0, 1);
  if (WasChecked || (BaseMPtr && LVRI)) [[likely]]
    return BaseMPtr + Offset;
  //  if (!sizeIsKnown(ObjSize)) [[unlikely]]
  //    ObjSize = __objsan_get_object_size(VPtr, EncodingNo);
  return EncodingCommonTy::checkAndAdjust(VPtr, Magic, BaseMPtr, AccessSize,
                                          Offset, ObjSize,
                                          /*FailOnError=*/false);
}

#define SPEC_LOAD(SIZE)                                                        \
  uint64_t __objsan_pre_spec_load_##SIZE(char *VPtr, int64_t Offset) {         \
    VPtr += Offset;                                                            \
    uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(VPtr);                \
    if (!EncodingNo)                                                           \
      return 0;                                                                \
    uint64_t ObjSize, NumOffsetBits;                                           \
    auto *BaseMPtr = [&]() -> char * {                                         \
      ENCODING_NO_SWITCH(getBasePointerInfo, EncodingNo, 0, VPtr, &ObjSize,    \
                         &NumOffsetBits);                                      \
    }();                                                                       \
    if (void *AccMPtr =                                                        \
            __objsan_pre_load(VPtr, BaseMPtr, nullptr, SIZE, ObjSize,          \
                              NumOffsetBits, EncodingNo, false)) {             \
      switch (SIZE) {                                                          \
      case 1:                                                                  \
        return *reinterpret_cast<uint8_t *>(AccMPtr);                          \
      case 2:                                                                  \
        return *reinterpret_cast<uint16_t *>(AccMPtr);                         \
      case 4:                                                                  \
        return *reinterpret_cast<uint32_t *>(AccMPtr);                         \
      case 8:                                                                  \
        void *V = *reinterpret_cast<void **>(AccMPtr);                         \
        __builtin_prefetch(V, 1, 1);                                           \
        return reinterpret_cast<uint64_t>(V);                                  \
      };                                                                       \
    }                                                                          \
    return 0;                                                                  \
  }

SPEC_LOAD(1)
SPEC_LOAD(2)
SPEC_LOAD(4)
SPEC_LOAD(8)

OBJSAN_SMALL_API_ATTRS
void *__objsan_pre_store(char *VPtr, char *BaseMPtr, char *LVRI,
                         uint64_t AccessSize, uint64_t ObjSize,
                         int64_t NumOffsetBits, int8_t EncodingNo,
                         int8_t WasChecked) {
  if (!EncodingNo) [[unlikely]]
    return VPtr;
  auto [Offset, Magic] = getOffsetAndMagic(VPtr, NumOffsetBits);
  __builtin_prefetch(BaseMPtr + Offset, 1, 1);
  if (WasChecked || (BaseMPtr && LVRI)) [[likely]]
    return BaseMPtr + Offset;
  //  if (!sizeIsKnown(ObjSize)) [[unlikely]]
  //    ObjSize = __objsan_get_object_size(VPtr, EncodingNo);
  return EncodingCommonTy::checkAndAdjust(VPtr, Magic, BaseMPtr, AccessSize,
                                          Offset, ObjSize,
                                          /*FailOnError=*/false);
}
}
