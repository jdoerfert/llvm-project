
#include <algorithm>
#include <cstdint>
#include <cstring>

#include "include/obj_encoding.h"

#define OBJSAN_SMALL_API_ATTRS [[gnu::flatten, clang::always_inline]]
#define OBJSAN_BIG_API_ATTRS

using namespace __objsan;

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

extern "C" {

OBJSAN_SMALL_API_ATTRS
void __objsan_pre_unreachable() {}

OBJSAN_BIG_API_ATTRS
char *__objsan_register_object(char *MPtr, uint64_t ObjSize,
                               bool RequiresTemporalCheck) {
  if (ObjSize < SmallObjectsTy::getMaxSize() && !RequiresTemporalCheck)
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
char *__objsan_pre_global(char *MPtr, int32_t ObjSize,
                          int8_t RequiresTemporalCheck) {
  return __objsan_register_object(MPtr, ObjSize, RequiresTemporalCheck);
}

OBJSAN_BIG_API_ATTRS
void __objsan_pre_call(int64_t IntrinsicId, int32_t num_parameters,
                       char *parameters, int8_t is_definition) {
  for (int32_t I = 0; I < num_parameters; ++I) {
    ParameterValuePackTy *VP = (ParameterValuePackTy *)parameters;
    if (VP->TypeId == 14) {
      char *VPValuePtr = reinterpret_cast<char *>(&VP->Value);
      char **PtrAddr = reinterpret_cast<char **>(VPValuePtr);
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
    uint32_t Padding = (VP->Size % 8 ? 8 - VP->Size % 8 : 0);
    if (I == AI->SizeLHSArgNo || I == AI->SizeRHSArgNo) {
      char *VPValuePtr = reinterpret_cast<char *>(&VP->Value);
      if (VP->Size == 4)
        ObjSize *= *(uint32_t *)(VPValuePtr + Padding);
      else if (VP->Size == 8) {
        ObjSize *= *(uint64_t *)VPValuePtr;
      } else
        __builtin_trap();
    }
    parameters += sizeof(ParameterValuePackTy) + VP->Size + Padding;
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

OBJSAN_SMALL_API_ATTRS
void __objsan_post_function(int32_t NumAllocas,
                            char *__restrict *__restrict Allocas) {
  for (int32_t I = 0; I < NumAllocas; ++I)
    __objsan_free_alloca(Allocas[I]);
}

OBJSAN_SMALL_API_ATTRS
void __objsan_pre_function(int32_t NumArgs, char *__restrict Arguments) {
  // This is for main only!
  if (NumArgs != 2)
    return;
  ParameterValuePackTy *ArgCVP = (ParameterValuePackTy *)Arguments;
  if (ArgCVP->Size != 4 || ArgCVP->TypeId != 12)
    return;
  char *ArgCVPVPtr = reinterpret_cast<char *>(&ArgCVP->Value);
  int32_t ArgC = *reinterpret_cast<int32_t *>(ArgCVPVPtr + 4);
  Arguments += sizeof(ParameterValuePackTy) + 8;
  ParameterValuePackTy *ArgVVP = (ParameterValuePackTy *)Arguments;
  if (ArgVVP->Size != 8 || ArgVVP->TypeId != 14)
    return;
  auto ArgVSize = sizeof(char *) * ArgC;
  char **ArgV = *reinterpret_cast<char ***>(&ArgVVP->Value);
  for (int32_t I = 0; I < ArgC; ++I) {
    auto StrSize = strlen(ArgV[I]) + 1;
    ArgV[I] = __objsan_register_object(ArgV[I], StrSize,
                                       /*RequiresTemporalCheck=*/true);
  }
  *reinterpret_cast<char **>(&ArgVVP->Value) =
      __objsan_register_object(reinterpret_cast<char *>(ArgV), ArgVSize,
                               /*RequiresTemporalCheck=*/true);
}

OBJSAN_SMALL_API_ATTRS
uint64_t __objsan_get_object_size(char *__restrict VPtr) {
  uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(VPtr);
  if (!EncodingNo)
    return 0;
  ENCODING_NO_SWITCH(getSize, EncodingNo, 0, VPtr)
}

OBJSAN_SMALL_API_ATTRS
uint8_t __objsan_get_encoding(char *__restrict VPtr) {
  return EncodingCommonTy::getEncodingNo(VPtr);
}

OBJSAN_SMALL_API_ATTRS
char *__objsan_post_base_pointer_info(char *__restrict VPtr,
                                      uint64_t *__restrict SizePtr,
                                      uint8_t *__restrict EncodingNoPtr) {
  uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(VPtr);
  *EncodingNoPtr = EncodingNo;
  *SizePtr = 0;
  ENCODING_NO_SWITCH(getBasePointerInfo, EncodingNo, 0, VPtr, SizePtr);
}

OBJSAN_SMALL_API_ATTRS
void *__objsan_post_loop_value_range(int64_t InitialLoopValue,
                                     int64_t FinalLoopValue, int64_t MaxOffset,
                                     char *BaseMPtr, uint64_t ObjSize,
                                     int8_t EncodingNo,
                                     int8_t IsDefinitivelyExecuted) {
  char *VPtr = (char *)InitialLoopValue;
  int64_t LoopSize = FinalLoopValue - InitialLoopValue;
  if (!EncodingNo) [[unlikely]]
    return nullptr;
  int64_t NumOffsetBits;
  if (EncodingNo == 1)
    NumOffsetBits = SmallObjectsTy::NumOffsetBits;
  else
    NumOffsetBits = LargeObjectsTy::NumOffsetBits;
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
                        int8_t EncodingNo, int8_t WasChecked) {
  //  printf("pl %p %p %p %llu %llu %i %i\n", VPtr, BaseMPtr, LVRI, AccessSize,
  //         ObjSize, EncodingNo, WasChecked);
  if (!EncodingNo) [[unlikely]]
    return VPtr;
  int64_t NumOffsetBits;
  if (EncodingNo == 1)
    NumOffsetBits = SmallObjectsTy::NumOffsetBits;
  else
    NumOffsetBits = LargeObjectsTy::NumOffsetBits;
  auto [Offset, Magic] = getOffsetAndMagic(VPtr, NumOffsetBits);
  //  __builtin_prefetch(BaseMPtr + Offset, 0, 1);
  if (WasChecked || (BaseMPtr && LVRI)) [[likely]]
    return BaseMPtr + Offset;
  //  if (!sizeIsKnown(ObjSize)) [[unlikely]]
  //    ObjSize = __objsan_get_object_size(VPtr, EncodingNo);
  return EncodingCommonTy::checkAndAdjust(VPtr, Magic, BaseMPtr, AccessSize,
                                          Offset, ObjSize,
                                          /*FailOnError=*/false);
}

OBJSAN_SMALL_API_ATTRS
int8_t __objsan_check_ptr_load(char *VPtr, int64_t Offset) {
  VPtr += Offset;
  uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(VPtr);
  if (!EncodingNo)
    return 0;
  uint64_t ObjSize;
  auto *BaseMPtr = [&]() -> char * {
    ENCODING_NO_SWITCH(getBasePointerInfo, EncodingNo, 0, VPtr, &ObjSize);
  }();
  if (void *AccMPtr = __objsan_pre_load(VPtr, BaseMPtr, nullptr, sizeof(void *),
                                        ObjSize, EncodingNo, false)) {
    return 1;
  }
  return 0;
}

#define SPEC_LOAD(SIZE)                                                        \
  uint64_t __objsan_pre_spec_load_##SIZE(char *VPtr, int64_t Offset) {         \
    VPtr += Offset;                                                            \
    uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(VPtr);                \
    if (!EncodingNo)                                                           \
      return 0;                                                                \
    uint64_t ObjSize;                                                          \
    auto *BaseMPtr = [&]() -> char * {                                         \
      ENCODING_NO_SWITCH(getBasePointerInfo, EncodingNo, 0, VPtr, &ObjSize);   \
    }();                                                                       \
    if (void *AccMPtr = __objsan_pre_load(VPtr, BaseMPtr, nullptr, SIZE,       \
                                          ObjSize, EncodingNo, false)) {       \
      switch (SIZE) {                                                          \
      case 1:                                                                  \
        return *reinterpret_cast<uint8_t *>(AccMPtr);                          \
      case 2:                                                                  \
        return *reinterpret_cast<uint16_t *>(AccMPtr);                         \
      case 4:                                                                  \
        return *reinterpret_cast<uint32_t *>(AccMPtr);                         \
      case 8:                                                                  \
        void *V = *reinterpret_cast<void **>(AccMPtr);                         \
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
                         int8_t EncodingNo, int8_t WasChecked) {
  //  printf("ps %p %p %p %llu %llu %i %i\n", VPtr, BaseMPtr, LVRI, AccessSize,
  //         ObjSize, EncodingNo, WasChecked);
  if (!EncodingNo) [[unlikely]]
    return VPtr;
  int64_t NumOffsetBits;
  if (EncodingNo == 1)
    NumOffsetBits = SmallObjectsTy::NumOffsetBits;
  else
    NumOffsetBits = LargeObjectsTy::NumOffsetBits;
  auto [Offset, Magic] = getOffsetAndMagic(VPtr, NumOffsetBits);
  //  __builtin_prefetch(BaseMPtr + Offset, 1, 1);
  if (WasChecked || (BaseMPtr && LVRI)) [[likely]]
    return BaseMPtr + Offset;
  //  if (!sizeIsKnown(ObjSize)) [[unlikely]]
  //    ObjSize = __objsan_get_object_size(VPtr, EncodingNo);
  return EncodingCommonTy::checkAndAdjust(VPtr, Magic, BaseMPtr, AccessSize,
                                          Offset, ObjSize,
                                          /*FailOnError=*/false);
}
}
