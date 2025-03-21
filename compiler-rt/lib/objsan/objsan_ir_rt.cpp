#include "include/common.h"

#include "include/obj_encoding.h"

#define OBJSAN_SMALL_API_ATTRS [[gnu::flatten, clang::always_inline]]
#define OBJSAN_BIG_API_ATTRS [[clang::always_inline]]

#ifdef DEBUG
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif

extern "C" double strtod(const char *, char **);
extern "C" int execvp(const char *__file, char *const *__argv);

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

__attribute__((always_inline)) static std::pair<uint64_t, uint64_t>
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
  PRINTF("%s %p %lli\n", __PRETTY_FUNCTION__, MPtr, ObjSize);
  return __objsan_register_object(MPtr, ObjSize, RequiresTemporalCheck);
}

// OBJSAN_BIG_API_ATTRS
__attribute__((optnone, noinline)) char *
__objsan_pre_global(char *MPtr, int32_t ObjSize, int8_t IsDefinition,
                    int8_t RequiresTemporalCheck) {
  uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(MPtr);
  PRINTF("%s %p %i [%i] (%i)\n", __PRETTY_FUNCTION__, MPtr, ObjSize, EncodingNo,
         IsDefinition);
  if (!IsDefinition)
    return MPtr;
  auto *VPtr = __objsan_register_object(MPtr, ObjSize, RequiresTemporalCheck);
  PRINTF(" -> %p\n", VPtr);
  return VPtr;
}

OBJSAN_BIG_API_ATTRS
void __objsan_pre_call(void *Callee, int64_t IntrinsicId,
                       int32_t num_parameters, char *parameters) {
  PRINTF("%s start: %i\n", __PRETTY_FUNCTION__, num_parameters);
  for (int32_t I = 0; I < num_parameters; ++I) {
    ParameterValuePackTy *VP = (ParameterValuePackTy *)parameters;
    fflush(stdout);
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
  if (Callee == &strtod) {
    assert(0);
  }

  if (Callee == &execvp) {
    ParameterValuePackTy *VP =
        (ParameterValuePackTy *)(parameters - sizeof(ParameterValuePackTy) - 8);
    char *VPValuePtr = reinterpret_cast<char *>(&VP->Value);
    char **PtrAddr = reinterpret_cast<char **>(VPValuePtr);
    int I = 0;
    while (PtrAddr[I])
      ++I;
    char **FakeEnv = (char **)malloc(I * sizeof(char *));
    I = 0;
    while (PtrAddr[I]) {
      FakeEnv[I] = LargeObjects.decode(PtrAddr[I]);
      ++I;
    }
    *PtrAddr = (char *)FakeEnv;
  }
  PRINTF("%s done\n", __PRETTY_FUNCTION__);
}

OBJSAN_BIG_API_ATTRS
char *__objsan_post_call(char *MPtr, uint64_t ObjSize,
                         int8_t RequiresTemporalCheck) {
  return __objsan_register_object(MPtr, ObjSize, RequiresTemporalCheck);
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
  // PRINTF("%s start\n", __PRETTY_FUNCTION__);
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
  PRINTF("argc %i, %p\n", ArgC, ArgV);
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

OBJSAN_SMALL_API_ATTRS char *
__objsan_post_base_pointer_info(char *__restrict VPtr, uint64_t *SizePtr,
                                uint8_t *EncNoPtr) {
  uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(VPtr);
  *EncNoPtr = EncodingNo;
  *SizePtr = 0;
  if (!EncodingNo)
    return VPtr;
  char *MPtr = [&]() {
    ENCODING_NO_SWITCH(getBasePointerInfo, EncodingNo, VPtr, VPtr, SizePtr);
  }();
  PRINTF("%s P: %p/%p Enc: %i OS: %llu\n", __PRETTY_FUNCTION__, VPtr, MPtr,
         EncodingNo, *SizePtr);
  return MPtr;
}

OBJSAN_SMALL_API_ATTRS
void *__objsan_get_mptr(char *__restrict VPtr, char *__restrict BaseMPtr,
                        uint8_t EncodingNo) {
  [[maybe_unused]] uint8_t EncodingNo2 = EncodingCommonTy::getEncodingNo(VPtr);
  PRINTF("%s %p %p %i %i start\n", __PRETTY_FUNCTION__, VPtr, BaseMPtr,
         EncodingNo, EncodingNo2);
  if (!EncodingNo) [[unlikely]]
    return VPtr;
  int64_t NumOffsetBits;
  if (EncodingNo == 1)
    NumOffsetBits = SmallObjectsTy::NumOffsetBits;
  else
    NumOffsetBits = LargeObjectsTy::NumOffsetBits;
  auto [Offset, Magic] = getOffsetAndMagic(VPtr, NumOffsetBits);
  // TODO: Checm magic
  return BaseMPtr + Offset;
}

OBJSAN_SMALL_API_ATTRS
char *__objsan_post_loop_value_range(char *BeginMPtr, char *EndMPtr,
                                     int64_t MaxOffset, char *BaseVPtr,
                                     char *BaseMPtr, uint64_t ObjSize,
                                     int8_t EncodingNo,
                                     int8_t IsDefinitivelyExecuted) {
  PRINTF("%s start\n", __PRETTY_FUNCTION__);
  int64_t LoopSize = EndMPtr - BeginMPtr;
  if (EncodingNo && !EncodingCommonTy::check(
                        BeginMPtr, BaseMPtr, LoopSize + MaxOffset, ObjSize,
                        /*FailOnError=*/IsDefinitivelyExecuted)) [[unlikely]] {
    FPRINTF("r bad %p-%p %p %llu %llu %i\n", BeginMPtr, EndMPtr, BaseMPtr,
            LoopSize, ObjSize, EncodingNo);
    return nullptr;
  }
  return /* not null */ (char *)(0x1);
}

OBJSAN_SMALL_API_ATTRS
void __objsan_pre_ranged_access(char *MPtr, char *BaseMPtr, int64_t AccessSize,
                                uint64_t ObjSize, int8_t EncodingNo) {
  PRINTF("%s start P: %p B: %p AS: %llu OS: %llu Enc: %i\n",
         __PRETTY_FUNCTION__, MPtr, BaseMPtr, AccessSize, ObjSize, EncodingNo);
  if (EncodingNo)
    EncodingCommonTy::check(MPtr, BaseMPtr, AccessSize, ObjSize,
                            /*FailOnError=*/true);
}

OBJSAN_SMALL_API_ATTRS
void *__objsan_pre_load(char *VPtr, char *BaseMPtr, char *LVRI,
                        uint64_t AccessSize, char *MPtr, uint64_t ObjSize,
                        int8_t EncodingNo, int8_t WasChecked) {
  PRINTF("%s start P: %p/%p B: %p L: %p AS: %llu OS: %llu Enc: %i C: %i\n",
         __PRETTY_FUNCTION__, VPtr, MPtr, BaseMPtr, LVRI, AccessSize, ObjSize,
         EncodingNo, WasChecked);
  if (EncodingNo && !WasChecked && !LVRI &&
      !EncodingCommonTy::check(MPtr, BaseMPtr, AccessSize, ObjSize,
                               /*FailOnError=*/false)) [[unlikely]] {
    FPRINTF("l bad %p %p %p %llu %llu %i %i\n", MPtr, BaseMPtr, LVRI,
            AccessSize, ObjSize, EncodingNo, WasChecked);
    __builtin_trap();
    return nullptr;
  }
  return MPtr;
}

OBJSAN_SMALL_API_ATTRS
void *__objsan_pre_store(char *VPtr, char *BaseMPtr, char *LVRI,
                         uint64_t AccessSize, char *MPtr, uint64_t ObjSize,
                         int8_t EncodingNo, int8_t WasChecked) {
  PRINTF("%s start P: %p/%p B: %p L: %p AS: %llu OS: %llu Enc: %i C: %i\n",
         __PRETTY_FUNCTION__, VPtr, MPtr, BaseMPtr, LVRI, AccessSize, ObjSize,
         EncodingNo, WasChecked);
  if (EncodingNo && !WasChecked && !LVRI &&
      !EncodingCommonTy::check(MPtr, BaseMPtr, AccessSize, ObjSize,
                               /*FailOnError=*/false)) [[unlikely]] {
    FPRINTF("s bad %p %p %p %llu %llu %i %i\n", MPtr, BaseMPtr, LVRI,
            AccessSize, ObjSize, EncodingNo, WasChecked);
    __builtin_trap();
    return nullptr;
  }
  return MPtr;
}

OBJSAN_SMALL_API_ATTRS
void __objsan_check_non_zero(char *VPtr) {
  if (!VPtr)
    __builtin_trap();
}

OBJSAN_SMALL_API_ATTRS
void *__objsan_decode(char *VPtr) {
  uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(VPtr);
  ENCODING_NO_SWITCH(decode, EncodingNo, VPtr, VPtr);
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
  char *MPtr = /* TODO */ nullptr;
  if (__objsan_pre_load(VPtr, BaseMPtr, nullptr, sizeof(void *), MPtr, ObjSize,
                        EncodingNo, false)) {
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
    char *MPtr = /* TODO */ nullptr;                                           \
    if (void *AccMPtr = __objsan_pre_load(VPtr, BaseMPtr, nullptr, SIZE, MPtr, \
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
}
