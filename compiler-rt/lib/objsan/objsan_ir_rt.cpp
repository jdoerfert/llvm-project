#include "include/common.h"

#include "include/obj_encoding.h"

#define OBJSAN_SMALL_API_ATTRS [[gnu::flatten, clang::always_inline]]
#define OBJSAN_BIG_API_ATTRS [[clang::always_inline]]

#ifndef __DARWIN_ALIAS
#define __DARWIN_ALIAS(sym)
#endif

#ifdef DEBUG
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif

// Forward declaration
extern "C" {
// FIXME: This one is not used.
// double strtod(const char *, char **);

// FIXME: This is a function in <unistd.h>. It will prevent us from freestanding
// build, and even on Windows.
#ifndef __OBJSAN_DEVICE__
int execvp(const char *__file, char *const *__argv);
int getopt(int argc, char *const argv[], const char *optstring)
    __DARWIN_ALIAS(getopt);
int getopt_long(int argc, char *const *argv, const char *optstring,
                const struct option *longopts, int *longindex);
#endif

void *malloc(size_t size);
size_t strlen(const char *str);
}

// Also in LightSan.cpp
static uint32_t MaxObjSizeForShadow = 64;

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
  PRINTF("%s %p %lli\n", __PRETTY_FUNCTION__, MPtr, ObjSize);
  return __objsan_register_object(MPtr, ObjSize, RequiresTemporalCheck);
}

// OBJSAN_BIG_API_ATTRS
__attribute__((optnone, noinline)) char *
__objsan_pre_global(char *MPtr, int32_t ObjSize, int8_t IsDefinition,
                    int8_t RequiresTemporalCheck) {
  [[maybe_unused]] uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(MPtr);
  PRINTF("%s %p %i [%i] (%i)\n", __PRETTY_FUNCTION__, MPtr, ObjSize, EncodingNo,
         IsDefinition);
  if (!IsDefinition)
    return MPtr;
  auto *VPtr = __objsan_register_object(MPtr, ObjSize, RequiresTemporalCheck);
  PRINTF(" -> %p\n", VPtr);
  return VPtr;
}

static inline void makeRealArgV(char *Ptr) {
  char **PtrAddr = *reinterpret_cast<char ***>(Ptr);
  int I = 0;
  while (PtrAddr[I])
    ++I;
  char **FakeEnv = (char **)malloc((I + 1) * sizeof(char *));
  I = 0;
  while (PtrAddr[I]) {
    FakeEnv[I] = LargeObjects.decode(PtrAddr[I]);
    ++I;
  }
  FakeEnv[I] = nullptr;
  *reinterpret_cast<char **>(Ptr) = (char *)FakeEnv;
}

OBJSAN_BIG_API_ATTRS
void __objsan_pre_call(void *Callee, int64_t IntrinsicId,
                       int32_t num_parameters, char *parameters,
                       int64_t AccessLength1, char *Obj1VPtr, char *Obj1MPtr,
                       char *Obj1BaseMPtr, int64_t Obj1Size, int8_t Obj1EncNo,
                       int64_t AccessLength2, char *Obj2VPtr, char *Obj2MPtr,
                       char *Obj2BaseMPtr, int64_t Obj2Size, int8_t Obj2EncNo,
                       int32_t ID) {
  PRINTF("%s start: %i\n", __PRETTY_FUNCTION__, num_parameters);
  if (Obj1EncNo)
    EncodingCommonTy::check(Obj1MPtr, Obj1BaseMPtr, AccessLength1, Obj1Size,
                            /*FailOnError=*/true, ID, ID);
  if (Obj2EncNo)
    EncodingCommonTy::check(Obj2MPtr, Obj2BaseMPtr, AccessLength2, Obj2Size,
                            /*FailOnError=*/true, ID, ID);

#ifndef __OBJSAN_DEVICE__
  //  TODO: this should switch on closed world
  // Copy the shadow over as part of a memcpy/move
  if (IntrinsicId == 238 || IntrinsicId == 241) {
    if (Obj1EncNo && Obj2EncNo && Obj1Size <= MaxObjSizeForShadow &&
        Obj1Size <= MaxObjSizeForShadow) {
      if (IntrinsicId == 238)
        __builtin_memcpy(Obj1MPtr + Obj1Size, Obj2MPtr + Obj2Size,
                         AccessLength1);
      else
        __builtin_memmove(Obj1MPtr + Obj1Size, Obj2MPtr + Obj2Size,
                          AccessLength1);
    }
  }
#endif

  for (int32_t I = 0; I < num_parameters; ++I) {
    ParameterValuePackTy *VP = (ParameterValuePackTy *)parameters;
    assert(VP->TypeId == 14);
    char *VPValuePtr = reinterpret_cast<char *>(&VP->Value);
    char **PtrAddr = reinterpret_cast<char **>(VPValuePtr);
    uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(*PtrAddr);
    if (EncodingNo)
      *PtrAddr = [&]() -> char * {
        ENCODING_NO_SWITCH(decode, EncodingNo, nullptr, *PtrAddr);
      }();
    parameters += sizeof(ParameterValuePackTy) + sizeof(void *);
  }

#ifndef __OBJSAN_DEVICE__
  if (Callee == &getopt) {
    ParameterValuePackTy *VP =
        (ParameterValuePackTy *)(parameters -
                                 2 * (sizeof(ParameterValuePackTy) + 8));
    makeRealArgV((char *)&VP->Value);
  } else if (Callee == &getopt_long) {
    ParameterValuePackTy *VP =
        (ParameterValuePackTy *)(parameters -
                                 4 * (sizeof(ParameterValuePackTy) + 8));
    makeRealArgV((char *)&VP->Value);
  } else if (Callee == &execvp) {
    ParameterValuePackTy *VP =
        (ParameterValuePackTy *)(parameters -
                                 (sizeof(ParameterValuePackTy) + 8));
    makeRealArgV((char *)&VP->Value);
  }
#endif

  PRINTF("%s done\n", __PRETTY_FUNCTION__);
}

OBJSAN_BIG_API_ATTRS
char *__objsan_post_call(char *MPtr, uint64_t ObjSize,
                         int8_t RequiresTemporalCheck) {
  PRINTF("%s start: p: %p os: %" PRIu64 " rtc: %i\n", __PRETTY_FUNCTION__, MPtr,
         ObjSize, RequiresTemporalCheck);
  if (!MPtr)
    return nullptr;
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
  uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(VPtr);
  ENCODING_NO_SWITCH(free, EncodingNo, , VPtr);
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
  auto ArgVSize = sizeof(char *) * (ArgC + 1);
  char **ArgV = *reinterpret_cast<char ***>(&ArgVVP->Value);
  char **NewArgV = (char **)malloc(2 * ArgVSize);
  // TODO: Use a global constant to determine if we assume closed world.
  // this does not. In a closed world we don't need a new array (malloc)
#if 1
  for (int32_t I = 0; I < ArgC; ++I) {
    auto StrSize = strlen(ArgV[I]) + 1;
    NewArgV[I] = ArgV[I];
    NewArgV[I + ArgC + 1] =
        __objsan_register_object(ArgV[I], StrSize,
                                 /*RequiresTemporalCheck=*/true);
  }
#endif
  NewArgV[2 * (ArgC + 1)] = nullptr;
  *reinterpret_cast<char **>(&ArgVVP->Value) =
      __objsan_register_object(reinterpret_cast<char *>(NewArgV), ArgVSize,
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

OBJSAN_SMALL_API_ATTRS char *
__objsan_post_base_pointer_info(char *__restrict VPtr, uint64_t *SizePtr,
                                uint8_t *EncNoPtr, int32_t ID) {
  uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(VPtr);
  *SizePtr = 0;
  *EncNoPtr = 0;
  if (!EncodingNo)
    return nullptr;
  char *MPtr = [&]() -> char * {
    ENCODING_NO_SWITCH(getBasePointerInfo, EncodingNo, nullptr, VPtr, SizePtr,
                       EncNoPtr);
  }();
  PRINTF("%s P: %p/%p Enc: %i OS: %" PRIu64 " [%i]\n", __PRETTY_FUNCTION__,
         VPtr, MPtr, EncodingNo, *SizePtr, ID);
  return MPtr;
}

OBJSAN_SMALL_API_ATTRS
void *__objsan_get_mptr(char *__restrict VPtr, char *__restrict BaseMPtr,
                        uint8_t EncodingNo) {
  PRINTF("%s %p %p %i start\n", __PRETTY_FUNCTION__, VPtr, BaseMPtr,
         EncodingNo);
  if (!EncodingNo) [[unlikely]]
    return VPtr;
  int64_t NumOffsetBits;
  if (EncodingNo == 1)
    NumOffsetBits = SmallObjectsTy::NumOffsetBits;
  else
    NumOffsetBits = LargeObjectsTy::NumOffsetBits;
  auto [VOffset, VMagic] = getOffsetAndMagic(VPtr, NumOffsetBits);
  auto [MOffset, MMagic] = getOffsetAndMagic(BaseMPtr, NumOffsetBits);
  // TODO: Checm magic
  return BaseMPtr + VOffset - MOffset;
}

OBJSAN_SMALL_API_ATTRS
char *__objsan_post_loop_value_range(char *BeginMPtr, char *EndMPtr,
                                     int64_t MaxOffset, char *BaseVPtr,
                                     char *BaseMPtr, uint64_t ObjSize,
                                     int8_t EncodingNo,
                                     int8_t IsDefinitivelyExecuted,
                                     int32_t MinID, int32_t MaxID) {
#ifdef STATS
  if (!EncodingNo) {
    ++SLoopR.Enc0;
    return BaseMPtr;
  }
  if (EncodingNo == 1)
    ++SLoopR.Enc1;
  else if (EncodingNo == 2)
    ++SLoopR.Enc2;
  else
    ++SLoopR.EncX;
#endif
  if (!EncodingNo)
    return nullptr;
  PRINTF("%s start\n", __PRETTY_FUNCTION__);
  int64_t LoopSize = EndMPtr - BeginMPtr;
  if (EncodingNo && !EncodingCommonTy::check(
                        BeginMPtr, BaseMPtr, LoopSize + MaxOffset, ObjSize,
                        /*FailOnError=*/IsDefinitivelyExecuted, MinID, MaxID))
      [[unlikely]] {
    PRINTF("r bad %p-%p %p %" PRId64 " +%" PRIu64 " %" PRIu64 " %i [%i:%i]\n",
           BeginMPtr, EndMPtr, BaseMPtr, LoopSize, MaxOffset, ObjSize,
           EncodingNo, MinID, MaxID);
    return nullptr;
  }
  return /* not null */ (char *)(0x1);
}

OBJSAN_SMALL_API_ATTRS
void __objsan_pre_ranged_access(char *MPtr, char *BaseMPtr, int64_t AccessSize,
                                uint64_t ObjSize, int8_t EncodingNo, int ID) {
#ifdef STATS
  if (!EncodingNo) {
    ++SRange.Enc0;
    return;
  }
  if (EncodingNo == 1)
    ++SRange.Enc1;
  else if (EncodingNo == 2)
    ++SRange.Enc2;
  else
    ++SRange.EncX;
#endif
  if (!EncodingNo)
    return;
  PRINTF("%s start P: %p B: %p AS: %" PRId64 " OS: %" PRIu64 " Enc: %i [%i]\n",
         __PRETTY_FUNCTION__, MPtr, BaseMPtr, AccessSize, ObjSize, EncodingNo,
         ID);
  if (EncodingNo)
    EncodingCommonTy::check(MPtr, BaseMPtr, AccessSize, ObjSize,
                            /*FailOnError=*/true, ID);
}

OBJSAN_SMALL_API_ATTRS
void *__objsan_pre_load(char *VPtr, char *BaseMPtr, char *LVRI,
                        uint64_t AccessSize, char *MPtr, uint64_t ObjSize,
                        int8_t EncodingNo, int8_t WasChecked, int32_t ID) {
#ifdef STATS
  if (!EncodingNo) {
    ++SLoads.Enc0;
    return MPtr;
  }
  if (EncodingNo == 1)
    ++SLoads.Enc1;
  else if (EncodingNo == 2)
    ++SLoads.Enc2;
  else
    ++SLoads.EncX;
#endif
  PRINTF("%s start P: %p/%p B: %p L: %p AS: %" PRIu64 " OS: %" PRIu64
         " Enc: %i C: %i\n",
         __PRETTY_FUNCTION__, VPtr, MPtr, BaseMPtr, LVRI, AccessSize, ObjSize,
         EncodingNo, WasChecked);
  if (EncodingNo && !WasChecked && !LVRI &&
      !EncodingCommonTy::check(MPtr, BaseMPtr, AccessSize, ObjSize,
                               /*FailOnError=*/false, ID)) [[unlikely]] {
    FPRINTF("l bad (%p) %p %p %p %" PRIu64 " %" PRIu64 " %i %i [%i]\n", VPtr,
            MPtr, BaseMPtr, LVRI, AccessSize, ObjSize, EncodingNo, WasChecked,
            ID);
    return nullptr;
  }
  return MPtr;
}

OBJSAN_SMALL_API_ATTRS
void *__objsan_pre_store(char *VPtr, char *BaseMPtr, char *LVRI,
                         uint64_t AccessSize, char *MPtr, uint64_t ObjSize,
                         int8_t EncodingNo, int8_t WasChecked, int32_t ID) {
#ifdef STATS
  if (!EncodingNo) {
    ++SStores.Enc0;
    return MPtr;
  }
  if (EncodingNo == 1)
    ++SStores.Enc1;
  else if (EncodingNo == 2)
    ++SStores.Enc2;
  else
    ++SStores.EncX;
#endif
  PRINTF("%s start P: %p/%p B: %p L: %p AS: %" PRIu64 " OS: %" PRIu64
         " Enc: %i C: %i [%i]\n",
         __PRETTY_FUNCTION__, VPtr, MPtr, BaseMPtr, LVRI, AccessSize, ObjSize,
         EncodingNo, WasChecked, ID);
  if (EncodingNo && !WasChecked && !LVRI &&
      !EncodingCommonTy::check(MPtr, BaseMPtr, AccessSize, ObjSize,
                               /*FailOnError=*/false, ID)) [[unlikely]] {
    FPRINTF("s bad (%p) %p %p %p %" PRIu64 " %" PRIu64 " %i %i [%i]\n", VPtr,
            MPtr, BaseMPtr, LVRI, AccessSize, ObjSize, EncodingNo, WasChecked,
            ID);
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
void *__objsan_post_load(char *BaseMPtr, char *LoadedMPtr, char *MPtr,
                         uint64_t ObjSize, int8_t EncodingNo, int32_t ID) {
  if (!EncodingNo || ObjSize > MaxObjSizeForShadow)
    return LoadedMPtr;
  auto **ShadowMPtr = (char **)(MPtr + ObjSize);
  auto *ShadowVPtr = *ShadowMPtr;
  char *ShadowVPtrMptr = (char *)__objsan_decode(ShadowVPtr);
  if (ShadowVPtrMptr == LoadedMPtr)
    return ShadowVPtr;
  if (ShadowVPtrMptr) {
    PRINTF("%p != %p (%ld : %ld) [%i]\n", ShadowVPtrMptr, LoadedMPtr,
           ShadowVPtrMptr - LoadedMPtr, LoadedMPtr - ShadowVPtrMptr, ID);
  }
  return LoadedMPtr;
}

OBJSAN_SMALL_API_ATTRS
void __objsan_post_store(char *BaseMPtr, char *StoredVPtr, char *MPtr,
                         uint64_t ObjSize, int8_t EncodingNo, int32_t ID) {
  if (!EncodingNo || ObjSize > MaxObjSizeForShadow)
    return;
  auto **ShadowMPtr = (char **)(MPtr + ObjSize);
  *ShadowMPtr = StoredVPtr;
}

OBJSAN_SMALL_API_ATTRS
void *__objsan_pre_va_arg(char *__restrict VPtr) {
  return __objsan_decode(VPtr);
}

OBJSAN_SMALL_API_ATTRS
uint8_t __objsan_post_icmp(uint8_t Result, uint32_t Predicate, char *LHS,
                           char *RHS, char *LHSBaseMPtr, char *RHSBaseMPtr,
                           int32_t ID) {
  if (LHSBaseMPtr != RHSBaseMPtr && LHSBaseMPtr && RHSBaseMPtr) {
    // TODO: this triggers on vectorized inserted alias checks
    //    FPRINTF("Pointer comparison of different objects (%p <> %p) [%p <> %p]
    //    [%i]!",
    //            LHS, RHS, LHSBaseMPtr, RHSBaseMPtr, ID);
    //    __builtin_trap();
  }
  auto *MPtrLHS = __objsan_decode(LHS);
  auto *MPtrRHS = __objsan_decode(RHS);
  switch (Predicate) {
  case 32:
    return MPtrLHS == MPtrRHS; // ==
  case 33:
    return MPtrLHS != MPtrRHS; // !=
  case 34:
    return MPtrLHS > MPtrRHS; // >u
  case 35:
    return MPtrLHS >= MPtrRHS; // >=u
  case 36:
    return MPtrLHS < MPtrRHS; // <u
  case 37:
    return MPtrLHS >= MPtrRHS; // >=u
  case 38:
    return (intptr_t)MPtrLHS > (intptr_t)MPtrRHS; // >s
  case 39:
    return (intptr_t)MPtrLHS >= (intptr_t)MPtrRHS; // >=s
  case 40:
    return (intptr_t)MPtrLHS < (intptr_t)MPtrRHS; // <s
  case 41:
    return (intptr_t)MPtrLHS <= (intptr_t)MPtrRHS; // <=s
  default:
    return Result;
  }
}

OBJSAN_SMALL_API_ATTRS
uint64_t __objsan_post_ptrtoint(char *VPtr, uint64_t PtrVal) {
  return reinterpret_cast<uint64_t>(__objsan_decode(VPtr));
}
OBJSAN_SMALL_API_ATTRS
void __objsan_post_ptrtoint_ind(char **VPtr, uint64_t VPtrSize,
                                uint64_t *PtrVal, uint64_t PtrValSize) {
  for (uint64_t I = 0; I < PtrValSize / sizeof(void *); ++I)
    PtrVal[I] = reinterpret_cast<uint64_t>(__objsan_decode(VPtr[I]));
}

OBJSAN_SMALL_API_ATTRS
int8_t __objsan_check_ptr_load(char *VPtr, int64_t Offset) {
  VPtr += Offset;
  uint8_t EncodingNo = EncodingCommonTy::getEncodingNo(VPtr);
  if (!EncodingNo)
    return 0;
  uint8_t EncNo;
  uint64_t ObjSize;
  auto *BaseMPtr = [&]() -> char * {
    ENCODING_NO_SWITCH(getBasePointerInfo, EncodingNo, nullptr, VPtr, &ObjSize,
                       &EncNo);
  }();
  char *MPtr = /* TODO */ nullptr;
  if (__objsan_pre_load(VPtr, BaseMPtr, nullptr, sizeof(void *), MPtr, ObjSize,
                        EncodingNo, false, 0)) {
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
    uint8_t EncNo;                                                             \
    uint64_t ObjSize;                                                          \
    auto *BaseMPtr = [&]() -> char * {                                         \
      ENCODING_NO_SWITCH(getBasePointerInfo, EncodingNo, nullptr, VPtr,        \
                         &ObjSize, &EncNo);                                    \
    }();                                                                       \
    char *MPtr = /* TODO */ nullptr;                                           \
    if (void *AccMPtr = __objsan_pre_load(VPtr, BaseMPtr, nullptr, SIZE, MPtr, \
                                          ObjSize, EncodingNo, false, 0)) {    \
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
