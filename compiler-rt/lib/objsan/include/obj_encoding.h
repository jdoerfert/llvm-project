#ifndef OBJSAN_OBJ_ENCODING_H
#define OBJSAN_OBJ_ENCODING_H

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdatomic.h>

namespace __objsan {

struct EncodingCommonTy {
  static constexpr uint64_t MAGIC = 0b101;
  static constexpr uint64_t NumMagicBits = 3;
  static constexpr uint64_t NumEncodingBits = 2;

  union EncTy {
    char *VPtr;
    struct __attribute__((packed)) {
      uint64_t Bits : (sizeof(char *) * 8) - NumEncodingBits;
      uint32_t EncodingId : NumEncodingBits;
    } Bits;
    EncTy(char *VPtr) : VPtr(VPtr) {}
  };

  static uint8_t getEncodingNo(char *VPtr) {
    EncTy E(VPtr);
    return E.Bits.EncodingId;
  }

  static bool getMagic(char *VPtr, uint64_t OffsetBits) {
    uint64_t PtrMagic =
        ((((uint64_t)VPtr) >> OffsetBits) & ((1ULL << NumMagicBits) - 1));
    return PtrMagic;
  }

  static char *checkAndAdjust(char *__restrict VPtr, uint64_t Magic,
                              char *__restrict MPtr, uint64_t AccessSize,
                              int64_t Offset, uint64_t ObjSize,
                              bool FailOnError) {
#if 0
    printf("Check %p size %llu -- access %llu @ %lli\n", MPtr, ObjSize,
           AccessSize, Offset);
#endif
    if (Magic != MAGIC || Offset < 0 || Offset + AccessSize > ObjSize)
        [[unlikely]] {
      if (FailOnError) {
        fprintf(stderr, "memory out-of-bound %llu + %llu vs %llu! (Base %p)\n",
                Offset, AccessSize, ObjSize, (void *)MPtr);
        // TODO: Configure this to report if requested
        __builtin_trap();
      }
      // TODO: Configure this to trap or report if requested
      return nullptr;
    }
#if 0
    printf("--> %p\n", MPtr + Offset);
#endif
    return MPtr + Offset;
  }
};

template <uint64_t EncodingNo> struct EncodingBaseTy {
  static constexpr uint64_t NumEncodingBits = EncodingCommonTy::NumEncodingBits;
  static_assert(EncodingNo < (1UL << NumEncodingBits), "Encoding out-of-range");
  static constexpr uint64_t NumMagicBits = EncodingCommonTy::NumMagicBits;
  static constexpr uint64_t MAGIC = EncodingCommonTy::MAGIC;
  static constexpr uint64_t EncNo = EncodingNo;
};

template <uint64_t EncodingNo, uint64_t OffsetBits, uint64_t BucketBits,
          uint64_t RealPtrBits>
struct BucketSchemeTy : public EncodingBaseTy<EncodingNo> {
  using Base = EncodingBaseTy<EncodingNo>;
  static constexpr uint64_t NumOffsetBits = OffsetBits;
  static constexpr uint64_t NumBucketBits = BucketBits;
  static constexpr uint64_t NumRealPtrBits = RealPtrBits;
  static constexpr uint64_t NumEncodingBits = Base::NumEncodingBits;
  static constexpr uint64_t NumMagicBits = Base::NumMagicBits;

  static_assert(NumEncodingBits + NumMagicBits + NumOffsetBits * 2 +
                        NumBucketBits + NumRealPtrBits ==
                    (8 * sizeof(char *)),
                "Size mismatch!");

  static constexpr uint64_t NumBuckets = 1 << BucketBits;
  uint64_t Buckets[NumBuckets];

  void reset() {
    for (uint64_t I = 0; I < NumBuckets; ++I)
      Buckets[I] = 0;
  }

  union EncTy {
    char *VPtr;
    struct __attribute__((packed)) {
      int32_t Offset : NumOffsetBits;
      uint64_t Magic : NumMagicBits;
      uint64_t ObjSize : NumOffsetBits;
      uint64_t BuckedIdx : NumBucketBits;
      uint64_t RealPtr : NumRealPtrBits;
      uint64_t EncodingId : NumEncodingBits;
    } Bits;
    static_assert(sizeof(Bits) == sizeof(char *), "bad size");

    EncTy(uint64_t ObjSize, uint64_t BuckedIdx, uint64_t RealPtr) {
      Bits.Offset = 0;
      Bits.Magic = Base::MAGIC;
      Bits.ObjSize = ObjSize;
      Bits.BuckedIdx = BuckedIdx;
      Bits.RealPtr = RealPtr;
      Bits.EncodingId = EncodingNo;
    }
    EncTy(char *VPtr) : VPtr(VPtr) {}
  };
  static_assert(sizeof(EncTy) == sizeof(char *), "bad size");

  static constexpr uint64_t NumBucketValueBits =
      (8 * sizeof(char *) - NumRealPtrBits);
  static_assert(NumBucketValueBits <= sizeof(Buckets[0]) * 8,
                "Bucket value too large!");

  union DecTy {
    char *MPtr;
    struct __attribute__((packed)) {
      uint64_t RealPtr : NumRealPtrBits;
      uint64_t BucketValue : NumBucketValueBits;
    } Bits;

    DecTy(char *MPtr) : MPtr(MPtr) {}
    DecTy(uint64_t RealPtr, uint64_t BucketValue) {
      Bits.RealPtr = RealPtr;
      Bits.BucketValue = BucketValue;
    }
  };
  static_assert(sizeof(DecTy) == sizeof(char *), "bad size");

  static uint64_t getMaxSize() { return (1ULL << NumOffsetBits); }

  char *encode(char *MPtr, uint64_t ObjSize) {
    assert(ObjSize < getMaxSize() && "Object is not small!");

    DecTy D(MPtr);
    uint64_t BucketIdx = ~0u;
    for (uint64_t Idx = 0; Idx < NumBuckets; ++Idx) {
      uint64_t Zero = 0;
      uint64_t Desired = D.Bits.BucketValue;
      auto BucketValue = atomic_load((atomic_uint_least64_t *)(&Buckets[Idx]));
      if (BucketValue == Desired) [[likely]] {
        BucketIdx = Idx;
        break;
      }
      if (BucketValue)
        continue;
      if (atomic_compare_exchange_strong_explicit(
              (atomic_uint_least64_t *)&Buckets[Idx], &Zero, Desired,
              memory_order_release, memory_order_relaxed)) {
        BucketIdx = Idx;
        break;
      }
    }
    if (BucketIdx == ~0u) [[unlikely]]
      return nullptr;
    EncTy E(ObjSize, BucketIdx, D.Bits.RealPtr);
    return E.VPtr;
  }

  void free(char *VPtr) { assert(0 && "bucket objects cannot be freed!"); }

  char *decode(char *VPtr) {
    EncTy E(VPtr);
    DecTy D(E.Bits.RealPtr, Buckets[E.Bits.BuckedIdx]);
    return D.MPtr + E.Bits.Offset;
  }

  bool isMagicIntact(char *VPtr) {
    EncTy E(VPtr);
    return E.Bits.Magic == Base::MAGIC;
  }

  uint64_t getSize(char *VPtr) {
    EncTy E(VPtr);
    return E.Bits.ObjSize;
  }

  char *getBasePointerInfo(char *VPtr, uint64_t *__restrict SizePtr,
                           uint64_t *__restrict NumOffsetBitsPtr) {
    EncTy E(VPtr);
    DecTy D(E.Bits.RealPtr, Buckets[E.Bits.BuckedIdx]);
    __builtin_prefetch(D.MPtr, 0, 3);
    *SizePtr = E.Bits.ObjSize;
    *NumOffsetBitsPtr = NumOffsetBits;
    return D.MPtr;
  }

  char *getBase(char *VPtr) {
    EncTy E(VPtr);
    DecTy D(E.Bits.RealPtr, Buckets[E.Bits.BuckedIdx]);
    return D.MPtr;
  }

  char *getBaseVPtr(char *VPtr) {
    EncTy ED(VPtr);
    return VPtr - ED.Bits.Offset;
  }
};

template <uint64_t EncodingNo, uint64_t ObjectBits>
struct LedgerSchemeTy : public EncodingBaseTy<EncodingNo> {
  using Base = EncodingBaseTy<EncodingNo>;
  static constexpr uint64_t NumEncodingBits = Base::NumEncodingBits;
  static constexpr uint64_t NumMagicBits = Base::NumMagicBits;

  struct ObjDescTy {
    uint64_t ObjSize;
    char *Base;
  };

  static constexpr uint64_t NumObjectBits = ObjectBits;
  static constexpr uint64_t NumOffsetBits =
      (sizeof(void *) * 8) - NumObjectBits - NumEncodingBits - NumMagicBits;
  static constexpr uint64_t NumObjects = 1 << ObjectBits;

  ObjDescTy Objects[NumObjects];
  atomic_uint_least64_t NumObjectsUsed = 0;

  void reset() { NumObjectsUsed = 0; }

  union EncTy {
    char *VPtr;
    struct __attribute__((packed)) {
      int64_t Offset : NumOffsetBits;
      uint64_t Magic : NumMagicBits;
      uint64_t ObjectIdx : NumObjectBits;
      uint64_t EncodingId : NumEncodingBits;
    } Bits;
    static_assert(sizeof(Bits) == sizeof(char *), "bad size");

    EncTy(uint64_t ObjSize, uint64_t ObjectIdx) {
      Bits.Offset = 0;
      Bits.Magic = Base::MAGIC;
      Bits.ObjectIdx = ObjectIdx;
      Bits.EncodingId = EncodingNo;
    }
    EncTy(char *VPtr) : VPtr(VPtr) {}
  };
  static_assert(sizeof(EncTy) == sizeof(char *), "bad size");

  char *encode(char *MPtr, uint64_t ObjSize) {
    assert(ObjSize < (1ULL << NumOffsetBits));
    uint64_t ObjectIdx =
        atomic_fetch_add_explicit(&NumObjectsUsed, 1, memory_order_relaxed);
    if (ObjectIdx >= NumObjects) {
      fprintf(stderr, "out of objects!\n");
      __builtin_trap();
    }
    Objects[ObjectIdx] = {ObjSize, MPtr};
    EncTy E(ObjSize, ObjectIdx);
    return E.VPtr;
  }

  void free(char *VPtr) {
    EncTy E(VPtr);
    __builtin_assume(E.Bits.ObjectIdx < NumObjects);
    Objects[E.Bits.ObjectIdx].ObjSize = 0;
  }

  char *decode(char *VPtr) {
    EncTy E(VPtr);
    __builtin_assume(E.Bits.ObjectIdx < NumObjects);
    auto [ObjSize, Base] = Objects[E.Bits.ObjectIdx];
    return Base + E.Bits.Offset;
  }

  bool isMagicIntact(char *VPtr) {
    EncTy E(VPtr);
    return E.Bits.Magic == Base::MAGIC;
  }

  uint64_t getSize(char *VPtr) {
    EncTy E(VPtr);
    __builtin_assume(E.Bits.ObjectIdx < NumObjects);
    auto [ObjSize, Base] = Objects[E.Bits.ObjectIdx];
    return ObjSize;
  }

  char *getBasePointerInfo(char *VPtr, uint64_t *__restrict SizePtr,
                           uint64_t *__restrict NumOffsetBitsPtr) {
    EncTy E(VPtr);
    __builtin_assume(E.Bits.ObjectIdx < NumObjects);
    ObjDescTy &Obj = Objects[E.Bits.ObjectIdx];
    __builtin_prefetch(&Obj + 8, 0, 3);
    __builtin_prefetch(&Obj + 16, 0, 3);
    __builtin_prefetch(Obj.Base, 0, 3);
    *SizePtr = Obj.ObjSize;
    *NumOffsetBitsPtr = NumOffsetBits;
    return Obj.Base;
  }

  char *getBase(char *VPtr) {
    EncTy E(VPtr);
    __builtin_assume(E.Bits.ObjectIdx < NumObjects);
    auto [ObjSize, Base] = Objects[E.Bits.ObjectIdx];
    return Base;
  }

  char *getBaseVPtr(char *VPtr) {
    EncTy E(VPtr);
    return VPtr - E.Bits.Offset;
  }
};

template <uint64_t EncodingNo, uint64_t ObjectBits, uint64_t FixedSize>
struct FixedLedgerSchemeTy : public EncodingBaseTy<EncodingNo> {
  using Base = EncodingBaseTy<EncodingNo>;
  static constexpr uint64_t NumEncodingBits = Base::NumEncodingBits;
  static constexpr uint64_t NumMagicBits = Base::NumMagicBits;
  static constexpr uint64_t ObjSize = FixedSize;

  struct ObjDescTy {
    char *Base;
  };

  static constexpr uint64_t NumObjectBits = ObjectBits;
  static constexpr uint64_t NumOffsetBits =
      (sizeof(void *) * 8) - NumObjectBits - NumEncodingBits - NumMagicBits;
  static constexpr uint64_t NumObjects = 1 << ObjectBits;

  ObjDescTy Objects[NumObjects];
  atomic_uint_least64_t NumObjectsUsed = 0;

  void reset() { NumObjectsUsed = 0; }

  union EncTy {
    char *VPtr;
    struct __attribute__((packed)) {
      int64_t Offset : NumOffsetBits;
      uint64_t Magic : NumMagicBits;
      uint64_t ObjectIdx : NumObjectBits;
      uint64_t EncodingId : NumEncodingBits;
    } Bits;
    static_assert(sizeof(Bits) == sizeof(char *), "bad size");

    EncTy(uint64_t ObjectIdx) {
      Bits.Offset = 0;
      Bits.Magic = Base::MAGIC;
      Bits.ObjectIdx = ObjectIdx;
      Bits.EncodingId = EncodingNo;
    }
    EncTy(char *VPtr) : VPtr(VPtr) {}
  };
  static_assert(sizeof(EncTy) == sizeof(char *), "bad size");

  char *encode(char *MPtr, uint64_t ObjSize) {
    assert(ObjSize == FixedSize);
    uint64_t ObjectIdx =
        atomic_fetch_add_explicit(&NumObjectsUsed, 1, memory_order_relaxed);
    if (ObjectIdx >= NumObjects) {
      fprintf(stderr, "out of objects!\n");
      __builtin_trap();
    }
    Objects[ObjectIdx] = {MPtr};
    EncTy E(ObjectIdx);
    return E.VPtr;
  }

  void free(char *VPtr) {
    EncTy E(VPtr);
    __builtin_assume(E.Bits.ObjectIdx < NumObjects);
    Objects[E.Bits.ObjectIdx].Base = 0;
  }

  char *decode(char *VPtr) {
    EncTy E(VPtr);
    __builtin_assume(E.Bits.ObjectIdx < NumObjects);
    auto *Base = Objects[E.Bits.ObjectIdx];
    return Base + E.Bits.Offset;
  }

  bool isMagicIntact(char *VPtr) {
    EncTy E(VPtr);
    return E.Bits.Magic == Base::MAGIC;
  }

  uint64_t getSize(char *VPtr) {
    EncTy E(VPtr);
    return ObjSize;
  }

  char *getBasePointerInfo(char *VPtr, uint64_t *__restrict SizePtr,
                           uint64_t *__restrict NumOffsetBitsPtr) {
    EncTy E(VPtr);
    __builtin_assume(E.Bits.ObjectIdx < NumObjects);
    ObjDescTy &Obj = Objects[E.Bits.ObjectIdx];
    *SizePtr = ObjSize;
    *NumOffsetBitsPtr = NumOffsetBits;
    return Obj.Base;
  }

  char *getBase(char *VPtr) {
    EncTy E(VPtr);
    __builtin_assume(E.Bits.ObjectIdx < NumObjects);
    auto *Base = Objects[E.Bits.ObjectIdx];
    return Base;
  }

  char *getBaseVPtr(char *VPtr) {
    EncTy E(VPtr);
    return VPtr - E.Bits.Offset;
  }
};

} // namespace __objsan

#endif // OBJSAN_OBJ_ENCODING_H
