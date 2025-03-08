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
};

template <uint64_t EncodingNo> struct EncodingBaseTy {
  static constexpr uint64_t NumEncodingBits = EncodingCommonTy::NumEncodingBits;
  static_assert(EncodingNo < (1UL << NumEncodingBits), "Encoding out-of-range");

  static constexpr uint64_t MAGIC = 0b101;
  static constexpr uint64_t NumMagicBits = 3;

  static char *checkAndAdjust(char *MPtr, uint64_t AccessSize, int64_t Offset,
                              uint64_t ObjSize, bool FailOnError) {
    printf("Check %p size %llu -- access %llu @ %lli\n", MPtr, ObjSize,
           AccessSize, Offset);
    if (Offset < 0 || Offset + AccessSize > ObjSize) {
      if (!FailOnError)
        return nullptr;
      fprintf(stderr, "memory out-of-bound %llu + %llu vs %llu! (Base %p)\n",
              Offset, AccessSize, ObjSize, (void *)MPtr);
      __builtin_trap();
    }
    printf("--> %p\n", MPtr + Offset);
    return MPtr + Offset;
  }
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
  atomic_uint_least64_t Buckets[NumBuckets];

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
      if (atomic_compare_exchange_strong_explicit(&Buckets[Idx], &Zero, Desired,
                                                  memory_order_release,
                                                  memory_order_relaxed)) {
        BucketIdx = Idx;
        break;
      }
    }
    if (BucketIdx == ~0u) {
      fprintf(stderr, "out of buckets!\n");
      __builtin_trap();
    }
    EncTy E(ObjSize, BucketIdx, D.Bits.RealPtr);
    return E.VPtr;
  }

  void free(char *VPtr) { assert(0 && "bucket objects cannot be freed!"); }

  char *decode(char *VPtr) {
    EncTy E(VPtr);
    DecTy D(E.Bits.RealPtr, Buckets[E.Bits.BuckedIdx]);
    return D.MPtr + E.Bits.Offset;
  }

  char *checkAccess(char *VPtr, uint64_t AccessSize) {
    EncTy E(VPtr);
    DecTy D(E.Bits.RealPtr, Buckets[E.Bits.BuckedIdx]);
    return Base::checkAndAdjust(D.MPtr, AccessSize, E.Bits.Offset,
                                E.Bits.ObjSize);
  }

  char *checkAccessRange(char *BaseMPtr, uint64_t AccessSize, char *VPtr,
                         uint64_t ObjSize, int64_t BaseOffset,
                         bool FailOnError = true) {
    EncTy E(VPtr);
    if (ObjSize == ~0ULL)
      ObjSize = E.Bits.ObjSize;
    return Base::checkAndAdjust(BaseMPtr, AccessSize, E.Bits.Offset, ObjSize,
                                FailOnError);
  }

  bool isMagicIntact(char *VPtr) {
    EncTy E(VPtr);
    return E.Bits.Magic == Base::MAGIC;
  }

  uint64_t getSize(char *VPtr) {
    EncTy E(VPtr);
    return E.Bits.ObjSize;
  }

  char *getBasePointerInfo(char *VPtr, uint64_t *SizePtr, int64_t *OffsetPtr) {
    EncTy E(VPtr);
    DecTy D(E.Bits.RealPtr, Buckets[E.Bits.BuckedIdx]);
    *SizePtr = E.Bits.ObjSize;
    *OffsetPtr = E.Bits.Offset;
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
    char *Base;
    uint64_t ObjSize;
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
    Objects[ObjectIdx] = {MPtr, ObjSize};
    EncTy E(ObjSize, ObjectIdx);
    return E.VPtr;
  }

  void free(char *VPtr) {
    EncTy E(VPtr);
    Objects[E.Bits.ObjectIdx].ObjSize = 0;
  }

  char *decode(char *VPtr) {
    EncTy E(VPtr);
    auto [Base, ObjSize] = Objects[E.Bits.ObjectIdx];
    return Base + E.Bits.Offset;
  }

  char *checkAccess(char *VPtr, uint64_t AccessSize) {
    EncTy E(VPtr);
    auto [Base, ObjSize] = Objects[E.Bits.ObjectIdx];
    return Base::checkAndAdjust(Base, AccessSize, E.Bits.Offset, ObjSize);
  }

  char *checkAccessRange(char *BaseMPtr, uint64_t AccessSize, char *VPtr,
                         uint64_t ObjSize, int64_t BaseOffset,
                         bool FailOnError = true) {
    EncTy E(VPtr);
    if (ObjSize == ~0ULL)
      ObjSize = Objects[E.Bits.ObjectIdx].ObjSize;
    return Base::checkAndAdjust(BaseMPtr, AccessSize, E.Bits.Offset, ObjSize,
                                FailOnError);
  }

  bool isMagicIntact(char *VPtr) {
    EncTy E(VPtr);
    return E.Bits.Magic == Base::MAGIC;
  }

  uint64_t getSize(char *VPtr) {
    EncTy E(VPtr);
    auto [Base, ObjSize] = Objects[E.Bits.ObjectIdx];
    return ObjSize;
  }

  char *getBasePointerInfo(char *VPtr, uint64_t *SizePtr, int64_t *OffsetPtr) {
    EncTy E(VPtr);
    ObjDescTy &Obj = Objects[E.Bits.ObjectIdx];
    *SizePtr = Obj.ObjSize;
    *OffsetPtr = E.Bits.Offset;
    return Obj.Base;
  }

  char *getBase(char *VPtr) {
    EncTy E(VPtr);
    auto [Base, ObjSize] = Objects[E.Bits.ObjectIdx];
    return Base;
  }

  char *getBaseVPtr(char *VPtr) {
    EncTy E(VPtr);
    return VPtr - E.Bits.Offset;
  }
};

} // namespace __objsan

#endif // OBJSAN_OBJ_ENCODING_H
