#ifndef OBJSAN_OBJ_ENCODING_H
#define OBJSAN_OBJ_ENCODING_H

#include "common.h"

#if __has_builtin(__builtin_assume)
#define ASSUME(E) __builtin_assume((E));
#else
#define ASSUME(E)
#endif

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

  static bool check(char *MPtr, char *MBasePtr, uint64_t AccessSize,
                    uint64_t ObjSize, bool FailOnError, int32_t ID0,
                    int32_t ID1 = 0) {
    char *MEndPtr = MBasePtr + ObjSize;
    char *MAccEndPtr = MPtr + AccessSize;
    if (MPtr < MBasePtr || MAccEndPtr > MEndPtr) [[unlikely]] {
      if (FailOnError) {
        FPRINTF("memory out-of-bound %p + %" PRIu64 " vs %p + %" PRIu64
                "! [%i:%i] (Base %p, %p, "
                "check [%i:%i])\n",
                MPtr, AccessSize, MBasePtr, ObjSize,
                MPtr<MBasePtr, MAccEndPtr> MEndPtr, (void *)MBasePtr, MPtr, ID0,
                ID1);
        // TODO: Configure this to report if requested
        __builtin_trap();
      }
      // TODO: Configure this to trap or report if requested
      return false;
    }
    return true;
  }

  static char *checkAndAdjust(char *__restrict VPtr, uint64_t Magic,
                              char *__restrict MPtr, uint64_t AccessSize,
                              int64_t Offset, uint64_t ObjSize,
                              bool FailOnError) {
#if 0
    printf("Check %p size %" PRIu64 " -- access %" PRId64 " @ %lli\n", MPtr,
           ObjSize, AccessSize, Offset);
#endif
    if (Magic != MAGIC || Offset < 0 || Offset + AccessSize > ObjSize)
        [[unlikely]] {
      if (FailOnError) {
        FPRINTF("memory out-of-bound %" PRId64 " + %" PRIu64 " vs %" PRIu64 "! "
                "(Base %p, %" PRIu64 " check&adjust)\n",
                Offset, AccessSize, ObjSize, (void *)MPtr, Magic);
        // TODO: Configure this to report if requested
        __builtin_trap();
      }
      // TODO: Configure this to trap or report if requested
      return nullptr;
    }
#if 0
    printf("--> %p\n", MPtr + Offset);
#endif
    return MPtr;
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
  using BucketEntryTy = uint32_t;
  BucketEntryTy Buckets[NumBuckets];

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

    EncTy(uint64_t ObjSize, uint64_t BuckedIdx, uint64_t RealPtr,
          uint64_t Offset) {
      Bits.Offset = Offset;
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
  static_assert(8 * sizeof(char *) - NumRealPtrBits <= 8 * sizeof(Buckets[0]));

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
      BucketEntryTy Zero = 0;
      BucketEntryTy Desired = D.Bits.BucketValue;
      auto BucketValue = __scoped_atomic_load_n(
          &Buckets[Idx], OrderingTy::aquire, MemScopeTy::device);
      if (BucketValue == Desired) [[likely]] {
        BucketIdx = Idx;
        break;
      }
      if (BucketValue)
        continue;
      if (__scoped_atomic_compare_exchange(
              &Buckets[Idx], &Zero, &Desired, false, OrderingTy::release,
              OrderingTy::relaxed, MemScopeTy::device)) {
        BucketIdx = Idx;
        break;
      }
    }
    if (BucketIdx == ~0u) [[unlikely]] {
      //      __builtin_trap();
      // printf("out of objects (small)!\n");
      return MPtr;
    }
    EncTy M(MPtr);
    EncTy E(ObjSize, BucketIdx, D.Bits.RealPtr, M.Bits.Offset);
    return E.VPtr;
  }

  void free(char *VPtr) { /* NoOp */ }

  char *decode(char *VPtr) {
    EncTy E(VPtr);
    DecTy D(E.Bits.RealPtr, Buckets[E.Bits.BuckedIdx]);
    EncTy M(D.MPtr);
    return D.MPtr + E.Bits.Offset - M.Bits.Offset;
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
                           uint8_t *__restrict EncNoPtr) {
    EncTy E(VPtr);
    DecTy D(E.Bits.RealPtr, Buckets[E.Bits.BuckedIdx]);
    if (E.Bits.Magic != Base::MAGIC) {
      return nullptr;
    }
    *EncNoPtr = EncodingNo;
    //    __builtin_prefetch(D.MPtr, 0, 3);
    *SizePtr = E.Bits.ObjSize;
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
  uint64_t NumObjectsUsed;
  uint32_t FreedObjs[NumObjects];
  uint64_t NumFreedObjects;

  void reset() {
    NumObjectsUsed = 0;
    NumFreedObjects = 0;
  }

  union EncTy {
    char *VPtr;
    struct __attribute__((packed)) {
      int64_t Offset : NumOffsetBits;
      uint64_t Magic : NumMagicBits;
      uint64_t ObjectIdx : NumObjectBits;
      uint64_t EncodingId : NumEncodingBits;
    } Bits;
    static_assert(sizeof(Bits) == sizeof(char *), "bad size");

    EncTy(uint64_t ObjSize, uint64_t ObjectIdx, uint64_t Offset) {
      Bits.Offset = Offset;
      Bits.Magic = Base::MAGIC;
      Bits.ObjectIdx = ObjectIdx;
      Bits.EncodingId = EncodingNo;
    }
    EncTy(char *VPtr) : VPtr(VPtr) {}
  };
  static_assert(sizeof(EncTy) == sizeof(char *), "bad size");

  char *encode(char *MPtr, uint64_t ObjSize) {
    assert(ObjSize < (1ULL << NumOffsetBits));
    uint64_t ObjectIdx = __scoped_atomic_fetch_add(
        &NumObjectsUsed, 1, OrderingTy::relaxed, MemScopeTy::device);
    if (ObjectIdx >= NumObjects) {
      // fprintf(stderr, "out of objects (large) (%lu)!\n", NumFreedObjects);
      if (!NumFreedObjects)
        return MPtr;
      ObjectIdx = FreedObjs[--NumFreedObjects];
      //      __builtin_trap();
      //      return MPtr;
    }
    EncTy M(MPtr);
    Objects[ObjectIdx] = {ObjSize, MPtr};
    EncTy E(ObjSize, ObjectIdx, M.Bits.Offset);
    return E.VPtr;
  }

  void free(char *VPtr) {
    EncTy E(VPtr);
    ASSUME(E.Bits.ObjectIdx < NumObjects);
    Objects[E.Bits.ObjectIdx].ObjSize = 0;
    // fprintf(stderr, "free (large) (%lu)!\n", NumFreedObjects);
    FreedObjs[NumFreedObjects++] = E.Bits.ObjectIdx;
  }

  char *decode(char *VPtr) {
    EncTy E(VPtr);
    ASSUME(E.Bits.ObjectIdx < NumObjects);
    auto [ObjSize, Base] = Objects[E.Bits.ObjectIdx];
    EncTy M(Base);
    return Base + E.Bits.Offset - M.Bits.Offset;
  }

  bool isMagicIntact(char *VPtr) {
    EncTy E(VPtr);
    return E.Bits.Magic == Base::MAGIC;
  }

  uint64_t getSize(char *VPtr) {
    EncTy E(VPtr);
    ASSUME(E.Bits.ObjectIdx < NumObjects);
    auto [ObjSize, Base] = Objects[E.Bits.ObjectIdx];
    return ObjSize;
  }

  char *getBasePointerInfo(char *VPtr, uint64_t *__restrict SizePtr,
                           uint8_t *__restrict EncNoPtr) {
    EncTy E(VPtr);
    ASSUME(E.Bits.ObjectIdx < NumObjects);
    if (E.Bits.Magic != Base::MAGIC) {
      return nullptr;
    }
    ObjDescTy &Obj = Objects[E.Bits.ObjectIdx];
    //    __builtin_prefetch(&Obj + 8, 0, 3);
    //    __builtin_prefetch(&Obj + 16, 0, 3);
    //    __builtin_prefetch(Obj.Base, 0, 3);
    *EncNoPtr = EncodingNo;
    *SizePtr = Obj.ObjSize;
    return Obj.Base;
  }

  char *getBase(char *VPtr) {
    EncTy E(VPtr);
    ASSUME(E.Bits.ObjectIdx < NumObjects);
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
  uint64_t NumObjectsUsed = 0;

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
    uint64_t ObjectIdx = __scoped_atomic_fetch_add(
        &NumObjectsUsed, 1, OrderingTy::relaxed, MemScopeTy::device);
    if (ObjectIdx >= NumObjects) {
      // FPRINTF("out of objects!\n");
      //      __builtin_trap();
    }
    Objects[ObjectIdx] = {MPtr};
    EncTy E(ObjectIdx);
    return E.VPtr;
  }

  void free(char *VPtr) {
    EncTy E(VPtr);
    ASSUME(E.Bits.ObjectIdx < NumObjects);
    Objects[E.Bits.ObjectIdx].Base = 0;
  }

  char *decode(char *VPtr) {
    EncTy E(VPtr);
    ASSUME(E.Bits.ObjectIdx < NumObjects);
    auto *Base = Objects[E.Bits.ObjectIdx];
    EncTy M(Base);
    return Base + E.Bits.Offset - M.Bits.Offset;
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
                           uint8_t *__restrict EncNoPtr) {
    EncTy E(VPtr);
    ASSUME(E.Bits.ObjectIdx < NumObjects);
    if (E.Bits.Magic != Base::MAGIC) {
      return nullptr;
    }
    ObjDescTy &Obj = Objects[E.Bits.ObjectIdx];
    *EncNoPtr = EncodingNo;
    *SizePtr = ObjSize;
    return Obj.Base;
  }

  char *getBase(char *VPtr) {
    EncTy E(VPtr);
    ASSUME(E.Bits.ObjectIdx < NumObjects);
    auto *Base = Objects[E.Bits.ObjectIdx];
    return Base;
  }

  char *getBaseVPtr(char *VPtr) {
    EncTy E(VPtr);
    return VPtr - E.Bits.Offset;
  }
};

using SmallObjectsTy = BucketSchemeTy</*EncodingNo=*/1,
                                      /*OffsetBits=*/12, /*BucketBits=*/3,
                                      /*RealPtrBits=*/32>;
using LargeObjectsTy = LedgerSchemeTy</*EncodingNo=*/2, /*ObjectBits=*/24>;

#ifdef STATS
#ifndef __OBJSAN_DEVICE__
struct StatsTy {
  uint64_t EncNull = 0;
  uint64_t Enc0 = 0;
  uint64_t Enc1 = 0;
  uint64_t Enc2 = 0;
  uint64_t EncX = 0;

  StatsTy(const char *S) : S(S) {}
  ~StatsTy() {
    printf("Stats:\n%s [null: %llu] [0: %llu] [1: %llu] [2: %llu] "
           "[3: %llu]\n",
           S, EncNull, Enc0, Enc1, Enc2, EncX);
  }
  const char *S;
};
extern StatsTy SLoads;
extern StatsTy SStores;
extern StatsTy SRange;
extern StatsTy SLoopR;
#endif
#endif

} // namespace __objsan

extern "C" {
extern __objsan::SmallObjectsTy __objsan_SmallObjects;
extern __objsan::LargeObjectsTy __objsan_LargeObjects;
};

//
#endif // OBJSAN_OBJ_ENCODING_H
